"""回归测试执行器 — 运行 Golden Dataset，支持轨迹评估和软失败阈值。

ref: Apollo ARS — path-dependent + semantic matchers
ref: Monte Carlo "Soft Failure" — tiered scoring
ref: Google Cloud GenAI Evaluation Labs — trajectory evaluation
ref: CI/CD for Evals (Kinde) — GitHub Actions + promptfoo pattern
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.evaluation.golden import GoldenCase, GoldenDataset, GoldenStore
from src.relay.client import get_relay_client
from src.shared.config import get_project_root

logger = structlog.get_logger()


@dataclass
class RegressionResult:
    """单个测试用例的执行结果。"""

    case_id: str
    passed: bool
    score: float = 0.0                    # 0-1 分数
    failure_type: str = ""                 # "hard_fail" / "soft_fail" / ""
    actual_output: str = ""
    expected: str = ""
    duration_ms: float = 0.0
    error: str = ""
    trace_id: str = ""


@dataclass
class RegressionReport:
    """回归测试报告。"""

    dataset_name: str
    total: int
    passed: int
    hard_failures: int
    soft_failures: int
    pass_rate: float = 0.0
    avg_score: float = 0.0
    total_duration_ms: float = 0.0
    results: list[RegressionResult] = field(default_factory=list)
    degraded_from_baseline: bool = False
    degradation_details: list[str] = field(default_factory=list)


class RegressionRunner:
    """回归测试执行器。

    用法:
        runner = RegressionRunner(use_relay=True)
        report = await runner.run("researcher")
    """

    def __init__(self, *, use_relay: bool = False) -> None:
        self._relay = get_relay_client(use_relay=use_relay)
        self._store = GoldenStore()

    # ── 运行 ───────────────────────────────────────

    async def run(
        self,
        agent_id: str,
        *,
        category: str = "",
        dataset_name: str = "",
        limit: int = 0,
    ) -> RegressionReport:
        """对指定 Agent 运行回归测试。"""
        cases: list[GoldenCase] = []

        if dataset_name and category:
            ds = self._store.load(dataset_name, category)
            cases = ds.cases
        else:
            cases = self._store.load_all(agent_id)
            if category:
                cases = [c for c in cases if c.category == category]

        if limit > 0:
            cases = cases[:limit]

        results: list[RegressionResult] = []
        start_time = time.monotonic()

        for case in cases:
            rr = await self._run_case(case)
            results.append(rr)

        total_time = (time.monotonic() - start_time) * 1000

        passed = [r for r in results if r.passed]
        hard_fails = [r for r in results if r.failure_type == "hard_fail"]
        soft_fails = [r for r in results if r.failure_type == "soft_fail"]

        return RegressionReport(
            dataset_name=agent_id,
            total=len(results),
            passed=len(passed),
            hard_failures=len(hard_fails),
            soft_failures=len(soft_fails),
            pass_rate=len(passed) / max(len(results), 1),
            avg_score=sum(r.score for r in results) / max(len(results), 1),
            total_duration_ms=total_time,
            results=results,
        )

    async def run_all(
        self,
        *,
        category: str = "",
        agent_ids: list[str] | None = None,
    ) -> dict[str, RegressionReport]:
        """对所有 Agent 运行回归测试。"""
        if agent_ids is None:
            from src.shared.constants import AGENT_IDS
            agent_ids = sorted(AGENT_IDS - {"main"})

        tasks = {aid: asyncio.create_task(self.run(aid, category=category)) for aid in agent_ids}
        reports: dict[str, RegressionReport] = {}
        for aid, task in tasks.items():
            try:
                reports[aid] = await task
            except Exception as e:
                logger.error("regression_agent_failed", agent_id=aid, error=str(e))
                reports[aid] = RegressionReport(
                    dataset_name=aid, total=0, passed=0, hard_failures=0, soft_failures=0,
                    results=[RegressionResult(
                        case_id="runner_error", passed=False, failure_type="hard_fail",
                        error=str(e),
                    )],
                )
        return reports

    # ── 单个用例 ────────────────────────────────────

    async def _run_case(self, case: GoldenCase) -> RegressionResult:
        """执行单个 Golden Case。"""
        start = time.monotonic()

        try:
            # 调用 Agent
            prompt = case.input.get("message", case.input.get("prompt", ""))
            trace_id = f"regression-{case.id}-{int(time.time())}"

            output = await self._relay.call_agent(
                case.agent_id, prompt, trace_id=trace_id, timeout=60,
            )

            duration = (time.monotonic() - start) * 1000

            # 评分
            score, passed, failure_type = self._evaluate(case, output)

            return RegressionResult(
                case_id=case.id,
                passed=passed,
                score=score,
                failure_type=failure_type,
                actual_output=output[:500],
                expected=json.dumps(case.expected, ensure_ascii=False)[:200],
                duration_ms=duration,
                trace_id=trace_id,
            )

        except Exception as e:
            return RegressionResult(
                case_id=case.id,
                passed=False,
                score=0.0,
                failure_type="hard_fail",
                error=str(e),
                duration_ms=(time.monotonic() - start) * 1000,
            )

    # ── 评分逻辑 ────────────────────────────────────

    def _evaluate(self, case: GoldenCase, output: str) -> tuple[float, bool, str]:
        """按断言类型评估输出。

        Returns: (score, passed, failure_type)
        """
        if case.assertion_type == "exact":
            expected_text = case.expected.get("output", "")
            score = 1.0 if output.strip() == expected_text.strip() else 0.0

        elif case.assertion_type == "contains":
            expected_substrings = case.expected.get("contains", [])
            if isinstance(expected_substrings, str):
                expected_substrings = [expected_substrings]
            hits = sum(1 for s in expected_substrings if s.lower() in output.lower())
            score = hits / max(len(expected_substrings), 1)

        elif case.assertion_type == "norootcause":
            forbidden = case.expected.get("forbidden", [])
            if isinstance(forbidden, str):
                forbidden = [forbidden]
            has_forbidden = any(f.lower() in output.lower() for f in forbidden)
            score = 0.0 if has_forbidden else 1.0

        else:  # semantic — 基于关键词和长度判断
            score = self._semantic_score(case.expected, output)

        # 判定失败类型
        threshold = case.soft_fail_threshold
        if score < threshold:
            return score, False, "hard_fail"
        elif score < 0.8:
            return score, True, "soft_fail"
        else:
            return score, True, ""

    @staticmethod
    def _semantic_score(expected: dict, output: str) -> float:
        """语义评分 — 基于期望条件的关键词匹配。"""
        checks = []
        weights = []

        if expected.get("task_completed") is not None:
            weights.append(0.3)
            # 简单启发式：输出非空且不是错误信息
            ok = len(output) > 20 and "error" not in output.lower()[:200]
            if expected["task_completed"]:
                checks.append(1.0 if ok else 0.0)
            else:
                checks.append(0.0 if ok else 1.0)

        if expected.get("no_hallucination") is not None:
            weights.append(0.3)
            hallucination_markers = [
                "据不可靠消息", "据匿名人士", "据悉", "据内部人士",
                "i made this up", "i don't know but i'll guess",
            ]
            has_hallucination = any(m in output.lower() for m in hallucination_markers)
            checks.append(0.0 if has_hallucination else 1.0)

        if expected.get("contains_keywords") is not None:
            weights.append(0.2)
            kws = expected["contains_keywords"]
            if isinstance(kws, str):
                kws = [kws]
            hits = sum(1 for kw in kws if kw.lower() in output.lower())
            checks.append(hits / max(len(kws), 1))

        if expected.get("min_length") is not None:
            weights.append(0.2)
            min_len = expected["min_length"]
            checks.append(min(len(output) / max(min_len, 1), 2.0) / 2.0)

        if not checks:
            # 默认：输出非空且长度合理
            return 0.8 if len(output) > 20 else 0.2

        # 加权平均
        total_weight = sum(weights) or 1
        return sum(c * w for c, w in zip(checks, weights)) / total_weight

    # ── 基线对比 ────────────────────────────────────

    def compare_baseline(self, report: RegressionReport, baseline: dict) -> list[str]:
        """对比基线，返回退化详情。"""
        degradations: list[str] = []
        prev_rate = baseline.get("pass_rate", 0)

        if report.pass_rate < prev_rate - 0.05:
            report.degraded_from_baseline = True
            degradations.append(
                f"Pass rate dropped from {prev_rate:.1%} to {report.pass_rate:.1%}"
            )

        return degradations

    @staticmethod
    def save_baseline(agent_id: str, report: RegressionReport) -> None:
        """保存基线数据。"""
        baselines_dir = get_project_root() / "config" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)
        path = baselines_dir / f"{agent_id}_regression.json"
        path.write_text(json.dumps({
            "agent_id": agent_id,
            "pass_rate": report.pass_rate,
            "avg_score": report.avg_score,
            "total_cases": report.total,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2, ensure_ascii=False))

    @staticmethod
    def load_baseline(agent_id: str) -> dict:
        """加载基线数据。"""
        path = get_project_root() / "config" / "baselines" / f"{agent_id}_regression.json"
        if path.exists():
            return json.loads(path.read_text())
        return {}
