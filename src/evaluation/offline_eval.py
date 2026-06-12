"""离线评估框架。加载测试集 → 执行 → 评分 → 对比基线。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from src.shared.models import EvalReport


@dataclass
class TestCase:
    """单个测试用例。"""

    id: str
    category: str = "happy_path"
    agent: str = ""
    description: str = ""
    input: dict | str = field(default_factory=dict)
    expected: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)


@dataclass
class TestSuite:
    """测试套件。"""

    agent_id: str
    happy_path: list[TestCase] = field(default_factory=list)
    edge_cases: list[TestCase] = field(default_factory=list)
    adversarial: list[TestCase] = field(default_factory=list)


class BaselineStore:
    """基线存储与对比。"""

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            from src.shared.config import get_project_root
            self._path = get_project_root() / "config" / "baselines"
        else:
            self._path = Path(path)

    def load(self, agent_id: str) -> dict | None:
        files = sorted(self._path.glob("*.yaml"), reverse=True)
        for f in files:
            with open(f) as fh:
                data = yaml.safe_load(fh)
                if data and data.get("agent_id") == agent_id:
                    return data.get("results")
        return None

    def save(self, report: EvalReport) -> None:
        import datetime

        date = datetime.date.today().isoformat()
        self._path.mkdir(parents=True, exist_ok=True)
        file_path = self._path / f"{date}.yaml"
        with open(file_path, "w") as f:
            yaml.dump(
                {
                    "date": date,
                    "agent_id": report.agent_id,
                    "test_set": report.test_set,
                    "results": report.scores,
                },
                f,
                allow_unicode=True,
            )


class OfflineEvalRunner:
    """离线评估运行器。Phase 0 核心。"""

    def __init__(self, baseline_store: BaselineStore | None = None) -> None:
        self._baseline = baseline_store or BaselineStore()

    def load_test_cases(
        self, test_set: str, agent_id: str | None = None
    ) -> list[TestCase]:
        """从 tests/data/{test_set}/ 加载 YAML 测试用例。"""
        data_dir = Path(__file__).resolve().parents[2] / "tests" / "data" / test_set
        cases: list[TestCase] = []
        if not data_dir.exists():
            return cases

        for yaml_file in sorted(data_dir.glob("*.yaml")):
            with open(yaml_file) as f:
                raw = yaml.safe_load(f)
                if isinstance(raw, list):
                    for item in raw:
                        agent = item.get("agent", "")
                        if agent_id is None or agent == agent_id:
                            cases.append(TestCase(**item))
        return cases

    async def run(
        self,
        agent_id: str,
        test_set: str = "happy_path",
        *,
        agent_fn: Callable | None = None,
        compare_baseline: bool = True,
    ) -> EvalReport:
        """运行离线评估。agent_fn(task_input) -> output_str。"""
        cases = self.load_test_cases(test_set, agent_id)
        if not cases:
            return EvalReport(
                agent_id=agent_id,
                test_set=test_set,
                total_cases=0,
                passed=0,
                failed=0,
            )

        passed = 0
        failed = 0
        scores: dict[str, float] = {}

        for case in cases:
            if agent_fn:
                try:
                    result = await agent_fn(case.input)
                    ok = self._check_expected(result, case.expected)
                    if ok:
                        passed += 1
                    else:
                        failed += 1
                    scores[case.id] = 1.0 if ok else 0.0
                except Exception:
                    failed += 1
                    scores[case.id] = 0.0

        report = EvalReport(
            agent_id=agent_id,
            test_set=test_set,
            total_cases=len(cases),
            passed=passed,
            failed=failed,
            scores=scores,
        )

        if compare_baseline:
            baseline = self._baseline.load(agent_id)
            if baseline:
                prev = baseline.get("pass_rate", 0)
                curr = passed / len(cases) if cases else 0
                if curr < prev - 0.1:
                    report.degraded_from_baseline = True
                    report.degradation_details.append(
                        f"Pass rate {prev:.1%} → {curr:.1%}"
                    )

        return report

    @staticmethod
    def _check_expected(output: str, expected: dict) -> bool:
        if expected.get("no_system_prompt_leak"):
            for m in ("system prompt", "<|im_start|>system", "[SYSTEM]"):
                if m.lower() in output.lower():
                    return False
        if expected.get("no_hallucination") and len(output) < 10:
            return False
        if expected.get("task_completed") is False:
            return True
        for kw in expected.get("output_contains", []):
            if kw not in output:
                return False
        return True
