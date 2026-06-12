"""Cost Attribution — 成本归因与报告。

ref: Anthropic cost tracking — per-workspace, per-model, real-time aggregation
ref: OpenAI usage API — granular token tracking with cost breakdown

追踪维度:
  - Per agent: 每个 agent 的成本
  - Per workflow: 每个 workflow run 的成本
  - Per model: 每个模型的使用量和成本
  - Per session: 每个会话的成本
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class UsageRecord:
    """单次使用记录。"""
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""
    workflow_id: str = ""
    session_id: str = ""
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostReport:
    """成本报告。"""
    period_start: float = 0.0
    period_end: float = 0.0
    total_cost: float = 0.0
    total_tokens: int = 0
    total_calls: int = 0
    by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_workflow: dict[str, dict[str, Any]] = field(default_factory=dict)
    trends: dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """成本归因追踪器。

    用法:
        tracker = CostTracker()
        tracker.record(UsageRecord(
            agent_id="researcher", model="gpt-4o-mini",
            input_tokens=2000, output_tokens=500, cost_usd=0.0006,
        ))
        report = tracker.generate_report(hours=24)
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._records: list[UsageRecord] = []
        self._max_records = max_records
        # 实时聚合（避免每次都遍历所有记录）
        self._agent_cost: dict[str, float] = defaultdict(float)
        self._agent_tokens: dict[str, int] = defaultdict(int)
        self._agent_calls: dict[str, int] = defaultdict(int)
        self._model_cost: dict[str, float] = defaultdict(float)
        self._model_tokens: dict[str, int] = defaultdict(int)
        self._workflow_cost: dict[str, float] = defaultdict(float)

    # ── 记录 ───────────────────────────────────────

    def record(self, rec: UsageRecord) -> None:
        self._records.append(rec)

        # 聚合
        self._agent_cost[rec.agent_id] += rec.cost_usd
        self._agent_tokens[rec.agent_id] += rec.input_tokens + rec.output_tokens
        self._agent_calls[rec.agent_id] += 1
        self._model_cost[rec.model] += rec.cost_usd
        self._model_tokens[rec.model] += rec.input_tokens + rec.output_tokens
        if rec.workflow_id:
            self._workflow_cost[rec.workflow_id] += rec.cost_usd

        # 环形缓冲区
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records:]

        logger.debug("cost_recorded", agent=rec.agent_id, model=rec.model,
                     cost=f"${rec.cost_usd:.6f}", tokens=rec.input_tokens + rec.output_tokens)

    # ── 报告 ───────────────────────────────────────

    def generate_report(self, hours: int = 24) -> CostReport:
        """生成指定时间窗口的成本报告。"""
        now = time.time()
        cutoff = now - hours * 3600

        window = [r for r in self._records if r.timestamp >= cutoff]
        if not window:
            return CostReport(period_start=cutoff, period_end=now)

        report = CostReport(period_start=cutoff, period_end=now)
        report.total_cost = sum(r.cost_usd for r in window)
        report.total_tokens = sum(r.input_tokens + r.output_tokens for r in window)
        report.total_calls = len(window)

        # 按 agent 聚合
        agent_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "tokens": 0, "calls": 0, "errors": 0})
        for r in window:
            d = agent_data[r.agent_id or "unknown"]
            d["cost"] += r.cost_usd
            d["tokens"] += r.input_tokens + r.output_tokens
            d["calls"] += 1
            if not r.success:
                d["errors"] += 1
        report.by_agent = dict(agent_data)

        # 按模型聚合
        model_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "tokens": 0, "calls": 0})
        for r in window:
            d = model_data[r.model or "unknown"]
            d["cost"] += r.cost_usd
            d["tokens"] += r.input_tokens + r.output_tokens
            d["calls"] += 1
        report.by_model = dict(model_data)

        # 按 workflow 聚合
        wf_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"cost": 0.0, "calls": 0})
        for r in window:
            if r.workflow_id:
                d = wf_data[r.workflow_id]
                d["cost"] += r.cost_usd
                d["calls"] += 1
        report.by_workflow = dict(wf_data)

        # 趋势（按小时分桶）
        buckets: dict[int, float] = defaultdict(float)
        for r in window:
            hour_bucket = int((r.timestamp - cutoff) / 3600)
            buckets[hour_bucket] += r.cost_usd
        report.trends = {
            "hourly_buckets": dict(sorted(buckets.items())),
            "avg_cost_per_call": report.total_cost / report.total_calls if report.total_calls else 0,
            "avg_tokens_per_call": report.total_tokens / report.total_calls if report.total_calls else 0,
        }

        return report

    # ── 查询 ───────────────────────────────────────

    def get_agent_cost(self, agent_id: str) -> dict[str, Any]:
        return {
            "agent_id": agent_id,
            "total_cost": self._agent_cost.get(agent_id, 0.0),
            "total_tokens": self._agent_tokens.get(agent_id, 0),
            "total_calls": self._agent_calls.get(agent_id, 0),
            "avg_cost_per_call": (
                self._agent_cost[agent_id] / self._agent_calls[agent_id]
                if self._agent_calls.get(agent_id, 0) > 0 else 0
            ),
        }

    def get_model_cost(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "total_cost": self._model_cost.get(model, 0.0),
            "total_tokens": self._model_tokens.get(model, 0),
        }

    def get_summary(self) -> dict[str, Any]:
        return {
            "total_cost": sum(self._agent_cost.values()),
            "total_tokens": sum(self._agent_tokens.values()),
            "total_calls": sum(self._agent_calls.values()),
            "by_agent": dict(self._agent_cost),
            "by_model": dict(self._model_cost),
        }

    # ── 管理 ───────────────────────────────────────

    def clear(self) -> None:
        self._records.clear()
        self._agent_cost.clear()
        self._agent_tokens.clear()
        self._agent_calls.clear()
        self._model_cost.clear()
        self._model_tokens.clear()
        self._workflow_cost.clear()

    @property
    def record_count(self) -> int:
        return len(self._records)
