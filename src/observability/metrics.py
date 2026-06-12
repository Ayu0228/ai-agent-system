"""指标收集器 — Token 用量、延迟、错误率、成本归因。

ref: TokenTrimmer — Apache 2.0 cost layer with multi-layer caching
ref: Claude-Code-LLM-Router — agent resource budgeting pattern
ref: ParetoBandit — arXiv 2604.00136, budget-paced adaptive routing
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.observability.tracer import Span, SpanKind


# 模型定价 (USD / 1M tokens) — ref: OpenAI, Anthropic, DeepSeek 官方定价 2026
_MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    # DeepSeek
    "deepseek-v4-pro": {"input": 0.55, "output": 2.19},
    "deepseek-v3": {"input": 0.27, "output": 1.10},
    # MoMo
    "mimo-v2.5-pro": {"input": 0.70, "output": 1.40},
    "mimo-v2.5": {"input": 0.35, "output": 0.70},
}

# 默认定价（未知模型）
_DEFAULT_PRICE = {"input": 1.00, "output": 4.00}


@dataclass
class MetricsSnapshot:
    """某一时刻的指标快照。"""
    timestamp: float = 0.0

    # Token 用量
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # 调用计数
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_errors: int = 0

    # 延迟 (ms)
    llm_latency_p50: float = 0.0
    llm_latency_p95: float = 0.0
    llm_latency_p99: float = 0.0

    # 成本 (USD)
    total_cost_usd: float = 0.0

    # 按维度分组
    by_agent: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """按 Agent 聚合的指标。"""
    agent_id: str
    llm_calls: int = 0
    tool_calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    span_count: int = 0


class MetricsCollector:
    """指标收集器。注册为 Tracer 的 span 监听器，实时聚合。

    用法:
        tracer = get_tracer()
        metrics = get_metrics()
        tracer.on_span_complete(metrics.on_span)
    """

    def __init__(self) -> None:
        self._llm_latencies: list[float] = []  # 最近 1000 次 LLM 调用延迟
        self._reset_counters()
        self._agent_metrics: dict[str, AgentMetrics] = {}

    def _reset_counters(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_llm_calls = 0
        self.total_tool_calls = 0
        self.total_errors = 0
        self.total_cost = 0.0
        self.start_time = time.time()

    # ── Span 监听 ──────────────────────────────────

    def on_span(self, span: Span) -> None:
        """span 完成时调用。"""
        # Token 计数
        if span.kind == SpanKind.LLM:
            self.total_input_tokens += span.input_tokens
            self.total_output_tokens += span.output_tokens
            self.total_llm_calls += 1
            if span.duration_ms > 0:
                self._llm_latencies.append(span.duration_ms)
                if len(self._llm_latencies) > 1000:
                    self._llm_latencies = self._llm_latencies[-1000:]

        elif span.kind == SpanKind.TOOL:
            self.total_tool_calls += 1

        # 错误计数
        if span.status == "error":
            self.total_errors += 1

        # 成本归因
        if span.cost_usd > 0:
            self.total_cost += span.cost_usd

        # Agent 维度
        if span.agent_id:
            if span.agent_id not in self._agent_metrics:
                self._agent_metrics[span.agent_id] = AgentMetrics(agent_id=span.agent_id)
            am = self._agent_metrics[span.agent_id]
            am.agent_id = span.agent_id
            am.span_count += 1
            if span.kind == SpanKind.LLM:
                am.llm_calls += 1
                am.input_tokens += span.input_tokens
                am.output_tokens += span.output_tokens
                am.total_latency_ms += span.duration_ms
            elif span.kind == SpanKind.TOOL:
                am.tool_calls += 1
            if span.status == "error":
                am.errors += 1
            am.total_cost += span.cost_usd

    # ── 成本计算 ───────────────────────────────────

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """计算单次 LLM 调用成本。"""
        pricing = _MODEL_PRICING.get(model, _DEFAULT_PRICE)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    @classmethod
    def estimate_cost_for_model(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """预估模型成本（不创建 span）。"""
        return cls.calculate_cost(model, input_tokens, output_tokens)

    # ── 查询 ───────────────────────────────────────

    def snapshot(self) -> MetricsSnapshot:
        """获取当前指标快照。"""
        latencies = sorted(self._llm_latencies) if self._llm_latencies else [0]
        n = len(latencies)

        return MetricsSnapshot(
            timestamp=time.time(),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_tokens=self.total_input_tokens + self.total_output_tokens,
            total_llm_calls=self.total_llm_calls,
            total_tool_calls=self.total_tool_calls,
            total_errors=self.total_errors,
            llm_latency_p50=latencies[int(n * 0.50)] if n > 0 else 0,
            llm_latency_p95=latencies[int(n * 0.95)] if n > 1 else 0,
            llm_latency_p99=latencies[int(n * 0.99)] if n > 2 else 0,
            total_cost_usd=self.total_cost,
            by_agent={
                aid: {
                    "llm_calls": am.llm_calls,
                    "tool_calls": am.tool_calls,
                    "errors": am.errors,
                    "input_tokens": am.input_tokens,
                    "output_tokens": am.output_tokens,
                    "total_cost": am.total_cost,
                    "avg_latency_ms": am.total_latency_ms / am.llm_calls if am.llm_calls > 0 else 0,
                }
                for aid, am in self._agent_metrics.items()
            },
            by_model={},  # 需要从 span 监听中积累
        )

    def get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        return self._agent_metrics.get(agent_id, AgentMetrics(agent_id=agent_id))

    def get_slo_status(self) -> dict[str, Any]:
        """SLO 状态检查。

        ref: Google SRE Book — SLO targets for AI agents
        """
        snap = self.snapshot()
        return {
            "llm_latency_p95_ms": snap.llm_latency_p95,
            "llm_latency_slo": "ok" if snap.llm_latency_p95 < 2000 else ("warn" if snap.llm_latency_p95 < 2500 else "breach"),
            "error_rate": snap.total_errors / max(snap.total_llm_calls + snap.total_tool_calls, 1),
            "error_rate_slo": "ok" if snap.total_errors / max(snap.total_llm_calls + snap.total_tool_calls, 1) < 0.01 else "breach",
            "total_cost_usd": snap.total_cost_usd,
        }

    def reset(self) -> None:
        self._reset_counters()
        self._llm_latencies.clear()
        self._agent_metrics.clear()


_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics
