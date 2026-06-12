"""分布式追踪 — 遵循 OpenTelemetry GenAI Semantic Conventions。

Span 层级:
    Trace
    └── ENTRY (user input, session boundary)
        └── AGENT (agent invocation)
            ├── STEP (ReAct round / workflow step)
            │   ├── LLM (model inference)
            │   ├── TOOL (tool execution)
            │   └── RETRIEVER (RAG retrieval)
            ├── GUARDRAIL (safety check)
            └── EVALUATOR (quality scoring)

ref: OpenTelemetry GenAI Semantic Conventions
ref: https://github.com/open-telemetry/semantic-conventions/issues/3540 (Skill Span proposal)
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class SpanKind(str, Enum):
    """OTel GenAI span kind — 与社区标准对齐。"""
    ENTRY = "entry"           # 用户输入边界
    AGENT = "agent"           # Agent 调用
    STEP = "step"             # ReAct 迭代 / 工作流步骤
    LLM = "llm"               # 模型推理
    TOOL = "tool"             # 工具执行
    RETRIEVER = "retriever"   # RAG 检索
    EMBEDDING = "embedding"   # 向量嵌入
    GUARDRAIL = "guardrail"   # 安全检测
    EVALUATOR = "evaluator"   # 质量评估
    RERANKER = "reranker"     # 上下文重排


@dataclass
class Span:
    """单个追踪 Span，记录完整属性。"""

    # 基础身份
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_span_id: str = ""
    session_id: str = ""

    # 类型与元数据
    kind: SpanKind = SpanKind.STEP
    name: str = ""
    agent_id: str = ""
    step_id: str = ""

    # 时间
    start_time: float = 0.0
    end_time: float = 0.0

    # GenAI 属性
    model: str = ""                          # gen_ai.request.model
    input_tokens: int = 0                    # gen_ai.usage.input_tokens
    output_tokens: int = 0                   # gen_ai.usage.output_tokens
    total_tokens: int = 0                    # gen_ai.usage.total_tokens
    finish_reason: str = ""                  # gen_ai.response.finish_reasons
    tool_name: str = ""                      # gen_ai.tool.name
    tool_args: dict | None = None            # gen_ai.tool.arguments
    retriever_query: str = ""                # gen_ai.retriever.query
    retriever_docs: int = 0                  # gen_ai.retriever.document_count
    guardrail_decision: str = ""             # gen_ai.guardrail.decision
    evaluator_score: float | None = None     # gen_ai.evaluator.score
    evaluator_dimension: str = ""            # gen_ai.evaluator.dimension

    # 追踪属性
    round_number: int = 0                    # gen_ai.react.round
    skill_name: str = ""                     # gen_ai.skill.name (proposed semconv #3540)

    # 状态
    status: str = "ok"                       # ok / error
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # 成本（派生自 token 用量）
    cost_usd: float = 0.0

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "kind": self.kind.value,
            "name": self.name,
            "agent_id": self.agent_id,
            "step_id": self.step_id,
            "start_time": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat() if self.end_time else None,
            "duration_ms": round(self.duration_ms, 2),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "retriever_query": self.retriever_query,
            "retriever_docs": self.retriever_docs,
            "guardrail_decision": self.guardrail_decision,
            "evaluator_score": self.evaluator_score,
            "evaluator_dimension": self.evaluator_dimension,
            "round_number": self.round_number,
            "skill_name": self.skill_name,
            "status": self.status,
            "error_message": self.error_message,
            "cost_usd": round(self.cost_usd, 6),
            "metadata": self.metadata,
        }


# 上下文变量 — 当前活跃的 span 栈
_current_trace: ContextVar[str] = ContextVar("trace_id", default="")
_current_session: ContextVar[str] = ContextVar("session_id", default="")
_span_stack: ContextVar[list[Span]] = ContextVar("span_stack", default=[])


@dataclass
class Trace:
    """一次完整的 Trace = 一组 Span 的树。"""
    trace_id: str
    session_id: str
    spans: list[Span] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.spans)

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000 if self.end_time > 0 else 0


class Tracer:
    """分布式追踪器。管理 Span 生命周期和 Trace 聚合。

    用法:
        tracer = get_tracer()
        with tracer.start_span(kind=SpanKind.AGENT, name="research", agent_id="researcher") as span:
            with tracer.start_span(kind=SpanKind.LLM, name="llm_call") as llm_span:
                llm_span.model = "gpt-4"
                llm_span.input_tokens = 100
    """

    def __init__(self) -> None:
        self._traces: dict[str, Trace] = {}
        self._completed_traces: list[Trace] = []
        self._span_listeners: list[callable] = []  # 每个 span 完成时触发

    # ── 上下文管理 ─────────────────────────────────

    @contextmanager
    def start_trace(self, session_id: str = "") -> Any:
        """开始一次新的 Trace。"""
        trace_id = uuid.uuid4().hex[:16]
        token_cv = _current_trace.set(trace_id)
        session_cv = _current_session.set(session_id or trace_id)
        _span_stack.set([])

        trace = Trace(trace_id=trace_id, session_id=session_id or trace_id)
        trace.start_time = time.time()
        self._traces[trace_id] = trace

        try:
            yield trace
        finally:
            trace.end_time = time.time()
            _current_trace.reset(token_cv)
            _current_session.reset(session_cv)
            _span_stack.set([])
            # 移到已完成列表
            self._completed_traces.append(self._traces.pop(trace_id, trace))
            # 如果 completed_traces 太多，清理旧数据（保留最近 1000 条）
            if len(self._completed_traces) > 1000:
                self._completed_traces = self._completed_traces[-500:]

    @contextmanager
    def start_span(
        self,
        *,
        kind: SpanKind = SpanKind.STEP,
        name: str = "",
        agent_id: str = "",
        step_id: str = "",
        model: str = "",
        tool_name: str = "",
        round_number: int = 0,
        skill_name: str = "",
    ) -> Any:
        """在当前 Trace 中开始一个新的 Span。"""
        span = Span(
            trace_id=_current_trace.get(""),
            parent_span_id=self._current_parent_id(),
            session_id=_current_session.get(""),
            kind=kind,
            name=name,
            agent_id=agent_id,
            step_id=step_id,
            model=model,
            tool_name=tool_name,
            round_number=round_number,
            skill_name=skill_name,
            start_time=time.time(),
        )

        stack = list(_span_stack.get())
        stack.append(span)
        _span_stack.set(stack)

        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.error_message = str(e)
            raise
        finally:
            span.end_time = time.time()

            # 添加到 Trace
            trace = self._traces.get(span.trace_id)
            if trace:
                trace.spans.append(span)

            # 通知监听器
            for listener in self._span_listeners:
                try:
                    listener(span)
                except Exception:
                    pass

    # ── Span 辅助 ──────────────────────────────────

    def _current_parent_id(self) -> str:
        stack = _span_stack.get()
        return stack[-1].span_id if stack else ""

    @property
    def current_trace_id(self) -> str:
        return _current_trace.get("")

    @property
    def current_session_id(self) -> str:
        return _current_session.get("")

    def get_current_trace(self) -> Trace | None:
        return self._traces.get(self.current_trace_id)

    # ── 监听器 ─────────────────────────────────────

    def on_span_complete(self, callback: callable) -> None:
        """注册 span 完成时的回调。用于 Metrics 收集、审计日志等。"""
        self._span_listeners.append(callback)

    # ── 查询 ───────────────────────────────────────

    def get_recent_traces(self, limit: int = 20) -> list[Trace]:
        return self._completed_traces[-limit:]

    def get_trace(self, trace_id: str) -> Trace | None:
        """查已完成或进行中的 Trace。"""
        if trace_id in self._completed_traces:
            return next(t for t in self._completed_traces if t.trace_id == trace_id)
        return self._traces.get(trace_id)

    def get_session_traces(self, session_id: str) -> list[Trace]:
        return [t for t in self._completed_traces if t.session_id == session_id]

    def clear(self) -> None:
        self._traces.clear()
        self._completed_traces.clear()


# 模块级单例
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
