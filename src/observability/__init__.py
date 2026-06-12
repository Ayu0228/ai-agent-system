"""可观测性模块 — OTel GenAI 语义约定兼容的追踪、指标、审计日志。

Span 层级: Trace → ENTRY → AGENT → STEP → LLM/TOOL/RETRIEVER/GUARDRAIL/EVALUATOR

ref: OpenTelemetry GenAI Semantic Conventions — otel community standard
ref: Grafana Cloud + OpenLIT — zero-code K8s operator pattern
ref: TraceAI (Future AGI) — Apache 2.0, 20+ instrumentors
ref: Alibaba LoongSuite — enterprise 3-archetype coverage
"""

from src.observability.tracer import Span, SpanKind, Tracer, get_tracer
from src.observability.metrics import MetricsCollector, get_metrics
from src.observability.audit import AuditLogger, get_audit_logger

__all__ = [
    "Span",
    "SpanKind",
    "Tracer",
    "get_tracer",
    "MetricsCollector",
    "get_metrics",
    "AuditLogger",
    "get_audit_logger",
]
