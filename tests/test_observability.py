"""可观测性模块测试。"""

from src.observability.tracer import Span, SpanKind, Tracer
from src.observability.metrics import MetricsCollector


class TestSpan:
    def test_create_span(self):
        span = Span(kind=SpanKind.LLM, name="test")
        assert span.kind == SpanKind.LLM
        assert span.name == "test"
        assert span.span_id
        assert span.status == "ok"

    def test_span_to_dict(self):
        span = Span(kind=SpanKind.LLM, name="test", model="gpt-4o", input_tokens=100, output_tokens=50, total_tokens=150)
        d = span.to_dict()
        assert d["kind"] == "llm"
        assert d["model"] == "gpt-4o"
        assert d["input_tokens"] == 100
        assert d["total_tokens"] == 150

    def test_span_duration(self):
        span = Span(start_time=1000.0, end_time=1001.5)
        assert span.duration_ms == 1500.0  # (1001.5 - 1000.0) * 1000


class TestTracer:
    def test_start_trace(self):
        tracer = Tracer()
        with tracer.start_trace(session_id="s1") as trace:
            assert trace.trace_id
            assert trace.session_id == "s1"
        traces = tracer.get_recent_traces(5)
        assert len(traces) == 1

    def test_nested_spans(self):
        tracer = Tracer()
        with tracer.start_trace():
            with tracer.start_span(kind=SpanKind.AGENT, name="agent") as s1:
                with tracer.start_span(kind=SpanKind.LLM, name="llm", model="gpt-4o") as s2:
                    s2.input_tokens = 100
                    s2.output_tokens = 50
                    s2.total_tokens = 150
                    s2.cost_usd = 0.001
        traces = tracer.get_recent_traces(5)
        assert len(traces) == 1
        assert len(traces[0].spans) == 2

    def test_span_error(self):
        tracer = Tracer()
        try:
            with tracer.start_trace():
                with tracer.start_span(kind=SpanKind.STEP, name="will_fail") as s:
                    raise ValueError("test error")
        except ValueError:
            pass
        traces = tracer.get_recent_traces(5)
        assert traces[0].spans[0].status == "error"
        assert "test error" in traces[0].spans[0].error_message

    def test_clear(self):
        tracer = Tracer()
        with tracer.start_trace():
            with tracer.start_span(kind=SpanKind.STEP):
                pass
        assert len(tracer.get_recent_traces(5)) == 1
        tracer.clear()
        assert len(tracer.get_recent_traces(5)) == 0


class TestMetrics:
    def test_calculate_cost_known_model(self):
        cost = MetricsCollector.calculate_cost("gpt-4o", 1_000_000, 1_000_000)
        assert cost == 12.50  # $2.50 + $10.00

    def test_calculate_cost_unknown_model(self):
        cost = MetricsCollector.calculate_cost("unknown-model", 1_000_000, 1_000_000)
        assert cost == 5.00  # default $1.00 + $4.00

    def test_snapshot_empty(self):
        mc = MetricsCollector()
        snap = mc.snapshot()
        assert snap.total_llm_calls == 0
        assert snap.total_cost_usd == 0.0

    def test_on_span_llm(self):
        mc = MetricsCollector()
        span = Span(kind=SpanKind.LLM, name="test", agent_id="researcher",
                     model="gpt-4o", input_tokens=500, output_tokens=200,
                     total_tokens=700, cost_usd=0.00325)
        span.end_time = span.start_time + 1.0  # 1000ms
        mc.on_span(span)
        snap = mc.snapshot()
        assert snap.total_llm_calls == 1
        assert snap.total_input_tokens == 500
        assert snap.total_output_tokens == 200
        assert snap.total_cost_usd == 0.00325
        assert "researcher" in snap.by_agent

    def test_on_span_error(self):
        mc = MetricsCollector()
        span = Span(kind=SpanKind.TOOL, name="bad_tool", status="error")
        mc.on_span(span)
        assert mc.total_errors == 1

    def test_get_slo_status(self):
        mc = MetricsCollector()
        # empty → ok
        slo = mc.get_slo_status()
        assert slo["llm_latency_slo"] == "ok"
        assert slo["error_rate_slo"] == "ok"
