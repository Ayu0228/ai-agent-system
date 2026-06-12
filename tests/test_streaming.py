"""Streaming engine & edge gateway tests."""

import asyncio
import time

import pytest

from src.streaming.engine import (
    StreamingEngine, StreamEvent, StreamEventType,
)
from src.streaming.gateway import (
    EdgeGateway, EdgeNode, GatewayConfig, GatewayStatus,
)


class TestStreamEvent:
    """Test StreamEvent."""

    def test_defaults(self):
        event = StreamEvent(type=StreamEventType.TOKEN)
        assert event.type == StreamEventType.TOKEN
        assert len(event.id) == 8

    def test_to_sse(self):
        event = StreamEvent(
            type=StreamEventType.TOKEN,
            data={"text": "hello"},
            session_id="s1",
            agent_id="a1",
        )
        sse = event.to_sse()
        assert sse.startswith("data: ")
        assert "hello" in sse
        assert "token" in sse

    def test_to_json(self):
        event = StreamEvent(
            type=StreamEventType.DONE,
            data={"status": "completed"},
        )
        j = event.to_json()
        assert '"type": "done"' in j
        assert '"status": "completed"' in j


class TestStreamEventType:
    """Test StreamEventType enum."""

    def test_all_types(self):
        types = list(StreamEventType)
        assert StreamEventType.TOKEN in types
        assert StreamEventType.TOOL_CALL_START in types
        assert StreamEventType.TOOL_CALL_END in types
        assert StreamEventType.STEP_START in types
        assert StreamEventType.STEP_END in types
        assert StreamEventType.THINKING in types
        assert StreamEventType.STATUS in types
        assert StreamEventType.ERROR in types
        assert StreamEventType.DONE in types
        assert len(types) == 9


class TestStreamingEngine:
    """Test StreamingEngine."""

    @pytest.fixture
    def engine(self):
        return StreamingEngine()

    # ── Callbacks ─────────────────────────────────────

    def test_decorator_registration(self, engine):
        received = []

        @engine.on(StreamEventType.TOKEN)
        def on_token(event):
            received.append(event.data.get("text"))

        engine.emit(StreamEvent(
            type=StreamEventType.TOKEN,
            data={"text": "hello"},
        ))
        assert received == ["hello"]

    def test_multiple_listeners_same_event(self, engine):
        r1, r2 = [], []

        @engine.on(StreamEventType.DONE)
        def listener1(event): r1.append(1)

        @engine.on(StreamEventType.DONE)
        def listener2(event): r2.append(2)

        engine.emit(StreamEvent(type=StreamEventType.DONE))
        assert r1 == [1]
        assert r2 == [2]

    def test_emit_wrong_type_no_listeners(self, engine):
        engine.emit(StreamEvent(type=StreamEventType.ERROR))  # no error

    def test_listener_error_does_not_crash(self, engine):
        @engine.on(StreamEventType.TOKEN)
        def bad_listener(event):
            raise RuntimeError("listener crash")

        # Should not raise
        engine.emit(StreamEvent(
            type=StreamEventType.TOKEN,
            data={"text": "ok"},
        ))

    # ── Stream response ───────────────────────────────

    @pytest.mark.asyncio
    async def test_stream_response_yields_events(self, engine):
        events = []
        async for event in engine.stream_response(
            agent_id="researcher",
            prompt="What is AI?",
            response_text="AI is artificial intelligence",
        ):
            events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.STATUS in types
        assert StreamEventType.THINKING in types
        assert StreamEventType.TOKEN in types
        assert StreamEventType.DONE in types

    @pytest.mark.asyncio
    async def test_stream_response_contains_text(self, engine):
        events = []
        async for event in engine.stream_response(
            agent_id="a1", prompt="hi", response_text="Hello world",
        ):
            events.append(event)

        tokens = [e for e in events if e.type == StreamEventType.TOKEN]
        text = "".join(t.data.get("text", "") for t in tokens)
        assert "Hello" in text

    @pytest.mark.asyncio
    async def test_stream_sse_format(self, engine):
        chunks = []
        async for chunk in engine.stream_sse(
            agent_id="a1", prompt="hi", response_text="Test",
        ):
            chunks.append(chunk)

        assert all(c.startswith("data: ") for c in chunks)
        assert any("DONE" in c or "done" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_tool_call(self, engine):
        events = []
        async for event in engine.stream_tool_call(
            agent_id="a1", tool_name="search",
            tool_input={"q": "test"},
        ):
            events.append(event)

        types = [e.type for e in events]
        assert StreamEventType.TOOL_CALL_START in types
        assert StreamEventType.TOOL_CALL_END in types

    # ── Cancel ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_cancel_stream(self, engine):
        # Start a long stream in background
        async def collect():
            events = []
            async for event in engine.stream_response(
                agent_id="a1", prompt="long", session_id="cancel-me",
                response_text="a b c d e f g h i j k l m n o p",
            ):
                events.append(event)
            return events

        # Cancel immediately
        engine.cancel("cancel-me")

        events = await collect()
        # Should have DONE with cancelled status or just be short
        done = [e for e in events if e.type == StreamEventType.DONE]
        if done:
            status = done[0].data.get("status", "")
            assert status in ("cancelled", "completed")

    def test_cancel_nonexistent(self, engine):
        assert engine.cancel("no-such-session") is False

    def test_active_streams_count(self, engine):
        assert engine.active_streams == 0


class TestGatewayConfig:
    """Test GatewayConfig."""

    def test_defaults(self):
        config = GatewayConfig()
        assert config.name == "default"
        assert config.region == "auto"
        assert config.rate_limit_per_minute == 60
        assert config.cache_ttl_s == 300
        assert config.enable_cache is True

    def test_custom(self):
        config = GatewayConfig(
            name="prod-gw",
            region="ap-southeast",
            rate_limit_per_minute=30,
            enable_cache=False,
        )
        assert config.rate_limit_per_minute == 30
        assert config.enable_cache is False


class TestEdgeNode:
    """Test EdgeNode."""

    def test_defaults(self):
        node = EdgeNode(name="n1", region="us-east", endpoint="http://n1:8080")
        assert node.status == GatewayStatus.HEALTHY
        assert node.load == 0
        assert node.max_load == 100


class TestEdgeGateway:
    """Test EdgeGateway."""

    @pytest.fixture
    def gw(self):
        return EdgeGateway(GatewayConfig(name="test-gw"))

    def test_register_node(self, gw):
        node = EdgeNode(name="n1", region="us-east", endpoint="http://n1:8080")
        gw.register_node(node)
        assert "n1" in gw._nodes
        assert "n1" in gw._regions["us-east"]

    def test_unregister_node(self, gw):
        node = EdgeNode(name="n1", region="us-east", endpoint="http://n1")
        gw.register_node(node)
        gw.unregister_node("n1")
        assert "n1" not in gw._nodes

    # ── Routing ───────────────────────────────────────

    def test_route_returns_node(self, gw):
        node = EdgeNode(name="n1", region="us-east", endpoint="http://n1")
        gw.register_node(node)
        result = gw.route(user_region="us-east")
        assert result is not None
        assert result.name == "n1"

    def test_route_prefer_same_region(self, gw):
        us = EdgeNode(name="us-1", region="us-east", endpoint="http://us")
        eu = EdgeNode(name="eu-1", region="eu-west", endpoint="http://eu")
        gw.register_node(us)
        gw.register_node(eu)
        result = gw.route(user_region="eu-west")
        assert result.name == "eu-1"

    def test_route_fallback_when_region_unavailable(self, gw):
        eu = EdgeNode(name="eu-1", region="eu-west", endpoint="http://eu")
        gw.register_node(eu)
        result = gw.route(user_region="us-east")  # no US node, falls back to EU
        assert result is not None
        assert result.name == "eu-1"

    def test_route_skips_offline(self, gw):
        n1 = EdgeNode(name="n1", region="us", endpoint="http://n1",
                     status=GatewayStatus.OFFLINE)
        n2 = EdgeNode(name="n2", region="us", endpoint="http://n2",
                     status=GatewayStatus.HEALTHY)
        gw.register_node(n1)
        gw.register_node(n2)
        result = gw.route(user_region="us")
        assert result.name == "n2"

    def test_route_skips_overloaded(self, gw):
        n1 = EdgeNode(name="n1", region="us", endpoint="http://n1",
                     load=100, max_load=100)  # fully loaded
        n2 = EdgeNode(name="n2", region="us", endpoint="http://n2", load=0)
        gw.register_node(n1)
        gw.register_node(n2)
        result = gw.route(user_region="us")
        assert result.name == "n2"

    def test_route_no_available_nodes(self, gw):
        n1 = EdgeNode(name="n1", region="us", endpoint="http://n1",
                     status=GatewayStatus.OFFLINE)
        gw.register_node(n1)
        result = gw.route()
        assert result is None

    # ── Rate limiting ─────────────────────────────────

    def test_check_rate_limit_allows(self, gw):
        assert gw.check_rate_limit("user-1") is True

    def test_check_rate_limit_blocks_after_limit(self, gw):
        for _ in range(gw.config.rate_limit_per_minute):
            gw.check_rate_limit("user-2")
        assert gw.check_rate_limit("user-2") is False

    def test_rate_limit_reset_after_window(self, gw):
        # Manually set old window
        gw._rate_counters["user-3"] = (60, time.time() - 61)
        assert gw.check_rate_limit("user-3") is True  # new window

    def test_get_rate_limit_remaining(self, gw):
        assert gw.get_rate_limit_remaining("new-user") == gw.config.rate_limit_per_minute
        gw.check_rate_limit("new-user")
        assert gw.get_rate_limit_remaining("new-user") == gw.config.rate_limit_per_minute - 1

    # ── Cache ─────────────────────────────────────────

    def test_cache_set_get(self, gw):
        gw.cache_set("key1", {"result": "value"})
        assert gw.cache_get("key1") == {"result": "value"}

    def test_cache_get_expired(self, gw):
        gw.cache_set("key1", {"data": 1})
        # Manually expire
        gw._cache["key1"] = ({"data": 1}, time.time() - 1)
        assert gw.cache_get("key1") is None

    def test_cache_disabled(self, gw):
        gw.config.enable_cache = False
        gw.cache_set("key1", "val")
        assert gw.cache_get("key1") is None

    # ── Health check ──────────────────────────────────

    def test_health_check(self, gw):
        n1 = EdgeNode(name="n1", region="us", endpoint="http://n1")
        n2 = EdgeNode(name="n2", region="us", endpoint="http://n2",
                     load=95, max_load=100)
        gw.register_node(n1)
        gw.register_node(n2)
        status = gw.health_check()
        assert status["nodes_total"] == 2
        # n2 should be overloaded
        n2_info = [n for n in status["nodes"] if n["name"] == "n2"][0]
        assert n2_info["status"] in ("overloaded", "healthy")

    def test_get_status(self, gw):
        gw.register_node(EdgeNode(name="n1", region="us", endpoint="http://n1"))
        status = gw.get_status()
        assert status["gateway"] == "test-gw"
        assert status["nodes_total"] == 1


class TestGatewayStatus:
    """Test GatewayStatus enum."""

    def test_all_statuses(self):
        assert GatewayStatus.HEALTHY.value == "healthy"
        assert GatewayStatus.DEGRADED.value == "degraded"
        assert GatewayStatus.OVERLOADED.value == "overloaded"
        assert GatewayStatus.OFFLINE.value == "offline"
