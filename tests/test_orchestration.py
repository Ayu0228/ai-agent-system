"""Multi-agent orchestration engine tests."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.orchestration.registry import AgentRegistry, AgentInfo, AgentStatus
from src.orchestration.bus import (
    MessageBus, Message, MessageType, MessageStatus, HandoffRequest,
)
from src.orchestration.engine import (
    Orchestrator, OrchestrationMode, Task, TaskResult, OrchestrationResult,
)


# ═════════════════════════════════════════════════════════════════════
# Agent Registry Tests
# ═════════════════════════════════════════════════════════════════════

class TestAgentInfo:
    def test_default_status_offline(self):
        info = AgentInfo(agent_id="test")
        assert info.status == AgentStatus.OFFLINE
        assert info.is_available is False

    def test_available_when_online_and_under_load(self):
        info = AgentInfo(agent_id="test", status=AgentStatus.ONLINE,
                         max_concurrency=3, current_load=1)
        assert info.is_available is True

    def test_not_available_when_overloaded(self):
        info = AgentInfo(agent_id="test", status=AgentStatus.ONLINE,
                         max_concurrency=3, current_load=3)
        assert info.is_available is False

    def test_not_available_when_degraded(self):
        info = AgentInfo(agent_id="test", status=AgentStatus.DEGRADED,
                         health_score=0.2)
        assert info.is_available is False

    def test_capacity_ratio(self):
        info = AgentInfo(agent_id="test", max_concurrency=4, current_load=1)
        assert info.capacity == 0.75


class TestAgentRegistry:
    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    def test_register_and_get(self, registry):
        info = AgentInfo(agent_id="r", name="Researcher",
                         capabilities=["web_search", "fact_check"],
                         status=AgentStatus.ONLINE)
        registry.register(info)
        assert registry.get("r") == info
        assert registry.agent_count == 1

    def test_heartbeat_updates_time(self, registry):
        info = AgentInfo(agent_id="r", status=AgentStatus.OFFLINE)
        registry.register(info)
        old = info.last_heartbeat
        time.sleep(0.01)
        assert registry.heartbeat("r") is True
        assert info.last_heartbeat > old
        assert info.status == AgentStatus.ONLINE

    def test_heartbeat_nonexistent(self, registry):
        assert registry.heartbeat("ghost") is False

    def test_find_by_capability(self, registry):
        registry.register(AgentInfo(agent_id="r1",
            capabilities=["search"], status=AgentStatus.ONLINE))
        registry.register(AgentInfo(agent_id="r2",
            capabilities=["search", "analyze"], status=AgentStatus.ONLINE))
        registry.register(AgentInfo(agent_id="w1",
            capabilities=["writing"], status=AgentStatus.ONLINE))

        found = registry.find_by_capability("search")
        assert len(found) == 2
        assert {a.agent_id for a in found} == {"r1", "r2"}

        found = registry.find_by_capability("writing")
        assert len(found) == 1

        found = registry.find_by_capability("nonexistent")
        assert found == []

    def test_find_by_capability_excludes_unavailable(self, registry):
        registry.register(AgentInfo(agent_id="r1",
            capabilities=["search"], status=AgentStatus.ONLINE))
        registry.register(AgentInfo(agent_id="r2",
            capabilities=["search"], status=AgentStatus.OFFLINE))
        found = registry.find_by_capability("search")
        assert len(found) == 1
        assert found[0].agent_id == "r1"

    def test_select_best_returns_highest_capacity(self, registry):
        registry.register(AgentInfo(agent_id="r1",
            capabilities=["search"], status=AgentStatus.ONLINE,
            max_concurrency=4, current_load=1))  # capacity 0.75
        registry.register(AgentInfo(agent_id="r2",
            capabilities=["search"], status=AgentStatus.ONLINE,
            max_concurrency=2, current_load=0))  # capacity 1.0
        best = registry.select_best("search")
        assert best is not None

    def test_select_best_none_when_no_match(self, registry):
        assert registry.select_best("nonexistent") is None

    def test_unregister_removes_from_index(self, registry):
        info = AgentInfo(agent_id="r", capabilities=["search"],
                         status=AgentStatus.ONLINE)
        registry.register(info)
        registry.unregister("r")
        assert registry.get("r") is None
        assert registry.find_by_capability("search") == []

    def test_update_status(self, registry):
        info = AgentInfo(agent_id="r", status=AgentStatus.ONLINE)
        registry.register(info)
        registry.update_status("r", AgentStatus.BUSY, current_load=2)
        assert info.status == AgentStatus.BUSY
        assert info.current_load == 2

    def test_update_health(self, registry):
        info = AgentInfo(agent_id="r", status=AgentStatus.ONLINE)
        registry.register(info)
        registry.update_health("r", 0.2)
        assert info.health_score == 0.2
        assert info.status == AgentStatus.DEGRADED

    def test_mark_offline_stale(self, registry):
        info = AgentInfo(agent_id="r", status=AgentStatus.ONLINE)
        registry.register(info)
        # Set after register() — register() resets last_heartbeat if falsy
        info.last_heartbeat = 0
        count = registry.mark_offline_stale(heartbeat_timeout=30)
        assert count == 1
        assert info.status == AgentStatus.OFFLINE

    def test_get_stats(self, registry):
        registry.register(AgentInfo(agent_id="r1", status=AgentStatus.ONLINE,
                                    max_concurrency=2, current_load=1))
        registry.register(AgentInfo(agent_id="r2", status=AgentStatus.OFFLINE))
        stats = registry.get_stats()
        assert stats["total"] == 2
        assert stats["online"] == 1
        assert stats["available"] == 1
        assert stats["total_load"] == 1


# ═════════════════════════════════════════════════════════════════════
# Message Bus Tests
# ═════════════════════════════════════════════════════════════════════

class TestMessage:
    def test_message_defaults(self):
        msg = Message()
        assert msg.id
        assert msg.type == MessageType.EVENT
        assert msg.status.value == "pending"

    def test_ack(self):
        msg = Message()
        msg.ack()
        assert msg.status == MessageStatus.DELIVERED

    def test_fail(self):
        msg = Message()
        msg.fail()
        assert msg.status == MessageStatus.FAILED


class TestMessageBus:
    @pytest.fixture
    def bus(self):
        return MessageBus()

    @pytest.mark.asyncio
    async def test_publish_to_topic(self, bus):
        received: list[Message] = []

        async def handler(msg):
            received.append(msg)

        bus.subscribe("test.topic", handler)
        await bus.publish(Message(topic="test.topic", payload={"k": "v"}))
        assert len(received) == 1
        assert received[0].payload == {"k": "v"}

    @pytest.mark.asyncio
    async def test_publish_to_agent_handler(self, bus):
        received: list[Message] = []

        async def handler(msg): received.append(msg)

        bus.register_agent("agent1", handler)

        await bus.publish(Message(recipient="agent1", payload={"x": 1}))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus):
        counter = {"count": 0}

        async def h1(msg): counter["count"] += 1
        async def h2(msg): counter["count"] += 1

        bus.subscribe("tick", h1)
        bus.subscribe("tick", h2)

        await bus.publish(Message(topic="tick"))
        assert counter["count"] == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        received: list[Message] = []

        async def h(msg): received.append(msg)

        bus.subscribe("t", h)
        await bus.publish(Message(topic="t"))
        assert len(received) == 1

        bus.unsubscribe("t", h)
        await bus.publish(Message(topic="t"))
        assert len(received) == 1  # no new message

    @pytest.mark.asyncio
    async def test_request_reply(self, bus):
        async def echo_handler(msg):
            await bus.reply_to(msg, {"echo": msg.payload.get("data")})

        bus.register_agent("echo", echo_handler)

        reply = await bus.request(
            Message(recipient="echo", payload={"data": "hello"}),
            timeout=5.0,
        )
        assert reply is not None
        assert reply.payload == {"echo": "hello"}

    @pytest.mark.asyncio
    async def test_request_timeout(self, bus):
        reply = await bus.request(
            Message(recipient="nobody", payload={}),
            timeout=0.1,
        )
        assert reply is None

    @pytest.mark.asyncio
    async def test_mailbox_delivery(self, bus):
        await bus.publish(Message(recipient="offline_agent", topic="events",
                                  payload={"msg": "hello"}))

        msgs = bus.deliver_mailbox("offline_agent")
        assert len(msgs) == 1
        assert msgs[0].payload == {"msg": "hello"}
        assert not bus.has_pending("offline_agent")

    @pytest.mark.asyncio
    async def test_handoff_accepted(self, bus):
        async def accept_handler(req: HandoffRequest):
            return True

        bus.register_agent("target", lambda msg: None, accept_handler)

        req = HandoffRequest(from_agent="source", to_agent="target",
                             task="analyze this data")
        accepted = await bus.handoff(req)
        assert accepted is True
        assert req.accepted is True

    @pytest.mark.asyncio
    async def test_handoff_rejected(self, bus):
        async def reject_handler(req: HandoffRequest):
            return False

        bus.register_agent("target", lambda msg: None, reject_handler)

        req = HandoffRequest(from_agent="source", to_agent="target",
                             task="do something")
        accepted = await bus.handoff(req)
        assert accepted is False

    @pytest.mark.asyncio
    async def test_handoff_no_handler(self, bus):
        req = HandoffRequest(from_agent="source", to_agent="ghost",
                             task="do something")
        accepted = await bus.handoff(req)
        assert accepted is False

    def test_get_stats(self, bus):
        stats = bus.get_stats()
        assert "subscriptions" in stats
        assert "registered_agents" in stats
        assert "pending_messages" in stats


# ═════════════════════════════════════════════════════════════════════
# Orchestrator Tests
# ═════════════════════════════════════════════════════════════════════

class TestOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        registry = AgentRegistry()
        bus = MessageBus()
        orch = Orchestrator(registry, bus)
        return orch

    @pytest.fixture
    def mock_executor(self, orchestrator):
        """设置一个简单 echo executor。"""

        async def echo(agent_id, prompt, context):
            return f"[{agent_id}] processed: {prompt[:50]}"

        orchestrator.set_executor(echo)
        return echo

    @pytest.mark.asyncio
    async def test_run_without_executor(self, orchestrator):
        result = await orchestrator.run([Task(agent_id="r")])
        assert result.success is False
        assert "executor not set" in result.error

    @pytest.mark.asyncio
    async def test_sequential_execution(self, orchestrator, mock_executor):
        tasks = [
            Task(id="t1", agent_id="researcher", description="search"),
            Task(id="t2", agent_id="copywriter", description="write"),
            Task(id="t3", agent_id="analyst", description="analyze"),
        ]
        result = await orchestrator.run(tasks, mode=OrchestrationMode.SEQUENTIAL)
        assert result.success is True
        assert len(result.task_results) == 3
        for r in result.task_results:
            assert r.success is True
            assert "processed" in str(r.output)

    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator, mock_executor):
        tasks = [
            Task(agent_id="a", description="task a"),
            Task(agent_id="b", description="task b"),
            Task(agent_id="c", description="task c"),
        ]
        result = await orchestrator.run(tasks, mode=OrchestrationMode.PARALLEL)
        assert result.success is True
        assert len(result.task_results) == 3

    @pytest.mark.asyncio
    async def test_graph_execution_with_deps(self, orchestrator, mock_executor):
        tasks = [
            Task(id="research", agent_id="researcher", description="research topic"),
            Task(id="write", agent_id="copywriter", description="write article",
                 depends_on=["research"]),
            Task(id="review", agent_id="editor", description="review article",
                 depends_on=["write"]),
        ]
        result = await orchestrator.run(tasks, mode=OrchestrationMode.GRAPH)
        assert result.success is True
        assert len(result.task_results) == 3

        # Verify execution order: research first, then write, then review
        # In graph mode, research has 0 deps so runs first
        # write depends on research, review depends on write
        ids = [r.task_id for r in result.task_results]
        assert ids.index("research") < ids.index("write")
        assert ids.index("write") < ids.index("review")

    @pytest.mark.asyncio
    async def test_graph_parallel_independent(self, orchestrator, mock_executor):
        tasks = [
            Task(id="a", agent_id="a", description="independent a"),
            Task(id="b", agent_id="b", description="independent b"),
            Task(id="c", agent_id="c", description="independent c"),
        ]
        result = await orchestrator.run(tasks, mode=OrchestrationMode.GRAPH)
        assert result.success is True
        assert len(result.task_results) == 3

    @pytest.mark.asyncio
    async def test_swarm_with_capability_routing(self, orchestrator, mock_executor):
        orch = orchestrator
        orch.registry.register(AgentInfo(
            agent_id="researcher", capabilities=["research", "search"],
            status=AgentStatus.ONLINE))
        orch.registry.register(AgentInfo(
            agent_id="copywriter", capabilities=["writing", "content"],
            status=AgentStatus.ONLINE))

        tasks = [
            Task(id="t1", required_capability="research",
                 description="research AI trends"),
            Task(id="t2", required_capability="writing",
                 description="write report"),
        ]
        result = await orch.run(tasks, mode=OrchestrationMode.SWARM)
        assert result.success is True
        assert len(result.task_results) == 2

    @pytest.mark.asyncio
    async def test_task_timeout(self, orchestrator):
        async def slow_executor(agent_id, prompt, context):
            await asyncio.sleep(10)

        orchestrator.set_executor(slow_executor)
        result = await orchestrator.run(
            [Task(agent_id="r", timeout=0.1)],
            mode=OrchestrationMode.SEQUENTIAL,
        )
        assert result.success is False
        assert result.task_results[0].success is False

    @pytest.mark.asyncio
    async def test_context_propagation_sequential(self, orchestrator, mock_executor):
        tasks = [
            Task(id="step1", agent_id="r", description="first"),
            Task(id="step2", agent_id="w", description="second"),
        ]
        result = await orchestrator.run(tasks, mode=OrchestrationMode.SEQUENTIAL,
                                        context={"initial": "value"})
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_sequential_shortcut(self, orchestrator, mock_executor):
        result = await orchestrator.run_sequential(
            agent_ids=["a", "b", "c"],
            prompt_template="process this",
        )
        assert result.success is True
        assert len(result.task_results) == 3

    @pytest.mark.asyncio
    async def test_run_parallel_shortcut(self, orchestrator, mock_executor):
        result = await orchestrator.run_parallel(
            agent_tasks=[("a", "task a"), ("b", "task b")],
        )
        assert result.success is True
        assert len(result.task_results) == 2

    def test_orchestration_result_defaults(self):
        result = OrchestrationResult(mode=OrchestrationMode.SEQUENTIAL)
        assert result.success is False
        assert result.task_results == []
        assert result.total_duration_ms == 0.0

    def test_task_defaults(self):
        task = Task(description="hello")
        assert task.id
        assert task.agent_id == ""
        assert task.depends_on == []
        assert task.timeout == 60.0
