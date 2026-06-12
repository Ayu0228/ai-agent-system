"""Relay 模块测试。"""

import pytest
from unittest.mock import AsyncMock, patch

from src.shared.constants import AGENT_IDS


class TestAgentRouter:
    """测试 Agent 路由器。"""

    def test_route_by_keyword(self):
        from src.relay.router import AgentRouter

        router = AgentRouter()
        assert router.route("search for latest AI news") == "researcher"
        assert router.route("write a blog post about Python") == "copywriter"
        assert router.route("debug the authentication bug") == "tech-dev"
        assert router.route("analyze the sales data") == "data-analyst"

    def test_route_to_preferred(self):
        from src.relay.router import AgentRouter

        router = AgentRouter()
        assert router.route("search for news", preferred_agent="main") == "main"

    def test_route_unknown_to_main(self):
        from src.relay.router import AgentRouter

        router = AgentRouter()
        result = router.route("xyzzy random gibberish task")
        assert result == "main"

    def test_route_case_insensitive(self):
        from src.relay.router import AgentRouter

        router = AgentRouter()
        assert router.route("RESEARCH quantum computing") == "researcher"

    def test_all_routing_targets_are_valid(self):
        from src.relay.router import AgentRouter

        router = AgentRouter()
        for agent_id in router.task_routing.values():
            assert agent_id in AGENT_IDS, f"{agent_id} not in AGENT_IDS"

    def test_list_agents(self):
        from src.relay.router import AgentRouter

        router = AgentRouter()
        agents = router.list_agents()
        assert len(agents) == 11
        ids = [a["id"] for a in agents]
        assert "main" in ids
        assert "researcher" in ids

    def test_get_capable_agents(self):
        from src.relay.router import AgentRouter

        router = AgentRouter()
        agents = router.get_capable_agents("web_search")
        assert "researcher" in agents


class TestRelayClient:
    """测试 Relay 客户端。"""

    @pytest.fixture
    def client(self):
        from src.relay.client import RelayClient
        return RelayClient(use_relay=False)

    async def test_call_agent_llm_fallback(self, client):
        """未启用 relay 时走 LLM fallback。"""
        client._llm = AsyncMock()
        client._llm.chat.return_value = "agent response"

        result = await client.call_agent("researcher", "search for news", trace_id="t1")
        assert result == "agent response\n\n〔ai-agent-system〕"
        client._llm.chat.assert_called_once()

    async def test_call_agent_loads_md_prompt(self, client):
        """验证加载了 Agent 的 AGENTS.md 作为 system prompt。"""
        client._llm = AsyncMock()
        client._llm.chat.return_value = "response with prompt"

        # researcher.md 存在
        from unittest.mock import patch
        from src.relay.client import PromptLoadResult
        mock_result = PromptLoadResult(
            content="# Researcher\n\nSystem prompt",
            estimated_tokens=10,
            source_path="agents/researcher.md",
        )
        with patch.object(client, "_load_agent_prompt", return_value=mock_result):
            result = await client.call_agent("researcher", "do research", trace_id="t2")
            assert result == "response with prompt\n\n〔ai-agent-system〕"
            # 检查 system prompt 被传入
            call_args = client._llm.chat.call_args[0][0]
            assert call_args[0]["role"] == "system"
            assert "Researcher" in call_args[0]["content"]

    async def test_broadcast_parallel(self, client):
        """并行广播到多个 Agent。"""
        client._llm = AsyncMock()
        client._llm.chat.side_effect = [
            "response 1", "response 2", "response 3",
        ]

        results = await client.broadcast(
            ["researcher", "data-analyst", "copywriter"],
            "analyze this topic",
            trace_id="t3",
        )
        assert len(results) == 3
        assert "researcher" in results
        assert "data-analyst" in results
        assert "copywriter" in results

    def test_load_agent_prompt_exists(self):
        from src.relay.client import RelayClient

        result = RelayClient._load_agent_prompt("researcher")
        prompt = result.content
        assert "研究员" in prompt or "Researcher" in prompt or len(prompt) > 0

    def test_load_agent_prompt_missing(self):
        from src.relay.client import RelayClient

        result = RelayClient._load_agent_prompt("nonexistent_agent")
        assert result.content == ""

    def test_relay_unavailable_by_default(self, client):
        assert client.relay_available is False

    @pytest.fixture
    def relay_client(self):
        from src.relay.client import RelayClient
        return RelayClient(use_relay=True)

    async def test_relay_call_subprocess(self, relay_client):
        """测试 relay 模式的子进程调用。"""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate.return_value = (b"relay response", b"")
            mock_exec.return_value = mock_proc

            result = await relay_client._relay_send(
                "researcher", "search query", "t4"
            )
            assert result == "relay response"

    async def test_relay_call_failure_fallback(self, relay_client):
        """Relay 失败时 fallback 到 LLM。"""
        relay_client._llm = AsyncMock()
        relay_client._llm.chat.return_value = "fallback response"

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate.return_value = (b"", b"relay error")
            mock_exec.return_value = mock_proc

            result = await relay_client.call_agent(
                "researcher", "search query", trace_id="t5"
            )
            assert "fallback" in result

    async def test_llm_fallback_includes_system_prompt(self, relay_client):
        """LLM fallback 应该加载 Agent 的 system prompt。"""
        relay_client._llm = AsyncMock()
        relay_client._llm.chat.return_value = "response"

        from src.relay.client import PromptLoadResult
        mock_result = PromptLoadResult(content="Custom system prompt", estimated_tokens=5)
        with patch.object(relay_client, "_load_agent_prompt", return_value=mock_result):
            await relay_client._llm_fallback("researcher", "task", "t6")
            call_args = relay_client._llm.chat.call_args[0][0]
            assert call_args[0]["role"] == "system"
            assert call_args[0]["content"] == "Custom system prompt"
