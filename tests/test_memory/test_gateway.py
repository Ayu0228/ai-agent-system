"""MemoryGateway 单元测试。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.shared.models import MemoryEntry, MemoryQuery, MemoryType, WriteEvaluation, WriteDecision


class TestMemoryGateway:
    """测试统一记忆网关。"""

    @pytest.fixture
    def gateway(self):
        from src.memory.gateway import MemoryGateway

        with patch("src.memory.gateway.LongTermMemory") as mock_lt, \
             patch("src.memory.gateway.WritePolicy") as mock_wp:
            gw = MemoryGateway()
            gw._long_term = mock_lt.return_value
            gw._write_policy = mock_wp.return_value
            yield gw

    async def test_write_novel_entry(self, gateway):
        """写入新记忆条目。"""
        from src.memory.long_term import MemorySearchResult

        entry = MemoryEntry(
            agent_id="researcher",
            content="youtube-dl download timeout 600s for long videos",
            memory_type=MemoryType.EXPERIENCE,
            importance=0.8,
        )
        gateway._long_term.search.return_value = []
        gateway._write_policy.evaluate.return_value = WriteEvaluation(
            decision=WriteDecision.WRITE, importance=0.8, reason="novel"
        )
        gateway._long_term.store.return_value = entry.id

        result = await gateway.write(entry, trace_id="t1")
        assert result == entry.id
        gateway._long_term.store.assert_called_once()

    async def test_write_skip_low_importance(self, gateway):
        """跳过低重要性条目。"""
        entry = MemoryEntry(
            agent_id="researcher",
            content="trivial info",
            importance=0.1,
        )
        gateway._long_term.search.return_value = []
        gateway._write_policy.evaluate.return_value = WriteEvaluation(
            decision=WriteDecision.SKIP, importance=0.1, reason="low_importance"
        )

        result = await gateway.write(entry, trace_id="t1")
        assert result == entry.id
        gateway._long_term.store.assert_not_called()

    async def test_search_delegates_to_long_term(self, gateway):
        """搜索委托给长期记忆。"""
        from src.memory.long_term import MemorySearchResult

        entry = MemoryEntry(
            agent_id="researcher",
            content="test memory",
        )
        gateway._long_term.search.return_value = [
            MemorySearchResult(entry=entry, score=0.9)
        ]

        results = await gateway.search(
            MemoryQuery(query_text="test memory"), trace_id="t1"
        )
        assert len(results) == 1
        assert results[0].entry.content == "test memory"

    async def test_share_to_other_agents(self, gateway):
        """跨 Agent 共享记忆。"""
        from src.memory.long_term import MemorySearchResult

        entry = MemoryEntry(
            agent_id="researcher",
            content="valuable insight",
            importance=0.9,
        )
        gateway._long_term.search.return_value = [
            MemorySearchResult(entry=entry, score=1.0)
        ]
        gateway._write_policy.evaluate.return_value = WriteEvaluation(
            decision=WriteDecision.WRITE, importance=0.9, reason="novel"
        )
        gateway._long_term.store.return_value = "shared-1"

        ok = await gateway.share(
            entry.id, from_agent="researcher",
            to_agents=["tech-dev", "data-analyst"], trace_id="t1"
        )
        assert ok is True
        assert gateway._long_term.store.call_count >= 2

    async def test_decay_delegates(self, gateway):
        """遗忘策略委托给长期记忆。"""
        gateway._long_term.decay.return_value = 5
        removed = await gateway.decay(trace_id="t1")
        assert removed == 5
