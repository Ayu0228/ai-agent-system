"""RAG 检索和 Verify 多模型校验测试。"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.shared.models import StepConfig, StepType


class TestRAGRetriever:
    """测试 RAGRetriever 检索和格式化。"""

    def test_retrieve_empty_query(self):
        from src.rag.retriever import RAGRetriever
        retriever = RAGRetriever()
        results = retriever.retrieve("")
        assert results == []

    def test_format_context_empty(self):
        from src.rag.retriever import RAGRetriever
        result = RAGRetriever.format_context([])
        assert result == ""

    def test_format_context_with_results(self):
        from src.rag.retriever import RAGRetriever
        results = [
            {"content": "Python async is important", "score": 0.85, "source": "researcher", "type": "fact"},
            {"content": "Use asyncio.run() to start", "score": 0.72, "source": "tech-dev", "type": "experience"},
        ]
        context = RAGRetriever.format_context(results)
        assert "Python async is important" in context
        assert "Use asyncio.run()" in context
        assert "0.85" in context
        assert "researcher" in context
        assert "tech-dev" in context
        assert "知识库" in context  # Chinese instruction
        assert "---" in context  # Delimiter

    def test_retrieve_with_mock_memory(self):
        """检索：取回结果 → 阈值过滤 → 去重排序。"""
        from src.rag.retriever import RAGRetriever
        from src.memory.long_term import LongTermMemory, MemorySearchResult
        from src.shared.models import MemoryEntry, MemoryType

        retriever = RAGRetriever(top_k=2, threshold=0.5)

        entry1 = MemoryEntry(agent_id="researcher", content="entry one",
                            memory_type=MemoryType.FACT, importance=0.8)
        entry2 = MemoryEntry(agent_id="tech-dev", content="entry two duplicate",
                            memory_type=MemoryType.EXPERIENCE, importance=0.6)
        entry3 = MemoryEntry(agent_id="researcher", content="entry three",
                            memory_type=MemoryType.FACT, importance=0.3)

        mock_results = [
            MemorySearchResult(entry=entry1, score=0.9),
            MemorySearchResult(entry=entry2, score=0.7),
            MemorySearchResult(entry=entry3, score=0.3),  # below threshold
        ]

        with patch.object(LongTermMemory, "search", return_value=mock_results):
            results = retriever.retrieve("test query")
            assert len(results) == 2
            assert results[0]["content"] == "entry one"
            assert results[0]["score"] == 0.9
            assert results[1]["content"] == "entry two duplicate"
            assert results[1]["score"] == 0.7

    def test_retrieve_dedup(self):
        """内容完全相同的结果只保留一条。"""
        from src.rag.retriever import RAGRetriever
        from src.memory.long_term import LongTermMemory, MemorySearchResult
        from src.shared.models import MemoryEntry, MemoryType

        retriever = RAGRetriever(top_k=5, threshold=0.5)

        entry1 = MemoryEntry(agent_id="a", content="same content",
                            memory_type=MemoryType.FACT, importance=0.8)
        entry2 = MemoryEntry(agent_id="b", content="same content",
                            memory_type=MemoryType.FACT, importance=0.6)

        mock_results = [
            MemorySearchResult(entry=entry1, score=0.9),
            MemorySearchResult(entry=entry2, score=0.7),
        ]

        with patch.object(LongTermMemory, "search", return_value=mock_results):
            results = retriever.retrieve("test")
            assert len(results) == 1  # deduplicated


class TestVerifyStep:
    """测试多模型校验。"""

    @pytest.fixture
    def executor(self):
        from src.workflow.steps import StepExecutor
        return StepExecutor()

    async def test_verify_no_config(self, executor):
        """没有 verify 配置时直接返回。"""
        parsed = {"status": "success", "summary": "ok", "data": {}, "error": None, "confidence": "medium"}
        result = await executor._apply_verify(parsed, {}, "prompt", 60, "t1")
        assert result == parsed  # unchanged

    async def test_verify_no_agent(self, executor):
        """verify.agent 为空时直接返回。"""
        parsed = {"status": "success", "summary": "ok", "data": {}, "error": None, "confidence": "medium"}
        result = await executor._apply_verify(parsed, {"agent": ""}, "prompt", 60, "t1")
        assert result == parsed

    async def test_verify_agent_called(self, executor):
        """校验 Agent 正常返回，结果合并。"""
        with patch.object(executor, "_call_agent", return_value='{"summary":"looks ok","confidence":"high","data":{},"error":null}'):
            parsed = {"status": "success", "summary": "original", "data": {}, "error": None, "confidence": "medium"}
            result = await executor._apply_verify(
                parsed, {"agent": "tech-dev"}, "original prompt", 60, "t1"
            )
            assert result["confidence"] == "high"
            assert result["data"]["_verified_by"] == "tech-dev"
            assert "_verify_feedback" in result["data"]

    async def test_verify_agent_finds_issue_lowers_confidence(self, executor):
        """校验 Agent 发现问题 → 置信度降为 low。"""
        with patch.object(executor, "_call_agent", return_value='{"summary":"error found","confidence":"low","data":{},"error":"factual error"}'):
            parsed = {"status": "success", "summary": "original", "data": {}, "error": None, "confidence": "high"}
            result = await executor._apply_verify(
                parsed, {"agent": "tech-dev"}, "prompt", 60, "t1"
            )
            assert result["confidence"] == "low"

    async def test_verify_agent_fails_gracefully(self, executor):
        """校验 Agent 调用失败 → 标低置信度但不阻断。"""
        with patch.object(executor, "_call_agent", side_effect=Exception("timeout")):
            parsed = {"status": "success", "summary": "original", "data": {}, "error": None, "confidence": "high"}
            result = await executor._apply_verify(
                parsed, {"agent": "tech-dev"}, "prompt", 60, "t1"
            )
            assert result["confidence"] == "low"
            assert "_verify_error" in result["data"]

    async def test_verify_with_custom_prompt(self, executor):
        """自定义校验 prompt。"""
        with patch.object(executor, "_call_agent", return_value='{"summary":"ok","confidence":"high","data":{},"error":null}'):
            parsed = {"status": "success", "summary": "original", "data": {}, "error": None, "confidence": "medium"}
            result = await executor._apply_verify(
                parsed,
                {"agent": "tech-dev", "prompt": "custom verify prompt"},
                "original", 60, "t1"
            )
            # _call_agent should be called with the custom prompt
            call_args = executor._call_agent.call_args
            assert "custom verify prompt" in call_args[0][1]
