"""经验系统测试。"""

import pytest
from unittest.mock import AsyncMock, patch

from src.shared.models import (
    Experience, MemoryEntry, MemoryType, StepResult, WriteEvaluation, WriteDecision,
)


class TestExperienceExtractor:
    """测试经验提取器。"""

    def test_parse_valid_json(self):
        from src.experience.extractor import ExperienceExtractor

        raw = """```json
[{"trigger": "timeout", "symptom": "request hangs", "root_cause": "network", "solution": "retry", "outcome": "success", "confidence": 0.8, "tags": ["network"]}]
```"""
        result = ExperienceExtractor._parse(raw, "researcher", "web_scraping")
        assert len(result) == 1
        assert result[0].trigger == "timeout"
        assert result[0].solution == "retry"
        assert result[0].confidence == 0.8
        assert result[0].agent_id == "researcher"

    def test_parse_invalid_json(self):
        from src.experience.extractor import ExperienceExtractor

        result = ExperienceExtractor._parse("not json at all", "agent", "task")
        assert result == []

    def test_parse_no_fence(self):
        from src.experience.extractor import ExperienceExtractor

        raw = '[{"trigger": "error", "symptom": "crash", "root_cause": "bug", "solution": "patch", "outcome": "fixed", "confidence": 0.9}]'
        result = ExperienceExtractor._parse(raw, "agent", "task")
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_build_prompt_includes_steps(self):
        from src.experience.extractor import ExperienceExtractor

        steps = [
            StepResult(step_id="s1", status="success"),
            StepResult(step_id="s2", status="failed", error="timeout"),
        ]
        prompt = ExperienceExtractor._build_prompt(
            "researcher", "web_search", False, steps
        )
        assert "researcher" in prompt
        assert "web_search" in prompt
        assert "失败" in prompt
        assert "s1" in prompt
        assert "s2" in prompt

    def test_build_prompt_success(self):
        from src.experience.extractor import ExperienceExtractor

        steps = [StepResult(step_id="s1", status="success")]
        prompt = ExperienceExtractor._build_prompt(
            "researcher", "web_search", True, steps
        )
        assert "成功" in prompt

    async def test_extract_calls_llm(self):
        from src.experience.extractor import ExperienceExtractor

        extractor = ExperienceExtractor()
        extractor._llm = AsyncMock()
        extractor._llm.chat.return_value = """```json
[{"trigger": "t", "symptom": "s", "root_cause": "r", "solution": "fix", "outcome": "ok", "confidence": 0.7}]
```"""
        steps = [StepResult(step_id="s1", status="success")]
        result = await extractor.extract(
            "researcher", "task_type", True, steps, trace_id="t1"
        )
        assert len(result) == 1

    async def test_extract_handles_llm_error(self):
        from src.experience.extractor import ExperienceExtractor

        extractor = ExperienceExtractor()
        extractor._llm = AsyncMock()
        extractor._llm.chat.side_effect = Exception("API error")
        steps = [StepResult(step_id="s1", status="failed", error="crash")]
        result = await extractor.extract(
            "researcher", "task", False, steps, trace_id="t1"
        )
        assert result == []


class TestExperienceRetriever:
    """测试经验检索器。"""

    async def test_retrieve_sorts_by_validated_and_confidence(self):
        from src.experience.retriever import ExperienceRetriever
        from src.memory.long_term import MemorySearchResult

        retriever = ExperienceRetriever(gateway=AsyncMock())

        # 模拟搜索结果
        entries = [
            MemorySearchResult(entry=MemoryEntry(
                agent_id="researcher", content="exp1",
                memory_type=MemoryType.EXPERIENCE, tags=["type:search", "validated"],
                importance=0.9, id="e1",
            ), score=0.8),
            MemorySearchResult(entry=MemoryEntry(
                agent_id="researcher", content="exp2",
                memory_type=MemoryType.EXPERIENCE, tags=["type:search"],
                importance=0.7, id="e2",
            ), score=0.6),
            MemorySearchResult(entry=MemoryEntry(
                agent_id="researcher", content="exp3",
                memory_type=MemoryType.EXPERIENCE, tags=["type:search", "validated"],
                importance=0.95, id="e3",
            ), score=0.9),
        ]
        retriever._gateway.search.return_value = entries

        result = await retriever.retrieve("researcher", "search", top_k=3, trace_id="t1")
        assert len(result) <= 3
        # 已验证的应该排在前面
        if len(result) >= 2:
            assert result[0].validated is True

    def test_format_context_block(self):
        from src.experience.retriever import ExperienceRetriever

        experiences = [
            Experience(
                agent_id="researcher", task_type="search",
                trigger="timeout", symptom="hang", root_cause="network",
                solution="use longer timeout", outcome="resolved",
                confidence=0.9, validated=True,
            ),
            Experience(
                agent_id="researcher", task_type="scrape",
                trigger="blocked", symptom="403", root_cause="anti-bot",
                solution="add headers", outcome="bypassed",
                confidence=0.7, validated=False,
            ),
        ]
        block = ExperienceRetriever.format_context_block(experiences)
        assert "历史经验" in block
        assert "timeout" in block or "longer timeout" in block
        assert "已验证" in block

    def test_format_empty(self):
        from src.experience.retriever import ExperienceRetriever

        assert ExperienceRetriever.format_context_block([]) == ""


class TestExperienceValidator:
    """测试经验验证器。"""

    async def test_approved_experience_stored(self):
        from src.experience.validator import ExperienceValidator

        gateway = AsyncMock()
        gateway.write.return_value = "memory-123"
        validator = ExperienceValidator(gateway)

        exp = Experience(
            agent_id="researcher", task_type="search",
            trigger="timeout", symptom="hang", root_cause="network",
            solution="retry with backoff", outcome="resolved",
            confidence=0.85, validated=False,
        )
        entry_id, rules = await validator.validate_and_store(exp, approved=True, trace_id="t1")
        assert entry_id == "memory-123"
        assert isinstance(rules, list)
        gateway.write.assert_called_once()

    async def test_rejected_experience_discarded(self):
        from src.experience.validator import ExperienceValidator

        gateway = AsyncMock()
        validator = ExperienceValidator(gateway)

        exp = Experience(
            agent_id="researcher", task_type="search",
            trigger="error", symptom="crash", root_cause="unknown",
            solution="restart", outcome="uncertain",
            confidence=0.3, validated=False,
        )
        entry_id, rules = await validator.validate_and_store(exp, approved=False, trace_id="t1")
        assert entry_id is None
        assert rules == []
        gateway.write.assert_not_called()
