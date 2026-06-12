"""MemoryConsolidator procedural_rule extraction tests."""

import pytest

from src.memory.consolidator import MemoryConsolidator
from src.shared.models import MemoryEntry, MemoryType


class TestProceduralRuleExtraction:
    """Test that consolidator prompt and parsing handle procedural_rule category."""

    def test_prompt_includes_procedural_rule_category(self):
        mems = [
            MemoryEntry(
                agent_id="agent-1",
                content="When API returns 429, I should wait and retry",
                memory_type=MemoryType.EXPERIENCE,
            ),
        ]
        prompt = MemoryConsolidator._build_extraction_prompt(mems)

        assert "procedural_rule" in prompt
        assert "WHEN" in prompt
        assert "THEN" in prompt

    def test_parse_extraction_accepts_procedural_rule(self):
        raw = """```json
[{"category": "procedural_rule", "content": "WHEN api returns 429 THEN wait 5s and retry", "importance": 0.9, "tags": ["rule", "api"]}]
```"""
        items = MemoryConsolidator._parse_extraction(raw)
        assert len(items) == 1
        assert items[0]["category"] == "procedural_rule"
        assert "WHEN" in items[0]["content"]
        assert "THEN" in items[0]["content"]

    def test_parse_extraction_mixed_categories(self):
        raw = """```json
[
  {"category": "factual_knowledge", "content": "Python 3.12 has better error messages", "importance": 0.7, "tags": ["python"]},
  {"category": "procedural_rule", "content": "WHEN timeout > 30s THEN switch to smaller model", "importance": 0.85, "tags": ["rule"]},
  {"category": "error_lesson", "content": "Don't cache auth tokens", "importance": 0.6, "tags": ["cache"]}
]
```"""
        items = MemoryConsolidator._parse_extraction(raw)
        assert len(items) == 3
        categories = {i["category"] for i in items}
        assert "procedural_rule" in categories
        assert "factual_knowledge" in categories
        assert "error_lesson" in categories

    def test_parse_extraction_no_fence(self):
        raw = '[{"category": "procedural_rule", "content": "WHEN X THEN Y", "importance": 0.5, "tags": []}]'
        items = MemoryConsolidator._parse_extraction(raw)
        assert len(items) == 1
        assert items[0]["category"] == "procedural_rule"

    def test_parse_extraction_invalid_json(self):
        items = MemoryConsolidator._parse_extraction("not valid json")
        assert items == []
