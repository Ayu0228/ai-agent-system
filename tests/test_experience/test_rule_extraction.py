"""ExperienceExtractor rule extraction tests — extract_rules() and _parse_rules()."""

import pytest

from src.shared.models import Experience, Rule, ScopeLevel
from src.experience.extractor import ExperienceExtractor


class TestParseRules:
    """Test _parse_rules static method."""

    def test_parse_simple_rule(self):
        exp = Experience(
            agent_id="agent-1",
            task_type="research",
            trigger="API timeout",
            symptom="request failed",
            root_cause="upstream slow",
            solution="switch to smaller model",
            outcome="success",
            confidence=0.8,
        )
        raw = """```json
[{"name": "Timeout Fallback", "trigger_condition": "WHEN api timeout > 30s", "action": "THEN switch to smaller model", "confidence": 0.75, "description": "Fallback on timeout"}]
```"""
        rules = ExperienceExtractor._parse_rules(raw, [exp], "agent-1", ScopeLevel.AGENT)
        assert len(rules) == 1
        assert rules[0].name == "Timeout Fallback"
        assert rules[0].trigger_condition == "WHEN api timeout > 30s"
        assert rules[0].action == "THEN switch to smaller model"
        assert rules[0].confidence == 0.75
        assert rules[0].scope == ScopeLevel.AGENT
        assert rules[0].agent_id == "agent-1"
        assert rules[0].source_experience_id == exp.id

    def test_parse_multiple_rules(self):
        exps = [
            Experience(agent_id="a1", task_type="coding", trigger="t1", symptom="s1",
                       root_cause="r1", solution="s1", outcome="o1", confidence=0.7),
            Experience(agent_id="a1", task_type="coding", trigger="t2", symptom="s2",
                       root_cause="r2", solution="s2", outcome="o2", confidence=0.9),
        ]
        raw = """```json
[
  {"name": "Rule 1", "trigger_condition": "WHEN condition A", "action": "THEN action A", "confidence": 0.6, "description": "desc 1"},
  {"name": "Rule 2", "trigger_condition": "WHEN condition B", "action": "THEN action B", "confidence": 0.8, "description": "desc 2"}
]
```"""
        rules = ExperienceExtractor._parse_rules(raw, exps, "agent-1", ScopeLevel.GLOBAL)
        assert len(rules) == 2
        # First rule inherits first experience as source
        assert rules[0].source_experience_id == exps[0].id
        assert rules[1].source_experience_id == exps[1].id

    def test_parse_rules_confidence_inheritance(self):
        """When item doesn't specify confidence, use avg experience confidence * 0.8."""
        exp = Experience(
            agent_id="a1", task_type="t", trigger="t", symptom="s",
            root_cause="r", solution="s", outcome="o", confidence=0.9,
        )
        # No confidence in JSON → should inherit avg_exp_confidence * 0.8 = 0.72
        raw = '[{"name": "Rule", "trigger_condition": "WHEN x", "action": "THEN y", "description": "desc"}]'
        rules = ExperienceExtractor._parse_rules(raw, [exp], "a1", ScopeLevel.AGENT)
        assert len(rules) == 1
        assert rules[0].confidence == pytest.approx(0.72)

    def test_parse_rules_invalid_json(self):
        rules = ExperienceExtractor._parse_rules("not json", [], "a1", ScopeLevel.AGENT)
        assert rules == []

    def test_parse_rules_empty_experience_average(self):
        raw = '[{"name": "R", "trigger_condition": "WHEN x", "action": "THEN y", "confidence": 0.5}]'
        rules = ExperienceExtractor._parse_rules(raw, [], "a1", ScopeLevel.AGENT)
        # avg_confidence = 0 / 1 = 0 → 0 * 0.8 = 0 (but item has explicit confidence)
        assert len(rules) == 1
        assert rules[0].confidence == 0.5  # Explicit overrides

    def test_parse_rules_no_fence(self):
        exp = Experience(agent_id="a1", task_type="t", trigger="t", symptom="s",
                         root_cause="r", solution="s", outcome="o", confidence=0.5)
        raw = '[{"name": "R", "trigger_condition": "WHEN x", "action": "THEN y", "confidence": 0.5}]'
        rules = ExperienceExtractor._parse_rules(raw, [exp], "a1", ScopeLevel.AGENT)
        assert len(rules) == 1
