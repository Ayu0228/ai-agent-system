"""KnowledgeGraph rule conflict detection tests."""

import pytest

from src.knowledge.version_graph import KnowledgeGraph


class TestRuleConflicts:
    """Test KnowledgeGraph.detect_rule_conflicts()."""

    def test_no_conflict_different_triggers(self):
        kg = KnowledgeGraph()
        existing = [("get current weather forecast", "call weather api", 0.8)]
        conflicts = kg.detect_rule_conflicts(
            "calculate monthly revenue", "query database for revenue",
            existing,
        )
        assert len(conflicts) == 0

    def test_no_conflict_same_trigger_same_action(self):
        kg = KnowledgeGraph()
        existing = [("api timeout", "retry with backoff", 0.7)]
        conflicts = kg.detect_rule_conflicts(
            "api timeout", "retry with backoff", existing,
        )
        assert len(conflicts) == 0

    def test_conflict_similar_trigger_different_action(self):
        kg = KnowledgeGraph()
        existing = [("api returns 429 rate limit error", "wait 5 seconds and retry", 0.7)]
        conflicts = kg.detect_rule_conflicts(
            "api returns 429 rate limit exceeded",
            "switch to backup api endpoint",
            existing,
        )
        assert len(conflicts) == 1
        assert conflicts[0]["conflict_with"] == 0
        assert conflicts[0]["overlap"] > 0.5
        assert "auto_resolve" in conflicts[0]

    def test_auto_resolve_keep_new(self):
        """High confidence diff → auto-resolve. But note the calculation needs fixing."""
        kg = KnowledgeGraph()
        existing = [("api rate limit exceeded", "retry with backoff", 0.3)]
        conflicts = kg.detect_rule_conflicts(
            "api rate limit exceeded",
            "switch to backup endpoint",
            existing,
        )
        assert len(conflicts) == 1
        # conf_diff = abs(0.5 - 0.3) = 0.2 >= 0.2 → keep_new
        assert conflicts[0]["auto_resolve"] == "keep_new"

    def test_auto_resolve_conflict_when_close_confidence(self):
        kg = KnowledgeGraph()
        existing = [("api rate limit exceeded", "retry with backoff", 0.45)]
        conflicts = kg.detect_rule_conflicts(
            "api rate limit exceeded",
            "switch to backup endpoint",
            existing,
        )
        assert len(conflicts) == 1
        # conf_diff = abs(0.5 - 0.45) = 0.05 < 0.2 → conflict (needs human review)
        assert conflicts[0]["auto_resolve"] == "conflict"

    def test_multiple_existing_rules_finds_all_conflicts(self):
        kg = KnowledgeGraph()
        existing = [
            ("api returns 429 rate limit", "retry with backoff", 0.8),
            ("api 429 rate limit exceeded", "switch endpoint", 0.6),
            ("completely different trigger", "some other action", 0.5),
        ]
        conflicts = kg.detect_rule_conflicts(
            "api returns 429 rate limit exceeded",
            "block further requests",
            existing,
        )
        # Should conflict with first two (similar trigger, different action)
        assert len(conflicts) == 2

    def test_empty_trigger_no_conflicts(self):
        kg = KnowledgeGraph()
        existing = [("some trigger", "some action", 0.5)]
        conflicts = kg.detect_rule_conflicts("", "some action", existing)
        assert len(conflicts) == 0

    def test_empty_existing_rules(self):
        kg = KnowledgeGraph()
        conflicts = kg.detect_rule_conflicts("new trigger", "new action", [])
        assert conflicts == []
