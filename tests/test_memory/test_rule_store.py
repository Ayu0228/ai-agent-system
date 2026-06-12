"""RuleStore tests — CRUD, cooldown, conflict resolution, decay."""

import time

import pytest

from src.shared.models import Rule, RuleStatus, ScopeLevel
from src.memory.rule_store import RuleStore


@pytest.fixture
def store():
    s = RuleStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def active_rule():
    return Rule(
        name="Retry on 429",
        description="When API returns 429, wait and retry",
        trigger_condition="api returns 429 rate limit",
        action="wait 5 seconds and retry",
        scope=ScopeLevel.AGENT,
        agent_id="agent-1",
        confidence=0.8,
        cooldown_seconds=60,
    )


class TestRuleStoreCRUD:
    """Basic CRUD operations."""

    def test_add_and_get(self, store, active_rule):
        rid = store.add_rule(active_rule)
        assert rid == active_rule.id

        retrieved = store.get_rule(rid)
        assert retrieved is not None
        assert retrieved.name == "Retry on 429"
        assert retrieved.trigger_condition == "api returns 429 rate limit"
        assert retrieved.action == "wait 5 seconds and retry"
        assert retrieved.status == RuleStatus.ACTIVE

    def test_get_nonexistent(self, store):
        assert store.get_rule("nonexistent-id") is None

    def test_list_active_filters_by_agent(self, store):
        r1 = Rule(agent_id="agent-1", trigger_condition="cond a", action="action a")
        r2 = Rule(agent_id="agent-2", trigger_condition="cond b", action="action b")
        store.add_rule(r1)
        store.add_rule(r2)

        results = store.list_active("agent-1")
        assert len(results) == 1
        assert results[0].agent_id == "agent-1"

    def test_list_active_empty_agent_returns_all(self, store):
        store.add_rule(Rule(agent_id="agent-1", trigger_condition="cond a", action="action a"))
        store.add_rule(Rule(agent_id="agent-2", trigger_condition="cond b", action="action b"))

        results = store.list_active()
        assert len(results) == 2

    def test_list_active_excludes_deprecated(self, store):
        r = Rule(agent_id="agent-1", trigger_condition="cond", action="act")
        store.add_rule(r)
        store.update_status(r.id, RuleStatus.DEPRECATED)

        assert len(store.list_active("agent-1")) == 0

    def test_list_by_scope(self, store):
        store.add_rule(Rule(agent_id="a1", scope=ScopeLevel.AGENT, trigger_condition="c1", action="a1"))
        store.add_rule(Rule(agent_id="a2", scope=ScopeLevel.GLOBAL, trigger_condition="c2", action="a2"))
        store.add_rule(Rule(agent_id="a3", scope=ScopeLevel.AGENT, trigger_condition="c3", action="a3"))

        agent_rules = store.list_by_scope(ScopeLevel.AGENT)
        assert len(agent_rules) == 2

        global_rules = store.list_by_scope(ScopeLevel.GLOBAL)
        assert len(global_rules) == 1

    def test_update_status(self, store, active_rule):
        store.add_rule(active_rule)
        assert store.update_status(active_rule.id, RuleStatus.CONFLICT)
        updated = store.get_rule(active_rule.id)
        assert updated.status == RuleStatus.CONFLICT


class TestCooldown:
    """Cooldown mechanism tests."""

    def test_check_cooldown_no_fire(self, store, active_rule):
        store.add_rule(active_rule)
        assert not store.check_cooldown(active_rule.id)

    def test_check_cooldown_within_window(self, store, active_rule):
        active_rule.cooldown_seconds = 300
        store.add_rule(active_rule)

        # Simulate a recent fire by directly setting last_fired_at
        store._db.execute(
            "UPDATE rules SET last_fired_at = ? WHERE id = ?",
            (time.time() - 30, active_rule.id),
        )
        store._db.commit()
        assert store.check_cooldown(active_rule.id)

    def test_check_cooldown_expired(self, store, active_rule):
        active_rule.cooldown_seconds = 5
        store.add_rule(active_rule)

        store._db.execute(
            "UPDATE rules SET last_fired_at = ? WHERE id = ?",
            (time.time() - 60, active_rule.id),
        )
        store._db.commit()
        assert not store.check_cooldown(active_rule.id)

    def test_record_fire_updates_metrics(self, store, active_rule):
        store.add_rule(active_rule)
        store.record_fire(active_rule.id, success=True)

        updated = store.get_rule(active_rule.id)
        assert updated.fire_count == 1
        assert updated.success_rate == 1.0
        assert updated.last_fired_at is not None

    def test_record_fire_running_average(self, store, active_rule):
        active_rule.success_rate = 0.5
        store.add_rule(active_rule)
        store.record_fire(active_rule.id, success=False)

        updated = store.get_rule(active_rule.id)
        # Running average: (0.5 * 0 + 0) / 1 = 0.0
        assert updated.success_rate == 0.0

        store.record_fire(active_rule.id, success=True)
        updated = store.get_rule(active_rule.id)
        # (0.0 * 1 + 1) / 2 = 0.5
        assert updated.success_rate == 0.5


class TestConflictResolution:
    """Conflict resolution tests."""

    def test_empty_list(self, store):
        assert store.resolve_conflicts([]) == []

    def test_single_rule(self, store, active_rule):
        result = store.resolve_conflicts([active_rule])
        assert len(result) == 1
        assert result[0].id == active_rule.id

    def test_sorts_by_confidence_times_success_rate(self, store):
        r1 = Rule(confidence=0.9, success_rate=0.5, trigger_condition="cond a", action="act a")
        r2 = Rule(confidence=0.6, success_rate=0.9, trigger_condition="cond b", action="act b")
        # r1 score = 0.45, r2 score = 0.54 → r2 first

        result = store.resolve_conflicts([r1, r2])
        assert result[0].id == r2.id
        assert result[1].id == r1.id

    def test_deduplicates_same_trigger(self, store):
        r1 = Rule(confidence=0.9, success_rate=0.9, trigger_condition="rate limit exceeded", action="retry")
        r2 = Rule(confidence=0.5, success_rate=0.5, trigger_condition="rate limit exceeded", action="backoff")

        result = store.resolve_conflicts([r1, r2])
        assert len(result) == 1
        assert result[0].id == r1.id  # Higher score kept

    def test_different_triggers_both_kept(self, store):
        r1 = Rule(trigger_condition="rate limit", action="retry")
        r2 = Rule(trigger_condition="timeout error", action="escalate")

        result = store.resolve_conflicts([r1, r2])
        assert len(result) == 2


class TestDecay:
    """Rule decay tests."""

    def test_decay_marks_low_success_rate(self, store):
        r = Rule(agent_id="agent-1", trigger_condition="cond", action="act")
        store.add_rule(r)
        # Simulate 3 fires, all failures
        store._db.execute(
            "UPDATE rules SET success_rate = 0.1, fire_count = 3 WHERE id = ?",
            (r.id,),
        )
        store._db.commit()

        deprecated = store.decay_rules(success_rate_threshold=0.3)
        assert deprecated == 1
        updated = store.get_rule(r.id)
        assert updated.status == RuleStatus.DEPRECATED

    def test_decay_skips_insufficient_fires(self, store):
        r = Rule(agent_id="agent-1", trigger_condition="cond", action="act")
        store.add_rule(r)
        store._db.execute(
            "UPDATE rules SET success_rate = 0.1, fire_count = 1 WHERE id = ?",
            (r.id,),
        )
        store._db.commit()

        deprecated = store.decay_rules(success_rate_threshold=0.3)
        assert deprecated == 0
        updated = store.get_rule(r.id)
        assert updated.status == RuleStatus.ACTIVE

    def test_decay_keeps_high_success_rate(self, store):
        r = Rule(agent_id="agent-1", trigger_condition="cond", action="act")
        store.add_rule(r)
        store._db.execute(
            "UPDATE rules SET success_rate = 0.9, fire_count = 10 WHERE id = ?",
            (r.id,),
        )
        store._db.commit()

        deprecated = store.decay_rules(success_rate_threshold=0.3)
        assert deprecated == 0


class TestStats:
    """Statistics tests."""

    def test_get_stats_empty(self, store):
        stats = store.get_stats()
        assert stats["total"] == 0
        assert stats["avg_success_rate"] == 0.0

    def test_get_stats_with_rules(self, store):
        store.add_rule(Rule(agent_id="a1", trigger_condition="c1", action="a1"))
        store.add_rule(Rule(agent_id="a2", trigger_condition="c2", action="a2"))
        r3 = Rule(agent_id="a3", trigger_condition="c3", action="a3")
        store.add_rule(r3)
        store.update_status(r3.id, RuleStatus.DEPRECATED)

        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["by_status"]["active"] == 2
        assert stats["by_status"]["deprecated"] == 1
