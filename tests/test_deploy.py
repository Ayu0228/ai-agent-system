"""Deployment & IAM platform tests — canary, rollback, IAM."""

from __future__ import annotations

import time

import pytest

from src.deploy.canary import (
    CanaryDeployment, CanaryConfig, CanaryStep, CanaryStatus, MetricSnapshot,
)
from src.deploy.rollback import (
    RollbackManager, RollbackPolicy, RollbackTrigger, RollbackEvent,
)
from src.deploy.iam import (
    IAMManager, Role, Permission, AccessPolicy, ApiKey, AccessDecision,
)


# ═════════════════════════════════════════════════════════════════════
# Canary Deployment Tests
# ═════════════════════════════════════════════════════════════════════

class TestCanaryConfig:
    def test_default_4_steps(self):
        cfg = CanaryConfig.default("researcher", "v2.0", "v1.0")
        assert len(cfg.steps) == 4
        assert cfg.steps[0].traffic_weight == 5
        assert cfg.steps[1].traffic_weight == 25
        assert cfg.steps[2].traffic_weight == 50
        assert cfg.steps[3].traffic_weight == 100

    def test_custom_steps(self):
        cfg = CanaryConfig(
            agent_id="r", new_version="v2",
            steps=[
                CanaryStep(traffic_weight=10, observe_duration_s=60),
                CanaryStep(traffic_weight=100, observe_duration_s=120),
            ],
        )
        assert len(cfg.steps) == 2


class TestCanaryDeployment:
    @pytest.fixture
    def canary(self):
        cfg = CanaryConfig(
            agent_id="researcher",
            new_version="v2.0",
            previous_version="v1.0",
            steps=[
                CanaryStep(traffic_weight=5, observe_duration_s=0,
                           error_rate_threshold=0.01),
                CanaryStep(traffic_weight=25, observe_duration_s=0,
                           error_rate_threshold=0.01),
                CanaryStep(traffic_weight=50, observe_duration_s=0,
                           error_rate_threshold=0.01),
                CanaryStep(traffic_weight=100, observe_duration_s=0,
                           error_rate_threshold=0.01),
            ],
        )
        return CanaryDeployment(config=cfg)

    def test_start_sets_status_deploying(self, canary):
        canary.set_traffic_controller(lambda w: True)
        assert canary.start() is True
        assert canary.status == CanaryStatus.OBSERVING

    def test_start_no_steps(self):
        d = CanaryDeployment(config=CanaryConfig(agent_id="r", new_version="v2"))
        assert d.start() is False
        assert d.status == CanaryStatus.FAILED

    def test_run_sync_all_green(self, canary):
        canary.set_traffic_controller(lambda w: True)
        # All metrics pass
        canary.set_collector(lambda: MetricSnapshot(
            error_rate=0.0, latency_p95_ms=100,
            hallucination_score=1.0, total_requests=500,
        ))
        status = canary.run_sync()
        assert status == CanaryStatus.PROMOTED

    def test_run_sync_rollback_on_error(self, canary):
        canary.set_traffic_controller(lambda w: True)
        canary.set_rollback_handler(lambda a, v: True)
        # High error rate
        canary.set_collector(lambda: MetricSnapshot(
            error_rate=0.15, latency_p95_ms=100,
            hallucination_score=1.0, total_requests=500,
        ))
        status = canary.run_sync()
        assert status == CanaryStatus.ROLLED_BACK

    def test_run_sync_rollback_on_hallucination(self, canary):
        canary.set_traffic_controller(lambda w: True)
        canary.set_rollback_handler(lambda a, v: True)
        canary.set_collector(lambda: MetricSnapshot(
            error_rate=0.0, latency_p95_ms=100,
            hallucination_score=0.2, total_requests=500,
        ))
        status = canary.run_sync()
        assert status == CanaryStatus.ROLLED_BACK

    def test_current_traffic(self, canary):
        assert canary.current_traffic() == 5
        canary.current_step_index = 2
        assert canary.current_traffic() == 50

    def test_is_active(self, canary):
        assert canary.is_active is False
        canary.status = CanaryStatus.OBSERVING
        assert canary.is_active is True

    def test_elapsed_time(self, canary):
        assert canary.elapsed_s == 0.0
        canary.started_at = time.time() - 10
        assert 9.0 < canary.elapsed_s < 11.0


# ═════════════════════════════════════════════════════════════════════
# Rollback Manager Tests
# ═════════════════════════════════════════════════════════════════════

class TestRollbackPolicy:
    def test_defaults(self):
        p = RollbackPolicy(agent_id="r")
        assert p.error_rate_threshold == 0.05
        assert p.consecutive_failures == 3
        assert p.enabled is True

    def test_disabled(self):
        p = RollbackPolicy(agent_id="r", enabled=False)


class TestRollbackManager:
    @pytest.fixture
    def mgr(self):
        return RollbackManager()

    def test_evaluate_no_policy(self, mgr):
        triggers = mgr.evaluate("unknown", {"error_rate": 1.0})
        assert triggers == []

    def test_evaluate_clean_metrics(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r"))
        triggers = mgr.evaluate("r", {
            "error_rate": 0.01,
            "latency_p95_ms": 500,
            "hallucination_score": 0.9,
        })
        assert triggers == []

    def test_evaluate_error_rate_triggers(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", error_rate_threshold=0.05))
        triggers = mgr.evaluate("r", {"error_rate": 0.15})
        assert RollbackTrigger.ERROR_RATE in triggers

    def test_evaluate_latency_triggers(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", latency_p95_threshold_ms=2000))
        triggers = mgr.evaluate("r", {"latency_p95_ms": 5000})
        assert RollbackTrigger.LATENCY in triggers

    def test_evaluate_hallucination_triggers(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", hallucination_score_min=0.6))
        triggers = mgr.evaluate("r", {"hallucination_score": 0.3})
        assert RollbackTrigger.HALLUCINATION in triggers

    def test_evaluate_cost_spike(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", cost_spike_multiplier=3.0))
        triggers = mgr.evaluate("r", {"cost_per_request": 0.05}, baseline_cost=0.01)
        assert RollbackTrigger.COST_SPIKE in triggers

    def test_consecutive_failures_trigger_health_check(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", error_rate_threshold=0.05,
                                       consecutive_failures=2))
        # First failure
        mgr.evaluate("r", {"error_rate": 0.10})
        assert mgr.get_consecutive_failures("r") == 1
        # Second failure triggers HEALTH_CHECK
        triggers = mgr.evaluate("r", {"error_rate": 0.10})
        assert mgr.get_consecutive_failures("r") == 2
        assert RollbackTrigger.HEALTH_CHECK in triggers

    def test_consecutive_reset_on_success(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", error_rate_threshold=0.05))
        mgr.evaluate("r", {"error_rate": 0.10})
        assert mgr.get_consecutive_failures("r") == 1
        mgr.evaluate("r", {"error_rate": 0.01})
        assert mgr.get_consecutive_failures("r") == 0

    def test_execute_rollback(self, mgr):
        event = mgr.execute_rollback("r", "v2", "v1",
                                     trigger=RollbackTrigger.MANUAL,
                                     reason="testing")
        assert event.agent_id == "r"
        assert event.from_version == "v2"
        assert event.to_version == "v1"
        assert event.trigger == RollbackTrigger.MANUAL

    def test_execute_rollback_with_handler(self, mgr):
        calls = []
        mgr.set_handler(lambda a, f, t: calls.append((a, f, t)) or True)
        mgr.execute_rollback("r", "v2", "v1")
        assert len(calls) == 1
        assert calls[0] == ("r", "v2", "v1")

    def test_get_history(self, mgr):
        mgr.execute_rollback("a", "v3", "v2", trigger=RollbackTrigger.ERROR_RATE)
        mgr.execute_rollback("b", "v2", "v1", trigger=RollbackTrigger.LATENCY)
        assert len(mgr.get_history()) == 2
        assert len(mgr.get_history(agent_id="a")) == 1

    def test_cooldown_blocks_evaluation(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", error_rate_threshold=0.05,
                                       cooldown_s=3600))
        mgr.execute_rollback("r", "v2", "v1")
        # Immediately evaluate — should be blocked by cooldown
        triggers = mgr.evaluate("r", {"error_rate": 0.50})
        assert triggers == []

    def test_is_in_cooldown(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", cooldown_s=3600))
        assert mgr.is_in_cooldown("r") is False
        mgr.execute_rollback("r", "v2", "v1")
        assert mgr.is_in_cooldown("r") is True

    def test_disabled_policy_no_triggers(self, mgr):
        mgr.set_policy(RollbackPolicy(agent_id="r", error_rate_threshold=0.05,
                                       enabled=False))
        triggers = mgr.evaluate("r", {"error_rate": 1.0})
        assert triggers == []


# ═════════════════════════════════════════════════════════════════════
# IAM Tests
# ═════════════════════════════════════════════════════════════════════

class TestRole:
    def test_admin_has_all_permissions(self):
        role = Role(name="admin", permissions=[Permission.ADMIN_ALL])
        assert role.has_permission(Permission.AGENT_INVOKE) is True
        assert role.has_permission(Permission.COST_MANAGE) is True
        assert role.has_permission(Permission.SAFETY_OVERRIDE) is True

    def test_viewer_limited_permissions(self):
        role = Role(name="viewer", permissions=[
            Permission.CONFIG_READ, Permission.OBSERVE_READ,
        ])
        assert role.has_permission(Permission.CONFIG_READ) is True
        assert role.has_permission(Permission.AGENT_INVOKE) is False


class TestAccessPolicy:
    def test_empty_allows_all(self):
        p = AccessPolicy(name="open")
        assert p.can_access_agent("any") is True
        assert p.can_access_workflow("any") is True

    def test_allow_list_restricts(self):
        p = AccessPolicy(name="restricted",
                         allowed_agents=["researcher", "copywriter"])
        assert p.can_access_agent("researcher") is True
        assert p.can_access_agent("ops-monitor") is False

    def test_deny_list_overrides_allow(self):
        p = AccessPolicy(name="blocked",
                         denied_agents=["ops-monitor"])
        assert p.can_access_agent("researcher") is True
        assert p.can_access_agent("ops-monitor") is False


class TestIAMManager:
    @pytest.fixture
    def iam(self):
        mgr = IAMManager()
        mgr.create_default_roles()
        return mgr

    def test_default_roles_created(self, iam):
        assert iam.get_role("admin") is not None
        assert iam.get_role("developer") is not None
        assert iam.get_role("viewer") is not None

    def test_create_api_key(self, iam):
        raw, key = iam.create_api_key(name="test-key", role="developer")
        assert raw.startswith("ak-")
        assert key.role == "developer"
        assert key.enabled is True

    def test_authenticate_valid_key(self, iam):
        raw, key = iam.create_api_key(name="test", role="developer")
        authed = iam.authenticate(raw)
        assert authed is not None
        assert authed.role == "developer"

    def test_authenticate_invalid_key(self, iam):
        assert iam.authenticate("invalid-key") is None

    def test_authorize_admin_can_do_anything(self, iam):
        raw, _ = iam.create_api_key(name="admin-key", role="admin")
        decision = iam.authorize(raw, Permission.AGENT_DELETE, agent_id="any")
        assert decision.allowed is True

    def test_authorize_viewer_cannot_invoke(self, iam):
        raw, _ = iam.create_api_key(name="viewer-key", role="viewer")
        decision = iam.authorize(raw, Permission.AGENT_INVOKE)
        assert decision.allowed is False
        assert "lacks permission" in decision.reason

    def test_authorize_developer_can_invoke(self, iam):
        raw, _ = iam.create_api_key(name="dev-key", role="developer")
        decision = iam.authorize(raw, Permission.AGENT_INVOKE)
        assert decision.allowed is True

    def test_authorize_policy_restricts_agent(self, iam):
        iam.create_policy("researcher-only", AccessPolicy(
            name="researcher-only",
            allowed_agents=["researcher"],
        ))
        raw, _ = iam.create_api_key(name="restricted-key", role="developer",
                                     policies=["researcher-only"])
        assert iam.authorize(raw, Permission.AGENT_INVOKE,
                            agent_id="researcher").allowed is True
        assert iam.authorize(raw, Permission.AGENT_INVOKE,
                            agent_id="ops-monitor").allowed is False

    def test_revoke_key(self, iam):
        raw, key = iam.create_api_key(name="temp", role="developer")
        assert iam.authenticate(raw) is not None
        iam.revoke_key(key.key_hash)
        assert iam.authenticate(raw) is None

    def test_rotate_key(self, iam):
        raw_old, key_old = iam.create_api_key(name="rotating", role="developer")
        result = iam.rotate_key(key_old.key_hash, name="rotating")
        assert result is not None
        raw_new, key_new = result
        # Old key revoked
        assert iam.authenticate(raw_old) is None
        # New key works
        assert iam.authenticate(raw_new) is not None

    def test_expired_key_denied(self, iam):
        raw, key = iam.create_api_key(name="expiring", role="developer",
                                       expires_in_days=-1)  # expired immediately
        decision = iam.authorize(raw, Permission.AGENT_INVOKE)
        assert decision.allowed is False
        assert "expired" in decision.reason

    def test_get_stats(self, iam):
        iam.create_api_key(name="k1", role="developer")
        iam.create_api_key(name="k2", role="viewer")
        stats = iam.get_stats()
        assert stats["roles"] == 4
        assert stats["api_keys_total"] == 2
        assert stats["api_keys_active"] == 2
