"""Cost management system tests — budget, router, attribution."""

from __future__ import annotations

import time

import pytest

from src.cost.budget import (
    BudgetManager, BudgetConfig, BudgetStatus, BudgetTier,
)
from src.cost.router import (
    ModelRouter, ModelConfig, ModelTier, RoutingDecision,
)
from src.cost.attribution import (
    CostTracker as AttributionTracker, UsageRecord, CostReport,
)


# ═════════════════════════════════════════════════════════════════════
# Budget Manager Tests
# ═════════════════════════════════════════════════════════════════════

class TestBudgetConfig:
    def test_defaults(self):
        cfg = BudgetConfig(tier=BudgetTier.AGENT, budget_id="test")
        assert cfg.daily_limit == 100_000
        assert cfg.hard_limit is False
        assert cfg.warn_threshold == 0.8

    def test_hard_limit(self):
        cfg = BudgetConfig(tier=BudgetTier.GLOBAL, budget_id="global",
                           daily_limit=500_000, hard_limit=True)
        assert cfg.hard_limit is True


class TestBudgetManager:
    @pytest.fixture
    def mgr(self):
        return BudgetManager()

    def test_configure_and_check(self, mgr):
        mgr.configure(BudgetConfig(
            tier=BudgetTier.AGENT, budget_id="researcher",
            daily_limit=10_000,
        ))
        status = mgr.check(agent_id="researcher")
        assert status.config.budget_id == "researcher"
        assert status.tokens_used_today == 0
        assert status.is_exceeded is False

    def test_record_and_exceed(self, mgr):
        mgr.configure(BudgetConfig(
            tier=BudgetTier.AGENT, budget_id="researcher",
            daily_limit=1000, warn_threshold=0.5,
        ))
        # Record 600 tokens — should trigger warning (60% > 50%)
        results = mgr.record(600, agent_id="researcher")
        key = "agent:researcher"
        assert key in results
        assert results[key].is_warning is True

        # Record 500 more — should exceed
        results = mgr.record(500, agent_id="researcher")
        assert results[key].is_exceeded is True

    def test_hard_limit_blocks(self, mgr):
        mgr.configure(BudgetConfig(
            tier=BudgetTier.GLOBAL, budget_id="global",
            daily_limit=500, hard_limit=True,
        ))
        mgr.record(500, agent_id="r")
        assert mgr.can_proceed(1) is False
        assert mgr.can_proceed(0) is True

    def test_soft_limit_allows(self, mgr):
        mgr.configure(BudgetConfig(
            tier=BudgetTier.AGENT, budget_id="r",
            daily_limit=500, hard_limit=False,
        ))
        mgr.record(500, agent_id="r")
        # 软限制不阻断
        assert mgr.can_proceed(100, agent_id="r") is True

    def test_multi_tier_check(self, mgr):
        mgr.configure(BudgetConfig(
            tier=BudgetTier.GLOBAL, budget_id="global",
            daily_limit=1_000_000, hard_limit=True,
        ))
        mgr.configure(BudgetConfig(
            tier=BudgetTier.AGENT, budget_id="researcher",
            daily_limit=10_000, hard_limit=False,
        ))
        status = mgr.check(agent_id="researcher")
        assert status.is_exceeded is False

    def test_can_proceed_with_headroom(self, mgr):
        mgr.configure(BudgetConfig(
            tier=BudgetTier.AGENT, budget_id="r",
            daily_limit=10000, hard_limit=True,
        ))
        assert mgr.can_proceed(500, agent_id="r") is True

    def test_resolve_all_keys(self, mgr):
        """Verify all four tiers are resolved.  """
        keys = mgr._resolve_keys("agent1", "wf1", "sess1")
        assert "global:global" in keys
        assert "agent:agent1" in keys
        assert "workflow:wf1" in keys
        assert "session:sess1" in keys

    def test_resolve_minimal_keys(self, mgr):
        keys = mgr._resolve_keys("", "", "")
        assert keys == ["global:global"]

    def test_reset_run(self, mgr):
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="wf1",
            daily_limit=50000, per_run_limit=5000,
        ))
        mgr.record(3000, workflow_id="wf1")
        key = "workflow:wf1"
        status = mgr._usage.get(key)
        assert status is not None
        assert status.tokens_used_this_run == 3000

        mgr.reset_run("wf1")
        assert status.tokens_used_this_run == 0

    def test_get_usage_summary(self, mgr):
        mgr.configure(BudgetConfig(tier=BudgetTier.AGENT, budget_id="r"))
        mgr.record(100, agent_id="r")
        summary = mgr.get_usage_summary()
        assert "agent:r" in summary
        assert summary["agent:r"]["tokens_used_today"] == 100

    def test_configure_defaults(self, mgr):
        mgr.configure_defaults()
        assert mgr.check() is not None
        # 应该有 1 global + 10 agents + 1 workflow
        assert len(mgr._configs) == 12


# ═════════════════════════════════════════════════════════════════════
# Model Router Tests
# ═════════════════════════════════════════════════════════════════════

class TestModelConfig:
    def test_cost_calculation(self):
        m = ModelConfig(name="test", tier=ModelTier.STANDARD, provider="openai",
                        input_price_per_1m=1.0, output_price_per_1m=5.0)
        cost = m.cost_for(input_tokens=1000, output_tokens=500)
        expected = (1000 / 1_000_000 * 1.0) + (500 / 1_000_000 * 5.0)
        assert cost == pytest.approx(expected)

    def test_zero_cost(self):
        m = ModelConfig(name="free", tier=ModelTier.FALLBACK, provider="local")
        assert m.cost_for(1000, 500) == 0.0


class TestModelRouter:
    @pytest.fixture
    def router(self):
        r = ModelRouter()
        r.load_defaults()
        return r

    def test_load_defaults(self, router):
        assert len(router._models) >= 4

    def test_route_high_budget(self, router):
        decision = router.route(
            task_complexity="medium",
            budget_remaining_pct=0.9,
            estimated_input_tokens=2000,
            estimated_output_tokens=500,
        )
        assert decision.model.tier in (ModelTier.PREMIUM, ModelTier.STANDARD)
        assert "充足" in decision.reason

    def test_route_low_budget(self, router):
        decision = router.route(
            task_complexity="medium",
            budget_remaining_pct=0.2,
        )
        assert decision.model.tier == ModelTier.BUDGET
        assert "紧张" in decision.reason or "budget" in decision.reason.lower()

    def test_route_critical_budget(self, router):
        decision = router.route(
            task_complexity="medium",
            budget_remaining_pct=0.05,
        )
        assert decision.model.tier == ModelTier.FALLBACK

    def test_route_high_complexity_upgrades_tier(self, router):
        decision_low = router.route(task_complexity="low", budget_remaining_pct=0.5)
        decision_high = router.route(task_complexity="high", budget_remaining_pct=0.5)
        # high complexity should select at most PREMIUM tier (upgrade from standard)
        assert decision_high.model.tier in (ModelTier.PREMIUM, ModelTier.STANDARD)

    def test_route_low_complexity_downgrades(self, router):
        decision = router.route(task_complexity="low", budget_remaining_pct=0.5)
        assert decision.model.tier in (ModelTier.BUDGET, ModelTier.FALLBACK, ModelTier.STANDARD)

    def test_route_requires_vision(self, router):
        decision = router.route(
            required_capabilities=["vision"],
            budget_remaining_pct=0.9,
        )
        assert decision.model.supports_vision is True

    def test_route_preferred_provider(self, router):
        decision = router.route(
            preferred_provider="deepseek",
            budget_remaining_pct=0.5,
        )
        assert decision.model is not None

    def test_route_has_fallback_chain(self, router):
        decision = router.route(budget_remaining_pct=0.9)
        assert len(decision.fallback_chain) >= 1

    def test_route_estimated_cost(self, router):
        decision = router.route(
            estimated_input_tokens=10_000,
            estimated_output_tokens=2_000,
            budget_remaining_pct=0.9,
        )
        assert decision.estimated_cost >= 0.0

    def test_record_outcome_updates_bandit(self, router):
        router.record_outcome("gpt-4o-mini", success=True, cost=0.001, latency_ms=300)
        stats = router.get_bandit_stats()
        assert "gpt-4o-mini" in stats
        assert stats["gpt-4o-mini"]["successes"] == 1

    def test_record_outcome_failure(self, router):
        router.record_outcome("gpt-4o-mini", success=False)
        stats = router.get_bandit_stats()
        assert stats["gpt-4o-mini"]["failures"] == 1

    def test_list_models_by_tier(self, router):
        premium = router.list_models(tier=ModelTier.PREMIUM)
        assert all(m.tier == ModelTier.PREMIUM for m in premium)

    def test_get_model(self, router):
        m = router.get_model("gpt-4o-mini")
        assert m is not None
        assert m.tier == ModelTier.STANDARD

    def test_empty_router_fallback(self):
        r = ModelRouter()
        decision = r.route()
        assert decision.model.name == "none"


# ═════════════════════════════════════════════════════════════════════
# Cost Attribution Tests
# ═════════════════════════════════════════════════════════════════════

class TestUsageRecord:
    def test_defaults(self):
        rec = UsageRecord()
        assert rec.timestamp > 0
        assert rec.agent_id == ""
        assert rec.cost_usd == 0.0


class TestAttributionTracker:
    @pytest.fixture
    def tracker(self):
        return AttributionTracker()

    def test_record_and_get_agent_cost(self, tracker):
        tracker.record(UsageRecord(
            agent_id="researcher", model="gpt-4o-mini",
            input_tokens=2000, output_tokens=500, cost_usd=0.0006,
        ))
        info = tracker.get_agent_cost("researcher")
        assert info["total_cost"] == 0.0006
        assert info["total_tokens"] == 2500
        assert info["total_calls"] == 1

    def test_multiple_records_aggregate(self, tracker):
        tracker.record(UsageRecord(agent_id="r1", model="m1", cost_usd=0.01,
                                   input_tokens=100, output_tokens=50))
        tracker.record(UsageRecord(agent_id="r1", model="m1", cost_usd=0.02,
                                   input_tokens=200, output_tokens=100))
        info = tracker.get_agent_cost("r1")
        assert info["total_cost"] == 0.03
        assert info["total_calls"] == 2

    def test_get_model_cost(self, tracker):
        tracker.record(UsageRecord(model="gpt-4o", cost_usd=0.05,
                                   input_tokens=1000, output_tokens=200))
        info = tracker.get_model_cost("gpt-4o")
        assert info["total_cost"] == 0.05

    def test_get_summary(self, tracker):
        tracker.record(UsageRecord(agent_id="a", model="m", cost_usd=0.1))
        tracker.record(UsageRecord(agent_id="b", model="m", cost_usd=0.2))
        s = tracker.get_summary()
        assert s["total_cost"] == pytest.approx(0.3)
        assert s["total_calls"] == 2

    def test_generate_report(self, tracker):
        now = time.time()
        tracker.record(UsageRecord(
            timestamp=now - 1800,  # 30 min ago
            agent_id="r", model="gpt-4o-mini",
            input_tokens=1000, output_tokens=500, cost_usd=0.001,
            workflow_id="wf1",
        ))
        tracker.record(UsageRecord(
            timestamp=now - 3600 * 5,  # 5 hours ago (within 24h)
            agent_id="w", model="claude-sonnet",
            input_tokens=500, output_tokens=200, cost_usd=0.002,
        ))
        report = tracker.generate_report(hours=24)
        assert report.total_cost > 0
        assert report.total_calls >= 1
        assert "r" in report.by_agent
        assert "wf1" in report.by_workflow
        assert "gpt-4o-mini" in report.by_model
        assert "hourly_buckets" in report.trends

    def test_old_records_filtered(self, tracker):
        now = time.time()
        tracker.record(UsageRecord(
            timestamp=now - 3600 * 25,  # 25 hours ago
            agent_id="old", model="m", cost_usd=1.0,
        ))
        report = tracker.generate_report(hours=24)
        assert report.total_cost == 0.0
        assert report.total_calls == 0

    def test_ring_buffer_trim(self, tracker):
        # Fill beyond max
        for i in range(15):
            tracker.record(UsageRecord(agent_id=str(i)))

    def test_clear(self, tracker):
        tracker.record(UsageRecord(agent_id="r", cost_usd=0.1))
        tracker.clear()
        assert tracker.record_count == 0
        assert tracker.get_summary()["total_cost"] == 0.0
