"""Reflection round cap + budget integration tests."""

import pytest

from src.cost.budget import BudgetManager, BudgetConfig, BudgetTier
from src.planning.strategies import (
    ReflexionPlanner, ReActPlanner, Planner, PlanningStrategy,
)


class TestReflexionPlannerBudget:
    """Test ReflexionPlanner with budget integration."""

    @pytest.fixture
    def planner(self):
        return ReflexionPlanner(max_reflections=3, max_reflection_rounds=5)

    @pytest.fixture
    def budget(self):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=0, hard_limit=False, warn_threshold=0.9,
        ))
        return mgr

    def test_max_reflection_rounds_split(self, planner):
        """Verify old max_reflections is now reflection_history_window."""
        assert planner.reflection_history_window == 3
        assert planner.max_reflection_rounds == 5

    def test_plan_without_budget_uses_max_rounds(self, planner):
        result = planner.plan("test goal", current_round=1)
        assert result.strategy == PlanningStrategy.REFLEXION
        assert result.metadata.get("effective_max") == 5

    def test_plan_with_budget_manager(self, planner, budget):
        result = planner.plan("test goal", budget_manager=budget, current_round=1)
        assert result.strategy == PlanningStrategy.REFLEXION

    def test_effective_max_returns_default_when_no_budget(self, planner):
        effective = planner._compute_effective_max_rounds("test")
        assert effective == 5

    def test_effective_max_honors_budget(self, planner, budget):
        effective = planner._compute_effective_max_rounds("test", budget_manager=budget)
        assert effective == 5  # 默认无超额

    def test_effective_max_zero_when_hard_exceeded(self, planner):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=100, hard_limit=True,
        ))
        # Exhaust budget
        mgr.record(200, workflow_id="reflection")
        effective = planner._compute_effective_max_rounds("test", budget_manager=mgr)
        assert effective == 0

    def test_effective_max_halved_when_soft_exceeded(self, planner):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=100, hard_limit=False,
        ))
        mgr.record(200, workflow_id="reflection")  # Exceed soft limit
        effective = planner._compute_effective_max_rounds("test", budget_manager=mgr)
        assert effective == 2  # 5 // 2 = 2

    def test_should_continue_within_bounds(self, planner):
        assert planner.should_continue_reflection(1, effective_max=5) is True
        assert planner.should_continue_reflection(3, effective_max=5) is True

    def test_should_exit_past_max(self, planner):
        assert planner.should_continue_reflection(6, effective_max=5) is False

    def test_diminishing_returns_exit(self, planner):
        """Two rounds with < 0.05 improvement should exit."""
        planner._reflection_history = [
            {"confidence": 0.50},
            {"confidence": 0.52},  # delta = 0.02 < 0.05
        ]
        assert planner.should_continue_reflection(3, effective_max=5) is False

    def test_sufficient_improvement_continues(self, planner):
        """Two rounds with >= 0.05 improvement should continue."""
        planner._reflection_history = [
            {"confidence": 0.50},
            {"confidence": 0.65},  # delta = 0.15 >= 0.05
        ]
        assert planner.should_continue_reflection(3, effective_max=5) is True

    def test_budget_exhausted_stops_reflection(self, planner):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=100, hard_limit=True,
        ))
        mgr.record(200, workflow_id="reflection")
        assert planner.should_continue_reflection(2, effective_max=5, budget_manager=mgr) is False

    def test_plan_records_budget_on_execution(self, planner, budget):
        status_before = budget.check(workflow_id="reflection")
        tokens_before = status_before.tokens_used_this_run
        planner.plan("test budget record", budget_manager=budget, current_round=1)
        status_after = budget.check(workflow_id="reflection")
        assert status_after.tokens_used_this_run > tokens_before


class TestReActPlannerBudget:
    """Test ReActPlanner with budget awareness."""

    @pytest.fixture
    def budget(self):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=10_000, hard_limit=False, warn_threshold=0.9,
        ))
        return mgr

    def test_react_planner_accepts_budget(self, budget):
        planner = ReActPlanner(max_rounds=10)
        result = planner.plan("test", budget_manager=budget)
        assert result.strategy == PlanningStrategy.REACT

    def test_effective_react_rounds_without_budget(self):
        planner = ReActPlanner(max_rounds=10)
        rounds = planner._compute_effective_react_rounds()
        assert rounds == 4  # min(10//2, 4)

    def test_effective_react_rounds_with_sufficient_budget(self, budget):
        planner = ReActPlanner(max_rounds=10)
        rounds = planner._compute_effective_react_rounds(budget_manager=budget)
        assert rounds == 4

    def test_effective_react_rounds_tight_budget(self):
        planner = ReActPlanner(max_rounds=10)
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=100, hard_limit=True,
        ))
        mgr.record(200, workflow_id="reflection")
        rounds = planner._compute_effective_react_rounds(budget_manager=mgr)
        assert rounds == 1  # hard exceeded → min = 1


class TestPlannerAutoBudget:
    """Test Planner.auto_plan() budget awareness."""

    @pytest.fixture
    def budget(self):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=10_000, hard_limit=False, warn_threshold=0.9,
        ))
        return mgr

    def test_auto_plan_avoids_reflexion_when_budget_tight(self):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=100, hard_limit=True,
        ))
        mgr.record(200, workflow_id="reflection")

        p = Planner()
        result = p.auto_plan(
            "test", has_errors=["error1"],
            budget_manager=mgr,
        )
        # Should fall back to ReAct instead of Reflexion
        assert result.strategy == PlanningStrategy.REACT

    def test_auto_plan_uses_reflexion_with_sufficient_budget(self, budget):
        p = Planner()
        result = p.auto_plan(
            "test", has_errors=["error1"],
            budget_manager=budget,
        )
        assert result.strategy == PlanningStrategy.REFLEXION

    def test_auto_plan_no_errors_defaults_to_cot(self, budget):
        p = Planner()
        result = p.auto_plan("test", budget_manager=budget)
        assert result.strategy == PlanningStrategy.COT


class TestBudgetWorkflowTier:
    """Test WORKFLOW tier registration in BudgetManager."""

    def test_workflow_tier_registered_in_defaults(self):
        mgr = BudgetManager()
        mgr.configure_defaults()
        key = "workflow:reflection"
        assert key in mgr._configs
        assert mgr._configs[key].tier == BudgetTier.WORKFLOW
        assert mgr._configs[key].per_run_limit == 20_000

    def test_workflow_can_proceed_tracks_per_run(self):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="test_workflow",
            per_run_limit=1000, hard_limit=True,
        ))
        assert mgr.can_proceed(500, workflow_id="test_workflow") is True
        mgr.record(600, workflow_id="test_workflow")
        assert mgr.can_proceed(500, workflow_id="test_workflow") is False

    def test_reset_run_clears_workflow_count(self):
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="test_workflow",
            per_run_limit=1000, hard_limit=True,
        ))
        mgr.record(800, workflow_id="test_workflow")
        assert mgr.can_proceed(300, workflow_id="test_workflow") is False
        mgr.reset_run("test_workflow")
        assert mgr.can_proceed(300, workflow_id="test_workflow") is True
