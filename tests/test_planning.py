"""Planning strategies tests — CoT, ToT, ReAct, Reflexion."""

import pytest

from src.planning.strategies import (
    Planner, PlanResult, PlanStep, PlanningStrategy,
    ChainOfThought, TreeOfThought, ReActPlanner, ReflexionPlanner,
)


class TestPlanStep:
    """Test PlanStep dataclass."""

    def test_defaults(self):
        step = PlanStep()
        assert len(step.id) == 8
        assert step.confidence == 0.8
        assert step.estimated_tokens == 500
        assert step.depends_on == []

    def test_with_data(self):
        step = PlanStep(
            description="Analyze problem",
            action="think",
            confidence=0.95,
            estimated_tokens=200,
            tool_name="search",
            tool_args={"q": "test"},
            alternatives=["alt1", "alt2"],
        )
        assert step.action == "think"
        assert len(step.alternatives) == 2


class TestPlanResult:
    """Test PlanResult dataclass."""

    def test_defaults(self):
        result = PlanResult(strategy=PlanningStrategy.COT)
        assert result.strategy == PlanningStrategy.COT
        assert result.goal == ""
        assert result.confidence == 1.0

    def test_with_steps(self):
        steps = [
            PlanStep(description="Step 1", confidence=0.9),
            PlanStep(description="Step 2", confidence=0.8),
        ]
        result = PlanResult(
            strategy=PlanningStrategy.COT,
            goal="Test goal",
            steps=steps,
            total_steps=2,
            total_estimated_tokens=700,
        )
        assert result.total_steps == 2


class TestChainOfThought:
    """Test ChainOfThought planner."""

    @pytest.fixture
    def cot(self):
        return ChainOfThought()

    def test_plan_returns_steps(self, cot):
        result = cot.plan("What is 2+2?")
        assert result.strategy == PlanningStrategy.COT
        assert len(result.steps) >= 2  # at least analyze + verify
        assert result.total_steps >= 2

    def test_plan_has_reasoning(self, cot):
        result = cot.plan("Analyze recent AI trends")
        assert "CoT" in result.reasoning

    def test_first_step_is_analysis(self, cot):
        result = cot.plan("Solve math problem")
        assert "分析" in result.steps[0].description

    def test_last_step_is_verify(self, cot):
        result = cot.plan("Explain quantum computing")
        assert result.steps[-1].action == "verify"

    def test_chain_dependencies(self, cot):
        result = cot.plan("Research task", max_steps=5)
        for i in range(1, len(result.steps)):
            step = result.steps[i]
            if step.depends_on:
                # Depends on previous step
                assert step.depends_on[0] == result.steps[i - 1].id

    def test_confidence_decays(self, cot):
        result = cot.plan("Complex analysis", max_steps=7)
        # Overall confidence = product of all step confidences
        assert 0 < result.confidence < 1.0

    def test_cost_estimate(self, cot):
        result = cot.plan("Task")
        assert result.total_estimated_cost > 0
        assert result.total_estimated_cost < 0.01  # ~$3/M tokens

    def test_decompose(self, cot):
        parts = cot._decompose("Research AI safety", 4)
        assert len(parts) == 4
        assert all("步骤" in p for p in parts)

    def test_chain_confidence_perfect(self, cot):
        steps = [PlanStep(confidence=1.0), PlanStep(confidence=1.0)]
        assert cot._chain_confidence(steps) == 1.0

    def test_chain_confidence_empty(self, cot):
        assert cot._chain_confidence([]) == 0.0

    def test_plan_max_steps(self, cot):
        result = cot.plan("Task", max_steps=3)
        # 1 analyze + (3-2)=1 decompose + 1 verify = 3
        assert result.total_steps <= 3


class TestTreeOfThought:
    """Test TreeOfThought planner."""

    @pytest.fixture
    def tot(self):
        return TreeOfThought(beam_width=3, max_depth=5)

    def test_plan_has_branches(self, tot):
        result = tot.plan("Design a better search algorithm")
        assert result.strategy == PlanningStrategy.TOT
        assert result.alternatives_considered > 0
        assert len(result.steps) >= 2  # root + branches

    def test_plan_evaluate_step(self, tot):
        result = tot.plan("Creative writing task")
        actions = [s.action for s in result.steps]
        assert "verify" in actions  # evaluate step

    def test_beam_width_controls_branches(self, tot):
        result = tot.plan("Task")
        # Branch steps should respect beam_width
        branch_steps = [s for s in result.steps if s.action == "think" and s.depends_on]
        assert len(branch_steps) <= tot.beam_width

    def test_generate_branches(self, tot):
        branches = tot._generate_branches("Task")
        assert len(branches) == 4
        assert all(b.startswith("方案") for b in branches)

    def test_reasoning_includes_beam_info(self, tot):
        result = tot.plan("Task")
        assert f"beam={tot.beam_width}" in result.reasoning

    def test_confidence_boosted(self, tot):
        # ToT confidence is min(0.9, chain_confidence + 0.1)
        result = tot.plan("Task")
        assert result.confidence <= 0.9


class TestReActPlanner:
    """Test ReActPlanner."""

    @pytest.fixture
    def react(self):
        return ReActPlanner(max_rounds=10)

    def test_plan_interleaves_thought_action(self, react):
        result = react.plan("Find latest AI papers and summarize",
                           available_tools=["web_search", "calculator"])
        assert result.strategy == PlanningStrategy.REACT
        # Should have thought + action pairs
        actions = [s.action for s in result.steps]
        assert "think" in actions
        assert "tool_call" in actions

    def test_plan_with_tools(self, react):
        result = react.plan("Task", available_tools=["search", "read", "analyze"])
        tool_calls = [s for s in result.steps if s.action == "tool_call"]
        assert len(tool_calls) > 0
        # Should use provided tools
        tools_used = {s.tool_name for s in tool_calls}
        assert tools_used.issubset({"search", "read", "analyze"})

    def test_plan_default_tools(self, react):
        result = react.plan("Search the web")
        tool_calls = [s for s in result.steps if s.action == "tool_call"]
        assert len(tool_calls) > 0

    def test_plan_starts_with_thought(self, react):
        result = react.plan("Task")
        assert result.steps[0].action == "think"

    def test_reasoning_includes_tool_count(self, react):
        result = react.plan("Task", available_tools=["t1", "t2", "t3"])
        assert "tools" in result.reasoning


class TestReflexionPlanner:
    """Test ReflexionPlanner."""

    @pytest.fixture
    def reflexion(self):
        return ReflexionPlanner(max_reflections=3)

    def test_plan_without_errors(self, reflexion):
        result = reflexion.plan("Build a web scraper")
        assert result.strategy == PlanningStrategy.REFLEXION
        assert result.total_steps >= 2  # plan + execute + evaluate

    def test_plan_with_errors(self, reflexion):
        result = reflexion.plan(
            "Build a web scraper",
            previous_errors=["Timeout on request", "Parsing failed"],
        )
        # Should have reflect steps
        reflect_steps = [s for s in result.steps if s.action == "reflect"]
        assert len(reflect_steps) >= 2

    def test_plan_with_errors_lower_confidence(self, reflexion):
        no_errors = reflexion.plan("Task")
        with_errors = reflexion.plan("Task", previous_errors=["Error 1"])
        assert with_errors.confidence <= no_errors.confidence

    def test_reflection_capped(self, reflexion):
        many_errors = [f"Error {i}" for i in range(10)]
        result = reflexion.plan("Task", previous_errors=many_errors)
        reflect_steps = [s for s in result.steps if s.action == "reflect"]
        assert len(reflect_steps) <= reflexion.reflection_history_window

    def test_has_self_evaluation(self, reflexion):
        result = reflexion.plan("Task")
        # Last steps should include self-evaluation
        verify_steps = [s for s in result.steps if s.action == "verify"]
        assert len(verify_steps) >= 1

    def test_reflection_history(self, reflexion):
        reflexion.plan("Task 1", previous_errors=["e1"])
        reflexion.plan("Task 2", previous_errors=["e2"])
        history = reflexion.get_history()
        assert len(history) == 2

    def test_history_entries_have_fields(self, reflexion):
        reflexion.plan("Task", previous_errors=["e1"])
        history = reflexion.get_history()
        assert "goal" in history[0]
        assert "errors" in history[0]
        assert "confidence" in history[0]


class TestPlanner:
    """Test unified Planner with auto-strategy selection."""

    @pytest.fixture
    def planner(self):
        return Planner()

    def test_plan_with_strategy(self, planner):
        result = planner.plan("Task", strategy=PlanningStrategy.COT)
        assert result.strategy == PlanningStrategy.COT

        result = planner.plan("Task", strategy=PlanningStrategy.TOT)
        assert result.strategy == PlanningStrategy.TOT

        result = planner.plan("Task", strategy=PlanningStrategy.REACT)
        assert result.strategy == PlanningStrategy.REACT

        result = planner.plan("Task", strategy=PlanningStrategy.REFLEXION)
        assert result.strategy == PlanningStrategy.REFLEXION

    def test_plan_unknown_strategy(self, planner):
        result = planner.plan("Task", strategy="unknown")  # type: ignore
        assert result.goal == "Task"

    # ── Auto-plan heuristics ─────────────────────────

    def test_auto_plan_defaults_to_cot(self, planner):
        result = planner.auto_plan("Simple task")
        assert result.strategy == PlanningStrategy.COT

    def test_auto_plan_errors_trigger_reflexion(self, planner):
        result = planner.auto_plan("Task", has_errors=["e1"])
        assert result.strategy == PlanningStrategy.REFLEXION

    def test_auto_plan_tools_trigger_react(self, planner):
        result = planner.auto_plan("Task", requires_tools=True)
        assert result.strategy == PlanningStrategy.REACT

    def test_auto_plan_high_complexity_trigger_tot(self, planner):
        result = planner.auto_plan("Task", complexity="high")
        assert result.strategy == PlanningStrategy.TOT

    def test_auto_plan_errors_override_all(self, planner):
        # Errors take highest priority
        result = planner.auto_plan(
            "Task",
            complexity="high",
            requires_tools=True,
            has_errors=["e1"],
        )
        assert result.strategy == PlanningStrategy.REFLEXION


class TestPlanningStrategy:
    """Test PlanningStrategy enum."""

    def test_all_strategies(self):
        strategies = list(PlanningStrategy)
        assert PlanningStrategy.COT in strategies
        assert PlanningStrategy.TOT in strategies
        assert PlanningStrategy.REACT in strategies
        assert PlanningStrategy.REFLEXION in strategies
        assert len(strategies) == 4
