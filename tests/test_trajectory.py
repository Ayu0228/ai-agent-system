"""Trajectory evaluator tests — multi-step execution path evaluation."""

import pytest

from src.evaluation.trajectory import (
    TrajectoryEvaluator, TrajectoryStep, TrajectoryResult,
    StepQuality,
)


class TestStepQuality:
    """Test StepQuality enum."""

    def test_all_qualities(self):
        assert StepQuality.OPTIMAL.value == "optimal"
        assert StepQuality.ACCEPTABLE.value == "acceptable"
        assert StepQuality.SUBOPTIMAL.value == "suboptimal"
        assert StepQuality.WRONG.value == "wrong"
        assert StepQuality.REDUNDANT.value == "redundant"


class TestTrajectoryStep:
    """Test TrajectoryStep dataclass."""

    def test_defaults(self):
        step = TrajectoryStep(step_id="s1")
        assert step.step_id == "s1"
        assert step.quality == StepQuality.ACCEPTABLE
        assert step.is_correct is True
        assert step.status == "success"

    def test_with_data(self):
        step = TrajectoryStep(
            step_id="s2",
            agent_id="researcher",
            action="tool_call",
            tool_name="web_search",
            tool_input={"q": "test"},
            expected_tool="web_search",
            quality=StepQuality.OPTIMAL,
        )
        assert step.tool_name == "web_search"
        assert step.quality == StepQuality.OPTIMAL


class TestTrajectoryResult:
    """Test TrajectoryResult dataclass."""

    def test_defaults(self):
        result = TrajectoryResult()
        assert result.trajectory_score == 1.0
        assert result.hidden_failure_detected is False
        assert result.final_correct is True

    def test_overall_property(self):
        result = TrajectoryResult(trajectory_score=0.85)
        assert result.overall == 0.85

    def test_hidden_failure_property(self):
        result = TrajectoryResult(hidden_failure_detected=True)
        assert result.hidden_failure is True


class TestTrajectoryEvaluator:
    """Test TrajectoryEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return TrajectoryEvaluator()

    # ── Record & Trace ────────────────────────────────

    def test_start_trace(self, evaluator):
        evaluator.start_trace("trace-1")
        assert "trace-1" in evaluator._traces

    def test_record_step(self, evaluator):
        step = TrajectoryStep(step_id="s1", agent_id="a1")
        evaluator.record_step(step)
        trace = evaluator.get_trace()
        assert len(trace) == 1
        assert trace[0].step_id == "s1"

    def test_get_trace_empty(self, evaluator):
        assert evaluator.get_trace("nonexistent") == []

    # ── Evaluate empty ────────────────────────────────

    def test_evaluate_empty(self, evaluator):
        result = evaluator.evaluate("empty-trace")
        assert result.trace_id == "empty-trace"
        assert result.total_steps == 0

    # ── Tool selection scoring ────────────────────────

    def test_tool_selection_perfect(self, evaluator):
        for i in range(3):
            evaluator.record_step(TrajectoryStep(
                step_id=f"s{i}", action="tool_call",
                tool_name=f"tool_{i}",
                quality=StepQuality.OPTIMAL,
            ))
        result = evaluator.evaluate()
        assert result.tool_selection_score == 1.0

    def test_tool_selection_with_wrong(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", action="tool_call",
            tool_name="wrong_tool", quality=StepQuality.WRONG,
            expected_tool="correct_tool",
        ))
        evaluator.record_step(TrajectoryStep(
            step_id="s2", action="tool_call",
            tool_name="right_tool", quality=StepQuality.OPTIMAL,
        ))
        result = evaluator.evaluate()
        # 1 wrong out of 2 → penalty 0.5 * (1/2) = 0.25 → score 0.75
        assert result.tool_selection_score == 0.75
        assert result.wrong_steps == 1
        assert result.optimal_steps == 1
        assert len(result.issues) >= 1

    def test_no_tool_steps(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", action="reasoning", reasoning="thinking...",
        ))
        result = evaluator.evaluate()
        assert result.tool_selection_score == 1.0  # no tool steps = perfect by default

    # ── Hidden failure detection ──────────────────────

    def test_hidden_failure_wrong_tools_correct_answer(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", action="tool_call",
            tool_name="wrong_search", quality=StepQuality.WRONG,
            expected_tool="correct_search",
        ))
        evaluator.record_step(TrajectoryStep(
            step_id="s2", action="tool_call",
            tool_name="wrong_calc", quality=StepQuality.WRONG,
            expected_tool="correct_calc",
        ))
        result = evaluator.evaluate(final_correct=True)
        assert result.hidden_failure_detected is True
        assert "wrong tool" in result.hidden_failure_reason.lower()

    def test_hidden_failure_redundant(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", action="reasoning", reasoning="step 1",
        ))
        evaluator.record_step(TrajectoryStep(
            step_id="s2", action="reasoning", reasoning="step 2",
            quality=StepQuality.REDUNDANT,
        ))
        evaluator.record_step(TrajectoryStep(
            step_id="s3", action="reasoning", reasoning="step 3",
            quality=StepQuality.REDUNDANT,
        ))
        result = evaluator.evaluate(final_correct=True)
        assert result.hidden_failure_detected is True

    def test_no_hidden_failure_on_wrong_answer(self, evaluator):
        # Hidden failure only checked when final_correct=True
        evaluator.record_step(TrajectoryStep(
            step_id="s1", action="tool_call",
            tool_name="wrong", quality=StepQuality.WRONG,
        ))
        result = evaluator.evaluate(final_correct=False)
        assert result.hidden_failure_detected is False

    # ── Efficiency scoring ────────────────────────────

    def test_efficiency_perfect(self, evaluator):
        for i in range(3):
            evaluator.record_step(TrajectoryStep(
                step_id=f"s{i}", quality=StepQuality.OPTIMAL,
            ))
        result = evaluator.evaluate()
        assert result.efficiency_score == 1.0

    def test_efficiency_with_redundant(self, evaluator):
        evaluator.record_step(TrajectoryStep(step_id="s1"))
        evaluator.record_step(TrajectoryStep(
            step_id="s2", quality=StepQuality.REDUNDANT,
        ))
        result = evaluator.evaluate()
        # 1 redundant / 2 total → penalty → 1.0 - 0.5 = 0.5
        assert result.efficiency_score == 0.5

    # ── Error handling scoring ────────────────────────

    def test_error_handling_perfect(self, evaluator):
        for i in range(3):
            evaluator.record_step(TrajectoryStep(
                step_id=f"s{i}", status="success",
            ))
        result = evaluator.evaluate()
        assert result.error_handling_score == 1.0

    def test_error_handling_with_recovery(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", status="failed",
        ))
        evaluator.record_step(TrajectoryStep(
            step_id="s1r", status="success", retry_count=1,
        ))
        result = evaluator.evaluate()
        # 1 error + 1 recovery → 0.5 + 0.5*(1/2) = 0.75
        assert result.error_handling_score == 0.75

    def test_error_handling_no_recovery(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", status="failed",
        ))
        evaluator.record_step(TrajectoryStep(
            step_id="s2", status="failed",
        ))
        result = evaluator.evaluate()
        # 2 errors, 0 recovered → 0.5 + 0.5*(0/2) = 0.5
        assert result.error_handling_score == 0.5

    # ── Reasoning scoring ─────────────────────────────

    def test_reasoning_score_no_reasoning(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", action="tool_call", tool_name="search",
        ))
        result = evaluator.evaluate()
        assert result.reasoning_score == 0.5

    def test_reasoning_score_single_step(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", reasoning="I think we should search",
        ))
        result = evaluator.evaluate()
        assert result.reasoning_score == 0.7

    def test_reasoning_score_coherent(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", reasoning="The search query should target recent papers",
        ))
        evaluator.record_step(TrajectoryStep(
            step_id="s2", reasoning="Based on search results, recent papers show trend X",
        ))
        result = evaluator.evaluate()
        assert result.reasoning_score >= 0.5  # coherent → higher score

    # ── Trajectory score composite ────────────────────

    def test_trajectory_score_weighted(self, evaluator):
        for i in range(5):
            evaluator.record_step(TrajectoryStep(
                step_id=f"s{i}", quality=StepQuality.OPTIMAL,
            ))
        result = evaluator.evaluate()
        # All perfect → between 0.75 and 1.0 depending on reasoning etc.
        assert 0.7 <= result.trajectory_score <= 1.0

    # ── Suggestions ───────────────────────────────────

    def test_suggestions_for_wrong_tools(self, evaluator):
        evaluator.record_step(TrajectoryStep(
            step_id="s1", action="tool_call",
            tool_name="bad", quality=StepQuality.WRONG,
        ))
        result = evaluator.evaluate(final_correct=True)
        assert len(result.suggestions) > 0

    def test_suggestions_for_low_score(self, evaluator):
        for _ in range(5):
            evaluator.record_step(TrajectoryStep(
                step_id="s", action="tool_call",
                tool_name="bad", quality=StepQuality.WRONG,
            ))
        result = evaluator.evaluate(final_correct=True)
        assert len(result.suggestions) > 0

    # ── Legacy API compat ─────────────────────────────

    def test_evaluate_from_steps_compat(self, evaluator):
        class OldStepResult:
            def __init__(self, step_id, agent_id="", tool_name="",
                         status="success", retry_count=0, duration_ms=0):
                self.step_id = step_id
                self.agent_id = agent_id
                self.tool_name = tool_name
                self.status = status
                self.retry_count = retry_count
                self.duration_ms = duration_ms

        steps = [
            OldStepResult("s1", agent_id="a1", tool_name="search"),
            OldStepResult("s2", agent_id="a1", tool_name="read"),
        ]
        result = evaluator.evaluate_from_steps(steps)
        assert result.total_steps == 2
        assert result.trace_id == "legacy"

    def test_evaluate_from_steps_with_failures(self, evaluator):
        class OldStepResult:
            def __init__(self, step_id, status="success", tool_name=""):
                self.step_id = step_id
                self.agent_id = ""
                self.tool_name = tool_name
                self.status = status
                self.retry_count = 0
                self.duration_ms = 0

        steps = [
            OldStepResult("s1", status="success"),
            OldStepResult("s2", status="failed"),
        ]
        result = evaluator.evaluate_from_steps(steps)
        assert result.wrong_steps >= 1  # failed → WRONG quality

    def test_totals_computed(self, evaluator):
        evaluator.record_step(TrajectoryStep(step_id="s1", latency_ms=100, tokens_used=50))
        evaluator.record_step(TrajectoryStep(step_id="s2", latency_ms=200, tokens_used=75))
        result = evaluator.evaluate()
        assert result.total_latency_ms == 300
        assert result.total_tokens == 125
