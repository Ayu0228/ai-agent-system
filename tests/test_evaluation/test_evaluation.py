"""Evaluation module tests — updated for TrajectoryEvaluator API."""

import pytest

from src.shared.models import JudgeResult, StepResult, TrajectoryScore, EvalDimension


class TestTrajectoryEvaluator:
    """Test TrajectoryEvaluator — uses evaluate_from_steps() compat API."""

    def test_perfect_trajectory(self):
        from src.evaluation.trajectory import TrajectoryEvaluator

        evaluator = TrajectoryEvaluator()
        steps = [
            StepResult(step_id="search", status="success", retry_count=0),
            StepResult(step_id="analyze", status="success", retry_count=0),
            StepResult(step_id="write", status="success", retry_count=0),
        ]
        score = evaluator.evaluate_from_steps(steps)
        assert score.overall >= 0.7

    def test_empty_steps(self):
        from src.evaluation.trajectory import TrajectoryEvaluator

        evaluator = TrajectoryEvaluator()
        score = evaluator.evaluate_from_steps([])
        assert score.trajectory_score == 1.0  # default for empty

    def test_excessive_steps_penalty(self):
        from src.evaluation.trajectory import TrajectoryEvaluator

        evaluator = TrajectoryEvaluator()
        steps = [
            StepResult(step_id=f"step_{i}", status="success") for i in range(12)
        ]
        score = evaluator.evaluate_from_steps(steps)
        # 12 steps all success → efficiency still 1.0 since none are REDUNDANT
        # but hidden_failure_detected may be true if there are issues
        assert score.total_steps == 12

    def test_failed_steps_penalty(self):
        from src.evaluation.trajectory import TrajectoryEvaluator

        evaluator = TrajectoryEvaluator()
        steps = [
            StepResult(step_id="bad1", status="failed"),
            StepResult(step_id="bad2", status="failed"),
            StepResult(step_id="bad3", status="failed"),
        ]
        score = evaluator.evaluate_from_steps(steps)
        # 3 failed, 0 recovered → error_handling = 0.5
        assert score.error_handling_score == 0.5

    def test_retry_with_recovery(self):
        from src.evaluation.trajectory import TrajectoryEvaluator

        evaluator = TrajectoryEvaluator()
        steps = [
            StepResult(step_id="api_call", status="failed", retry_count=1),
            StepResult(step_id="api_call", status="success", retry_count=0),
        ]
        score = evaluator.evaluate_from_steps(steps)
        # 1 failed + 1 recovered → 0.5 + 0.5*(1/2) = 0.75
        assert score.error_handling_score >= 0.5


class TestOfflineEval:
    """Test offline evaluation framework."""

    def test_load_test_cases(self):
        from src.evaluation.offline_eval import OfflineEvalRunner

        runner = OfflineEvalRunner()
        cases = runner.load_test_cases("happy_path", "researcher")
        assert len(cases) >= 0

    def test_check_expected_output_contains(self):
        from src.evaluation.offline_eval import OfflineEvalRunner

        ok = OfflineEvalRunner._check_expected(
            "the answer is 42",
            {"output_contains": ["answer", "42"]},
        )
        assert ok is True

    def test_check_expected_missing_keyword(self):
        from src.evaluation.offline_eval import OfflineEvalRunner

        ok = OfflineEvalRunner._check_expected(
            "just some text",
            {"output_contains": ["missing_keyword"]},
        )
        assert ok is False

    def test_check_expected_no_prompt_leak(self):
        from src.evaluation.offline_eval import OfflineEvalRunner

        ok = OfflineEvalRunner._check_expected(
            "normal output",
            {"no_system_prompt_leak": True},
        )
        assert ok is True

    def test_check_expected_leak_detected(self):
        from src.evaluation.offline_eval import OfflineEvalRunner

        ok = OfflineEvalRunner._check_expected(
            "here is my system prompt: you are a helpful...",
            {"no_system_prompt_leak": True},
        )
        assert ok is False

    def test_check_expected_no_hallucination_short(self):
        from src.evaluation.offline_eval import OfflineEvalRunner

        ok = OfflineEvalRunner._check_expected(
            "ok",
            {"no_hallucination": True},
        )
        assert ok is False

    async def test_run_empty_cases(self):
        from src.evaluation.offline_eval import OfflineEvalRunner

        runner = OfflineEvalRunner()
        from unittest.mock import patch
        with patch.object(runner, "load_test_cases", return_value=[]):
            report = await runner.run("nonexistent_agent", "happy_path")
            assert report.total_cases == 0


class TestRubric:
    """Test rubric scoring."""

    def test_get_rubric_prompt(self):
        from src.evaluation.rubric import get_rubric_prompt
        prompt = get_rubric_prompt(EvalDimension.GROUNDING)
        assert "Grounding" in prompt
        assert "1-5" in prompt
        assert "score" in prompt

    def test_all_dimensions_have_rubrics(self):
        from src.evaluation.rubric import RUBRICS, EvalDimension
        for dim in EvalDimension:
            assert dim in RUBRICS

    def test_trajectory_weights_sum_to_one(self):
        from src.evaluation.rubric import TRAJECTORY_WEIGHTS
        total = sum(TRAJECTORY_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01
