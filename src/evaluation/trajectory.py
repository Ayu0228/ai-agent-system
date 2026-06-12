"""Trajectory Evaluator — 多步 Agent 完整执行路径评估。

ref: LangChain Trajectory Evaluation framework
    "Evaluating only final outputs misses critical failures in AI agents"
    "Correct final answers can hide broken reasoning" (hidden failure detection)

评估维度:
  1. 工具选择合理性
  2. 推理路径正确性
  3. 步骤效率（冗余检测）
  4. 隐藏失败检测（正确答案+错误推理）
  5. 轨迹完整性
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class StepQuality(str, Enum):
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    SUBOPTIMAL = "suboptimal"
    WRONG = "wrong"
    REDUNDANT = "redundant"


@dataclass
class TrajectoryStep:
    """轨迹中的单步。"""
    step_id: str
    agent_id: str = ""
    action: str = ""                      # tool_call / reasoning / query / handoff
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    reasoning: str = ""
    expected_action: str = ""             # 黄金标准
    expected_tool: str = ""
    quality: StepQuality = StepQuality.ACCEPTABLE
    latency_ms: float = 0.0
    tokens_used: int = 0
    is_correct: bool = True
    status: str = "success"               # 兼容旧接口
    retry_count: int = 0


@dataclass
class TrajectoryResult:
    """轨迹评估结果。"""
    trace_id: str = ""
    agent_id: str = ""
    final_output: str = ""
    final_correct: bool = True

    # 综合评分
    trajectory_score: float = 1.0
    tool_selection_score: float = 1.0
    reasoning_score: float = 1.0
    efficiency_score: float = 1.0
    error_handling_score: float = 1.0

    # 隐藏失败
    hidden_failure_detected: bool = False
    hidden_failure_reason: str = ""

    # 详情
    steps: list[TrajectoryStep] = field(default_factory=list)
    total_steps: int = 0
    optimal_steps: int = 0
    wrong_steps: int = 0
    redundant_steps: int = 0
    reflection_rounds: int = 0
    reflection_effective_max: int = 5
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    # 兼容旧接口
    @property
    def overall(self) -> float:
        return self.trajectory_score

    @property
    def hidden_failure(self) -> bool:
        return self.hidden_failure_detected


class TrajectoryEvaluator:
    """轨迹评估器。

    用法:
        evaluator = TrajectoryEvaluator()
        evaluator.start_trace("trace-1")
        evaluator.record_step(TrajectoryStep(
            step_id="s1", agent_id="researcher",
            action="tool_call", tool_name="web_search",
            expected_tool="web_search",
        ))
        result = evaluator.evaluate("trace-1", final_correct=True)

        # 兼容旧接口
        result = evaluator.evaluate_from_steps(step_results_list)
    """

    def __init__(self) -> None:
        self._traces: dict[str, list[TrajectoryStep]] = {}

    # ── 记录 ───────────────────────────────────────

    def start_trace(self, trace_id: str) -> None:
        if trace_id not in self._traces:
            self._traces[trace_id] = []

    def record_step(self, step: TrajectoryStep) -> None:
        trace_id = "default"
        self.start_trace(trace_id)
        self._traces[trace_id].append(step)

    def get_trace(self, trace_id: str = "default") -> list[TrajectoryStep]:
        return self._traces.get(trace_id, [])

    # ── 评估 (新接口) ───────────────────────────────

    def evaluate(self, trace_id: str = "default",
                 final_output: str = "",
                 final_correct: bool = True,
                 expected_steps: list[dict[str, Any]] | None = None) -> TrajectoryResult:
        steps = self._traces.get(trace_id, [])
        return self._do_evaluate(steps, trace_id, final_output, final_correct)

    # ── 评估 (兼容旧接口) ───────────────────────────

    def evaluate_from_steps(self, steps: list[Any],
                            trace_id: str = "legacy") -> TrajectoryResult:
        """兼容旧 StepResult 接口。"""
        converted = []
        for s in steps:
            ts = TrajectoryStep(
                step_id=getattr(s, "step_id", "?"),
                agent_id=getattr(s, "agent_id", ""),
                action="tool_call" if getattr(s, "tool_name", "") else "reasoning",
                tool_name=getattr(s, "tool_name", ""),
                status=getattr(s, "status", "success"),
                retry_count=getattr(s, "retry_count", 0),
                latency_ms=getattr(s, "duration_ms", 0.0),
            )
            if ts.status == "failed":
                ts.quality = StepQuality.WRONG
            converted.append(ts)
        return self._do_evaluate(converted, trace_id, "", True)

    def _do_evaluate(self, steps: list[TrajectoryStep], trace_id: str,
                     final_output: str, final_correct: bool) -> TrajectoryResult:
        if not steps:
            return TrajectoryResult(trace_id=trace_id)

        result = TrajectoryResult(
            trace_id=trace_id,
            agent_id=steps[0].agent_id,
            final_output=final_output,
            final_correct=final_correct,
            steps=steps,
            total_steps=len(steps),
        )

        # 1. 工具选择评估
        result.tool_selection_score = self._eval_tool_selection(steps, result)
        # 2. 推理路径评估
        result.reasoning_score = self._eval_reasoning(steps, result)
        # 3. 效率评估
        result.efficiency_score = self._eval_efficiency(steps, result)
        # 3.5 反射效率评估
        result.efficiency_score = min(result.efficiency_score,
                                       self._eval_reflection_efficiency(steps, result))
        # 4. 错误处理评估
        result.error_handling_score = self._eval_error_handling(steps, result)
        # 5. 隐藏失败检测
        self._detect_hidden_failure(steps, final_correct, result)

        result.trajectory_score = (
            result.tool_selection_score * 0.25 +
            result.reasoning_score * 0.30 +
            result.efficiency_score * 0.20 +
            result.error_handling_score * 0.25
        )
        result.total_latency_ms = sum(s.latency_ms for s in steps)
        result.total_tokens = sum(s.tokens_used for s in steps)
        result.suggestions = self._generate_suggestions(result)

        logger.info("trajectory_evaluated", trace_id=trace_id,
                    score=f"{result.trajectory_score:.2f}",
                    hidden_failure=result.hidden_failure_detected)
        return result

    # ── 各维度评估 ──────────────────────────────────

    def _eval_tool_selection(self, steps: list[TrajectoryStep],
                             result: TrajectoryResult) -> float:
        tool_steps = [s for s in steps if s.action == "tool_call"]
        # Count all wrong-quality steps (not just tool calls)
        all_wrong = sum(1 for s in steps if s.quality == StepQuality.WRONG)
        result.wrong_steps = all_wrong

        if not tool_steps:
            return 1.0

        optimal = sum(1 for s in tool_steps if s.quality == StepQuality.OPTIMAL)
        wrong = sum(1 for s in tool_steps if s.quality == StepQuality.WRONG)
        result.optimal_steps = optimal

        for s in tool_steps:
            if s.quality == StepQuality.WRONG:
                result.issues.append(f"Wrong tool '{s.tool_name}' at {s.step_id}")
                if s.expected_tool:
                    result.issues.append(f"  Expected: {s.expected_tool}")

        return max(0.0, 1.0 - wrong / len(tool_steps) * 0.5)

    def _eval_reasoning(self, steps: list[TrajectoryStep],
                        result: TrajectoryResult) -> float:
        reasoning_steps = [s for s in steps if s.reasoning]
        if not reasoning_steps:
            return 0.5

        coherent = 0
        for i in range(1, len(reasoning_steps)):
            prev_words = set(reasoning_steps[i - 1].reasoning.lower().split())
            curr_words = set(reasoning_steps[i].reasoning.lower().split())
            if len(prev_words & curr_words) >= 2:
                coherent += 1

        if len(reasoning_steps) <= 1:
            return 0.7
        return 0.5 + 0.5 * (coherent / (len(reasoning_steps) - 1))

    def _eval_efficiency(self, steps: list[TrajectoryStep],
                         result: TrajectoryResult) -> float:
        total = len(steps)
        redundant = sum(1 for s in steps if s.quality == StepQuality.REDUNDANT)
        result.redundant_steps = redundant
        if redundant > 0:
            result.issues.append(f"{redundant} redundant step(s) detected")

        if total == 0:
            return 1.0
        return max(0.2, 1.0 - redundant / total)

    def _eval_error_handling(self, steps: list[TrajectoryStep],
                             result: TrajectoryResult) -> float:
        failed = [s for s in steps if s.status == "failed"]
        recovered = [s for s in steps if s.retry_count > 0 and s.status == "success"]
        total_err = len(failed) + len(recovered)
        if total_err == 0:
            return 1.0
        return 0.5 + 0.5 * (len(recovered) / total_err)

    def _eval_reflection_efficiency(self, steps: list[TrajectoryStep],
                                      result: TrajectoryResult) -> float:
        """反射效率评估 — 过多反射轮次或衰减不力扣分。"""
        if result.reflection_rounds <= 0:
            return 1.0

        effective_max = result.reflection_effective_max
        if result.reflection_rounds > effective_max:
            excess = result.reflection_rounds - effective_max
            result.issues.append(
                f"Reflection rounds ({result.reflection_rounds}) exceed effective max ({effective_max})"
            )
            return max(0.3, 1.0 - excess * 0.2)

        if result.reflection_rounds >= effective_max * 0.8:
            return 0.75

        return 1.0

    def _detect_hidden_failure(self, steps: list[TrajectoryStep],
                               final_correct: bool,
                               result: TrajectoryResult) -> None:
        if not final_correct:
            return

        wrong_tools = [s for s in steps if s.quality == StepQuality.WRONG]
        if wrong_tools:
            result.hidden_failure_detected = True
            result.hidden_failure_reason = (
                f"Correct answer but {len(wrong_tools)} wrong tool(s): "
                + ", ".join(s.tool_name for s in wrong_tools)
            )

        redundant = [s for s in steps if s.quality == StepQuality.REDUNDANT]
        if redundant and not result.hidden_failure_detected:
            result.hidden_failure_detected = True
            result.hidden_failure_reason = (
                f"Correct answer but {len(redundant)} redundant step(s)"
            )

        # 正确答案但反射轮次过多（> 2x effective_max）
        if final_correct and result.reflection_rounds > 2 * result.reflection_effective_max:
            if not result.hidden_failure_detected:
                result.hidden_failure_detected = True
                result.hidden_failure_reason = (
                    f"Correct answer but {result.reflection_rounds} reflection rounds "
                    f"(effective max: {result.reflection_effective_max})"
                )

    def _generate_suggestions(self, result: TrajectoryResult) -> list[str]:
        suggestions: list[str] = []
        if result.wrong_steps > 0:
            suggestions.append(f"Fix {result.wrong_steps} wrong tool selections — review ACI docs")
        if result.redundant_steps > 1:
            suggestions.append(f"Optimize {result.redundant_steps} redundant steps")
        if result.hidden_failure_detected:
            suggestions.append(f"Hidden failure: {result.hidden_failure_reason}")
        if result.tool_selection_score < 0.6:
            suggestions.append("Tool selection score low — improve tool descriptions")
        if result.efficiency_score < 0.6:
            suggestions.append("Efficiency low — consider step consolidation or early-exit")
        return suggestions
