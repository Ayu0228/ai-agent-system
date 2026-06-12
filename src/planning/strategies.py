"""Planning Strategies — 4 种显式规划模式。

ref: Lilian Weng — "LLM Powered Autonomous Agents" planning taxonomy
ref: Anthropic — prompt chaining as the foundation of agent planning

模式:
  1. Chain-of-Thought (CoT) — 逐步推理，每步基于前一步
  2. Tree-of-Thought (ToT) — 分支探索 + BFS/DFS + 评估剪枝
  3. ReAct — Reasoning + Acting 交替，工具调用的推理框架
  4. Reflexion — 执行后反思，从错误中学习改进

每个策略输出结构化的 PlanResult，包含步骤、置信度、预估成本。
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PlanningStrategy(str, Enum):
    COT = "cot"               # Chain-of-Thought
    TOT = "tot"               # Tree-of-Thought
    REACT = "react"           # Reasoning + Acting
    REFLEXION = "reflexion"   # 执行后反思


@dataclass
class PlanStep:
    """规划步骤。"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    action: str = ""                       # think / tool_call / query / verify / reflect
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    confidence: float = 0.8                # 此步骤的置信度
    depends_on: list[str] = field(default_factory=list)
    estimated_tokens: int = 500
    alternatives: list[str] = field(default_factory=list)  # 备选方案


@dataclass
class PlanResult:
    """规划结果。"""
    strategy: PlanningStrategy
    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    total_steps: int = 0
    total_estimated_tokens: int = 0
    total_estimated_cost: float = 0.0
    confidence: float = 1.0               # 整体置信度 = ∏ step.confidence
    reasoning: str = ""                    # 规划推理过程
    alternatives_considered: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ChainOfThought:
    """Chain-of-Thought — 逐步推理链。

    每步输出作为下一步的输入。
    适合: 数学推理、逻辑推导、需要逐步构建答案的任务。
    """

    def plan(self, goal: str, max_steps: int = 7,
             context: dict[str, Any] | None = None) -> PlanResult:
        ctx = context or {}
        steps: list[PlanStep] = []

        # 1. 分析
        steps.append(PlanStep(
            description=f"分析问题: {goal[:80]}",
            action="think",
            confidence=0.95,
            estimated_tokens=200,
        ))

        # 2-N. 逐步推理
        subgoals = self._decompose(goal, max_steps - 2)
        for i, sg in enumerate(subgoals):
            steps.append(PlanStep(
                description=sg,
                action="think",
                confidence=0.85 - i * 0.05,
                depends_on=[steps[-1].id],
                estimated_tokens=300,
            ))

        # N. 验证
        if steps:
            steps.append(PlanStep(
                description="验证推理链的正确性并总结答案",
                action="verify",
                confidence=0.9,
                depends_on=[steps[-1].id],
                estimated_tokens=400,
            ))

        total_tokens = sum(s.estimated_tokens for s in steps)
        confidence = self._chain_confidence(steps)

        return PlanResult(
            strategy=PlanningStrategy.COT,
            goal=goal,
            steps=steps,
            total_steps=len(steps),
            total_estimated_tokens=total_tokens,
            total_estimated_cost=total_tokens * 0.003 / 1000,  # ~$3/M tokens
            confidence=confidence,
            reasoning=f"CoT: 将问题分解为 {len(steps)} 个推理步骤",
        )

    def _decompose(self, goal: str, n: int) -> list[str]:
        """问题分解 — 生产环境用 LLM 生成，此处为启发式。"""
        # 简单启发式分解
        parts = []
        keywords = ["分析", "收集", "计算", "比较", "评估", "总结", "验证"]
        for i in range(min(n, len(keywords))):
            parts.append(f"步骤{i + 1}: {keywords[i]} — {goal[:40]}")
        return parts

    def _chain_confidence(self, steps: list[PlanStep]) -> float:
        if not steps:
            return 0.0
        conf = 1.0
        for s in steps:
            conf *= s.confidence
        return round(conf, 3)


class TreeOfThought:
    """Tree-of-Thought — 分支探索 + BFS/DFS + 评估剪枝。

    在每一步生成多个候选，评估每个候选，保留最佳的 N 个。
    适合: 需要探索多种可能性的创造性任务（写作、策略、设计）。
    """

    def __init__(self, beam_width: int = 3, max_depth: int = 5) -> None:
        self.beam_width = beam_width
        self.max_depth = max_depth

    def plan(self, goal: str, context: dict[str, Any] | None = None) -> PlanResult:
        ctx = context or {}
        steps: list[PlanStep] = []

        # Root: 分析目标，生成多个方向
        root = PlanStep(
            description=f"探索 '{goal[:60]}' 的解决方案",
            action="think",
            confidence=0.9,
        )
        steps.append(root)

        # 分支探索
        branches = self._generate_branches(goal)
        for i, branch in enumerate(branches[:self.beam_width]):
            branch_step = PlanStep(
                description=f"路径 {i + 1}: {branch}",
                action="think",
                confidence=0.8 - i * 0.1,
                depends_on=[root.id],
                alternatives=branches[i + 1:self.beam_width + 1] if i == 0 else [],
            )
            steps.append(branch_step)

        # 评估 & 选择最佳路径
        if len(steps) > 1:
            steps.append(PlanStep(
                description=f"评估 {len(branches[:self.beam_width])} 条路径，选择最优",
                action="verify",
                confidence=0.85,
                depends_on=[s.id for s in steps[1:]],
                estimated_tokens=500,
            ))

        total_tokens = sum(s.estimated_tokens for s in steps)
        # ToT 探索多样性 → 整体置信度比 CoT 高
        confidence = min(0.9, self._chain_confidence(steps) + 0.1)

        return PlanResult(
            strategy=PlanningStrategy.TOT,
            goal=goal,
            steps=steps,
            total_steps=len(steps),
            total_estimated_tokens=total_tokens,
            total_estimated_cost=total_tokens * 0.005 / 1000,
            confidence=confidence,
            reasoning=f"ToT(beam={self.beam_width}, depth={self.max_depth}): 探索 {len(branches)} 条路径",
            alternatives_considered=len(branches),
        )

    def _generate_branches(self, goal: str) -> list[str]:
        return [
            f"方案A: 直接搜索相关信息",
            f"方案B: 分析现有数据后推理",
            f"方案C: 对比多种来源后综合",
            f"方案D: 从第一原理出发推导",
        ]

    def _chain_confidence(self, steps: list[PlanStep]) -> float:
        if not steps:
            return 0.0
        conf = 1.0
        for s in steps:
            conf *= s.confidence
        return round(conf, 3)


class ReActPlanner:
    """ReAct — Reasoning + Acting 交替模式。

    交替执行: Thought → Action → Observation → Thought → ...
    适合: 需要工具调用和信息检索的任务。
    """

    def __init__(self, max_rounds: int = 10) -> None:
        self.max_rounds = max_rounds

    def plan(self, goal: str, available_tools: list[str] | None = None,
             context: dict[str, Any] | None = None,
             budget_manager: Any = None) -> PlanResult:
        tools = available_tools or ["web_search", "calculator", "database_query"]
        steps: list[PlanStep] = []

        react_rounds = self._compute_effective_react_rounds(budget_manager)

        current_id: str | None = None
        for i in range(react_rounds):
            # Thought
            thought = PlanStep(
                description=f"思考: 分析当前状态并决定下一步" if i == 0
                else f"思考: 基于观察结果更新理解",
                action="think",
                confidence=0.9 - i * 0.05,
                depends_on=[current_id] if current_id else [],
            )
            steps.append(thought)
            current_id = thought.id

            # Action
            tool = tools[i % len(tools)]
            action = PlanStep(
                description=f"执行: 调用 {tool} 获取信息",
                action="tool_call",
                tool_name=tool,
                confidence=0.85 - i * 0.05,
                depends_on=[current_id],
                estimated_tokens=300,
            )
            steps.append(action)
            current_id = action.id

        # Final thought
        if steps:
            steps.append(PlanStep(
                description="综合所有观察结果，给出最终答案",
                action="think",
                confidence=0.9,
                depends_on=[steps[-1].id],
                estimated_tokens=400,
            ))

        total_tokens = sum(s.estimated_tokens for s in steps)
        confidence = self._chain_confidence(steps)

        if budget_manager is not None:
            budget_manager.record(total_tokens, workflow_id="reflection")

        return PlanResult(
            strategy=PlanningStrategy.REACT,
            goal=goal,
            steps=steps,
            total_steps=len(steps),
            total_estimated_tokens=total_tokens,
            total_estimated_cost=total_tokens * 0.004 / 1000,
            confidence=confidence,
            reasoning=f"ReAct({len(tools)} tools, {react_rounds} rounds): "
                      f"交替 Thinking 和 Acting",
        )

    def _compute_effective_react_rounds(self, budget_manager: Any = None) -> int:
        """计算有效的 ReAct 轮次 — 结合预算约束。"""
        # base: 保留原有逻辑（max_rounds // 2，但不超过 4）
        base = min(self.max_rounds // 2, 4)

        if budget_manager is not None:
            try:
                status = budget_manager.check(workflow_id="reflection")
                if status.is_exceeded and status.config.hard_limit:
                    return 1
                if not budget_manager.can_proceed(1500, workflow_id="reflection"):
                    return max(1, base // 2)
            except Exception:
                pass

        return max(1, base)

    def _chain_confidence(self, steps: list[PlanStep]) -> float:
        if not steps:
            return 0.0
        conf = 1.0
        for s in steps:
            conf *= s.confidence
        return round(conf, 3)


class ReflexionPlanner:
    """Reflexion — 执行后反思，从错误中学习。

    流程: Try → Evaluate → Reflect → Retry (with improved plan)
    适合: 错误恢复、自我改进的 Agent 系统。
    """

    def __init__(self, max_reflections: int = 3, max_reflection_rounds: int = 5) -> None:
        self.reflection_history_window = max_reflections  # 历史错误窗口（兼容旧接口）
        self.max_reflection_rounds = max_reflection_rounds  # 迭代轮次上限
        self._reflection_history: list[dict[str, Any]] = []
        self._last_round_confidence: float = 0.0

    def plan(self, goal: str, previous_errors: list[str] | None = None,
             context: dict[str, Any] | None = None,
             budget_manager: Any = None,
             current_round: int = 1) -> PlanResult:
        errors = previous_errors or []

        # 预算感知: 计算有效最大轮次
        effective_max = self._compute_effective_max_rounds(goal, budget_manager)

        # 衰减退出: 检查是否应该继续反射
        if current_round > 1 and not self.should_continue_reflection(
            current_round, effective_max, budget_manager,
        ):
            return PlanResult(
                strategy=PlanningStrategy.REFLEXION,
                goal=goal,
                reasoning=f"Reflexion early exit at round {current_round}/{effective_max}",
                metadata={"early_exit": True, "round": current_round},
            )

        steps: list[PlanStep] = []

        # 1. 初始方案
        steps.append(PlanStep(
            description=f"制定执行方案: {goal[:80]}",
            action="think",
            confidence=0.9 if not errors else 0.7,
            estimated_tokens=400,
        ))

        # 2. 如果有历史错误，添加反思步骤
        if errors:
            for i, err in enumerate(errors[-self.reflection_history_window:]):
                steps.append(PlanStep(
                    description=f"反思错误 {i + 1}: {err[:80]}",
                    action="reflect",
                    confidence=0.75,
                    depends_on=[steps[0].id],
                    estimated_tokens=300,
                    alternatives=["跳过此错误继续", "调整方案避免此错误"],
                ))

            # 调整方案
            steps.append(PlanStep(
                description="基于反思结果调整执行方案",
                action="think",
                confidence=0.8,
                depends_on=[s.id for s in steps[1:]],
                estimated_tokens=400,
            ))

        # 3. 执行
        steps.append(PlanStep(
            description="执行调整后的方案",
            action="tool_call",
            confidence=0.85 if not errors else 0.75,
            depends_on=[steps[-1].id],
            estimated_tokens=500,
        ))

        # 4. 自我评估
        steps.append(PlanStep(
            description="自我评估: 检查结果是否符合预期，是否需要进一步调整",
            action="verify",
            confidence=0.8,
            depends_on=[steps[-1].id],
            estimated_tokens=300,
        ))

        total_tokens = sum(s.estimated_tokens for s in steps)
        confidence = self._chain_confidence(steps)
        self._last_round_confidence = confidence

        # 预算记录
        if budget_manager is not None:
            budget_manager.record(total_tokens, workflow_id="reflection")

        plan_result = PlanResult(
            strategy=PlanningStrategy.REFLEXION,
            goal=goal,
            steps=steps,
            total_steps=len(steps),
            total_estimated_tokens=total_tokens,
            total_estimated_cost=total_tokens * 0.005 / 1000,
            confidence=confidence,
            reasoning=f"Reflexion(round={current_round}/{effective_max}, reflections={len(errors)}): "
                      f"从 {len(errors)} 个历史错误中学习并改进方案",
            metadata={"reflection_round": current_round, "effective_max": effective_max},
        )

        self._reflection_history.append({
            "goal": goal, "errors": errors, "confidence": confidence,
            "timestamp": plan_result.created_at, "round": current_round,
        })
        return plan_result

    def _compute_effective_max_rounds(self, goal: str,
                                       budget_manager: Any = None) -> int:
        """结合预算动态计算有效最大轮次。"""
        effective = self.max_reflection_rounds

        if budget_manager is not None:
            try:
                status = budget_manager.check(workflow_id="reflection")
                if status.is_exceeded:
                    if status.config.hard_limit:
                        return 0  # 硬阻断
                    # 软限制: 减半
                    effective = max(1, effective // 2)
                    logger.info("reflection_budget_tight", effective=effective)
            except Exception:
                pass

            # 检查是否还有足够预算执行额外轮次
            if not budget_manager.can_proceed(2000, workflow_id="reflection"):
                effective = min(effective, 2)

        return effective

    def should_continue_reflection(self, current_round: int,
                                    effective_max: int = 5,
                                    budget_manager: Any = None) -> bool:
        """判断是否应继续反射轮次。

        退出条件:
          1. 超过有效最大轮次
          2. 改进幅度 < 0.05 (衰减退出)
          3. 预算不足
        """
        if current_round > effective_max:
            return False

        # 改进衰减检测
        if len(self._reflection_history) >= 2:
            recent = self._reflection_history[-2:]
            if len(recent) == 2:
                prev_conf = recent[0].get("confidence", 0.0)
                curr_conf = recent[1].get("confidence", 0.0)
                improvement = curr_conf - prev_conf
                if improvement < 0.05 and current_round > 1:
                    logger.info("reflection_diminishing_returns",
                               improvement=round(improvement, 3),
                               round=current_round)
                    return False

        # 预算检查
        if budget_manager is not None:
            if not budget_manager.can_proceed(2000, workflow_id="reflection"):
                logger.info("reflection_budget_exhausted", round=current_round)
                return False

        return True

    def _chain_confidence(self, steps: list[PlanStep]) -> float:
        if not steps:
            return 0.0
        conf = 1.0
        for s in steps:
            conf *= s.confidence
        return round(conf, 3)

    def get_history(self) -> list[dict[str, Any]]:
        return self._reflection_history


class Planner:
    """统一规划器 — 根据任务特征自动选择最佳策略。

    用法:
        planner = Planner()
        result = planner.plan(goal="调研 2026 年 AI Agent 趋势",
                              strategy=PlanningStrategy.COT)
        # 或自动选择
        result = planner.auto_plan(goal="...", complexity="high")
    """

    def __init__(self) -> None:
        self.cot = ChainOfThought()
        self.tot = TreeOfThought(beam_width=3, max_depth=5)
        self.react = ReActPlanner(max_rounds=10)
        self.reflexion = ReflexionPlanner(max_reflections=3, max_reflection_rounds=5)

    def plan(self, goal: str, strategy: PlanningStrategy = PlanningStrategy.COT,
             **kwargs: Any) -> PlanResult:
        # CoT and ToT don't accept budget_manager — filter it
        strat_kwargs = dict(kwargs)
        budget_manager = strat_kwargs.pop("budget_manager", None)

        if strategy == PlanningStrategy.COT:
            return self.cot.plan(goal, **strat_kwargs)
        elif strategy == PlanningStrategy.TOT:
            return self.tot.plan(goal, **strat_kwargs)
        elif strategy == PlanningStrategy.REACT:
            return self.react.plan(goal, budget_manager=budget_manager, **strat_kwargs)
        elif strategy == PlanningStrategy.REFLEXION:
            return self.reflexion.plan(goal, budget_manager=budget_manager, **strat_kwargs)
        else:
            return PlanResult(strategy=strategy, goal=goal)

    def auto_plan(self, goal: str, complexity: str = "medium",
                  requires_tools: bool = False,
                  has_errors: list[str] | None = None,
                  budget_manager: Any = None,
                  **kwargs: Any) -> PlanResult:
        """根据任务特征自动选择策略。

        启发式:
          - 有历史错误 + 预算充足 → Reflexion
          - 需要工具 → ReAct
          - 高复杂度 / 创造性 → ToT
          - 默认 → CoT
        """
        if has_errors:
            # 预算紧张时避免 Reflexion（额外 LLM 调用成本高）
            if budget_manager is not None:
                try:
                    status = budget_manager.check(workflow_id="reflection")
                    if status.is_exceeded or not budget_manager.can_proceed(3000, workflow_id="reflection"):
                        logger.info("auto_plan_budget_skip_reflexion", goal=goal[:60])
                        strategy = PlanningStrategy.REACT
                    else:
                        strategy = PlanningStrategy.REFLEXION
                except Exception:
                    strategy = PlanningStrategy.REFLEXION
            else:
                strategy = PlanningStrategy.REFLEXION
        elif requires_tools:
            strategy = PlanningStrategy.REACT
        elif complexity == "high":
            strategy = PlanningStrategy.TOT
        else:
            strategy = PlanningStrategy.COT

        logger.info("auto_plan", goal=goal[:60], strategy=strategy.value,
                    complexity=complexity)
        return self.plan(goal, strategy=strategy, budget_manager=budget_manager, **kwargs)
