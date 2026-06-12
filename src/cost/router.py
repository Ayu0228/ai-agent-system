"""Model Router — ParetoBandit 自适应模型路由。

ref: ParetoBandit (arXiv 2604.00136) — budget-paced adaptive routing using contextual bandits
ref: OrcaRouter (MIT) — intelligent model selection with cost/quality tradeoffs
ref: Claude-Code-LLM-Router (Apache 2.0) — multi-model dispatch with fallback chains

核心思想:
  - 每个 model tier 有 cost 和 capability score
  - 基于剩余预算选择适当的 tier（预算充足=能力最强, 预算紧张=性价比最优）
  - ParetoBandit: 预算消耗 vs 任务完成质量的帕累托最优边界
  - 支持 fallback 链: 主模型失败 → 降级到 backup
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ModelTier(str, Enum):
    """模型层级 — 按成本和能力排序。"""
    PREMIUM = "premium"         # GPT-4o, Claude Opus — 最高能力, 最贵
    STANDARD = "standard"       # GPT-4o-mini, Claude Sonnet — 平衡
    BUDGET = "budget"           # DeepSeek, Haiku — 性价比
    FALLBACK = "fallback"       # 本地小模型 — 仅兜底


@dataclass
class ModelConfig:
    """模型配置。"""
    name: str
    tier: ModelTier
    provider: str                               # openai / anthropic / deepseek / local
    input_price_per_1m: float = 0.0             # USD per 1M input tokens
    output_price_per_1m: float = 0.0
    capability_score: float = 0.5               # 0-1 能力评分
    latency_ms: float = 500.0                   # 平均延迟
    max_tokens: int = 128_000
    supports_tools: bool = True
    supports_vision: bool = False
    enabled: bool = True

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000 * self.input_price_per_1m
                + output_tokens / 1_000_000 * self.output_price_per_1m)


# 默认模型配置（2024-2025 年定价）
DEFAULT_MODELS: list[ModelConfig] = [
    ModelConfig(name="gpt-4o", tier=ModelTier.PREMIUM, provider="openai",
                input_price_per_1m=2.50, output_price_per_1m=10.00,
                capability_score=0.95, supports_vision=True),
    ModelConfig(name="claude-opus-4", tier=ModelTier.PREMIUM, provider="anthropic",
                input_price_per_1m=15.00, output_price_per_1m=75.00,
                capability_score=0.97, supports_vision=True),
    ModelConfig(name="gpt-4o-mini", tier=ModelTier.STANDARD, provider="openai",
                input_price_per_1m=0.15, output_price_per_1m=0.60,
                capability_score=0.75),
    ModelConfig(name="claude-sonnet-4", tier=ModelTier.STANDARD, provider="anthropic",
                input_price_per_1m=3.00, output_price_per_1m=15.00,
                capability_score=0.85, supports_vision=True),
    ModelConfig(name="deepseek-v3", tier=ModelTier.BUDGET, provider="deepseek",
                input_price_per_1m=0.27, output_price_per_1m=1.10,
                capability_score=0.80),
    ModelConfig(name="claude-haiku-4", tier=ModelTier.BUDGET, provider="anthropic",
                input_price_per_1m=0.80, output_price_per_1m=4.00,
                capability_score=0.65),
    ModelConfig(name="deepseek-r1", tier=ModelTier.BUDGET, provider="deepseek",
                input_price_per_1m=0.55, output_price_per_1m=2.19,
                capability_score=0.82),
    ModelConfig(name="llama-local", tier=ModelTier.FALLBACK, provider="local",
                input_price_per_1m=0.0, output_price_per_1m=0.0,
                capability_score=0.30, supports_tools=False,
                latency_ms=200.0, max_tokens=8_000),
]


@dataclass
class RoutingDecision:
    """路由决策。"""
    model: ModelConfig
    reason: str = ""
    budget_remaining_pct: float = 1.0
    fallback_chain: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ModelRouter:
    """ParetoBandit 自适应模型路由器。

    用法:
        router = ModelRouter()
        router.add_model(ModelConfig(name="gpt-4o-mini", ...))

        decision = router.route(
            task_complexity="high",
            budget_remaining_pct=0.8,
            required_capabilities=["tools"],
        )
        # → RoutingDecision(model=gpt-4o, reason="high complexity + budget充足")
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}
        # ParetoBandit 状态: model_name → {successes, failures, total_cost, avg_quality}
        self._bandit_state: dict[str, dict[str, float]] = {}

    # ── 配置 ───────────────────────────────────────

    def add_model(self, model: ModelConfig) -> None:
        self._models[model.name] = model
        if model.name not in self._bandit_state:
            self._bandit_state[model.name] = {
                "successes": 0, "failures": 0, "total_cost": 0.0,
                "avg_latency": model.latency_ms,
            }

    def load_defaults(self) -> None:
        for m in DEFAULT_MODELS:
            self.add_model(m)

    # ── 路由 ───────────────────────────────────────

    def route(self, task_complexity: str = "medium",
              budget_remaining_pct: float = 1.0,
              estimated_input_tokens: int = 2000,
              estimated_output_tokens: int = 500,
              required_capabilities: list[str] | None = None,
              preferred_provider: str = "") -> RoutingDecision:
        """根据任务特征和预算状况选择最佳模型。

        ParetoBandit 决策逻辑:
          1. 预算充足 (>70%): 选能力最强的 tier
          2. 预算中等 (30-70%): 选性价比最优 tier（Pareto front）
          3. 预算紧张 (<30%): 选最便宜 tier
          4. 预算极度紧张 (<10%): fallback 本地模型

        Args:
            task_complexity: "low" / "medium" / "high"
            budget_remaining_pct: 剩余预算百分比 0-1
            estimated_input_tokens: 预估输入 token 数
            estimated_output_tokens: 预估输出 token 数
            required_capabilities: ["tools", "vision"] 等
            preferred_provider: 偏好供应商
        """
        caps = required_capabilities or []
        enabled = [m for m in self._models.values() if m.enabled]

        if not enabled:
            return RoutingDecision(
                model=ModelConfig(name="none", tier=ModelTier.FALLBACK, provider=""),
                reason="no models available",
                budget_remaining_pct=budget_remaining_pct,
            )

        # 过滤能力要求
        candidates = enabled
        if "vision" in caps:
            candidates = [m for m in candidates if m.supports_vision]
        if "tools" in caps:
            candidates = [m for m in candidates if m.supports_tools]

        if not candidates:
            candidates = enabled  # fallback to all

        # ParetoBandit: 根据预算选择 tier
        if budget_remaining_pct > 0.7:
            target_tier = ModelTier.PREMIUM
            reason = "budget充足, 选最高能力 tier"
        elif budget_remaining_pct > 0.3:
            target_tier = ModelTier.STANDARD
            reason = "budget中等, 选性价比最优 (Pareto front)"
        elif budget_remaining_pct > 0.1:
            target_tier = ModelTier.BUDGET
            reason = "budget紧张, 选最便宜 tier"
        else:
            target_tier = ModelTier.FALLBACK
            reason = "budget极度紧张, 降级到 fallback"

        # 按任务复杂度调整
        if task_complexity == "high" and budget_remaining_pct > 0.3:
            # 高复杂度任务提升一级
            upgrade = {
                ModelTier.FALLBACK: ModelTier.BUDGET,
                ModelTier.BUDGET: ModelTier.STANDARD,
                ModelTier.STANDARD: ModelTier.PREMIUM,
                ModelTier.PREMIUM: ModelTier.PREMIUM,
            }
            target_tier = upgrade.get(target_tier, target_tier)
            reason += " + 高复杂度任务提升 tier"
        elif task_complexity == "low" and budget_remaining_pct < 0.9:
            # 低复杂度降低一级
            downgrade = {
                ModelTier.PREMIUM: ModelTier.STANDARD,
                ModelTier.STANDARD: ModelTier.BUDGET,
                ModelTier.BUDGET: ModelTier.FALLBACK,
                ModelTier.FALLBACK: ModelTier.FALLBACK,
            }
            target_tier = downgrade.get(target_tier, target_tier)
            reason += " + 低复杂度任务降低 tier"

        # 在同 tier 内选最佳（考虑 provider 偏好和 bandit 表现）
        tier_candidates = [m for m in candidates if m.tier == target_tier]
        if not tier_candidates:
            # 降级到下一个可用 tier
            for fallback_tier in [ModelTier.STANDARD, ModelTier.BUDGET, ModelTier.FALLBACK]:
                tier_candidates = [m for m in candidates if m.tier == fallback_tier]
                if tier_candidates:
                    reason += f" (降级: {target_tier.value} → {fallback_tier.value})"
                    target_tier = fallback_tier
                    break

        if not tier_candidates:
            tier_candidates = candidates

        # 在候选者中选择: 优先 provider 偏好 → bandit 成功率 → 成本
        best = self._pick_best(tier_candidates, preferred_provider,
                               estimated_input_tokens, estimated_output_tokens)

        # 构建 fallback 链
        fallback_chain = self._build_fallback_chain(best, candidates)

        est_cost = best.cost_for(estimated_input_tokens, estimated_output_tokens)

        return RoutingDecision(
            model=best,
            reason=reason,
            budget_remaining_pct=budget_remaining_pct,
            fallback_chain=[m.name for m in fallback_chain],
            estimated_cost=est_cost,
        )

    def record_outcome(self, model_name: str, success: bool,
                       cost: float = 0.0, latency_ms: float = 0.0,
                       quality_score: float = 0.0) -> None:
        """记录模型调用结果 — 更新 ParetoBandit 状态。"""
        state = self._bandit_state.get(model_name)
        if not state:
            return
        if success:
            state["successes"] += 1
        else:
            state["failures"] += 1
        state["total_cost"] += cost
        if latency_ms > 0:
            # EMA 平滑延迟
            state["avg_latency"] = 0.7 * state["avg_latency"] + 0.3 * latency_ms

    # ── 内部 ───────────────────────────────────────

    def _pick_best(self, candidates: list[ModelConfig],
                   preferred_provider: str,
                   input_tokens: int, output_tokens: int) -> ModelConfig:
        """在同 tier 候选者中选择最佳模型。

        排序权重: provider 偏好 > 成功率 > 成本
        """
        def score(m: ModelConfig) -> float:
            s = 0.0
            # Provider 偏好
            if preferred_provider and m.provider == preferred_provider:
                s += 100.0
            # Bandit 成功率
            b = self._bandit_state.get(m.name, {})
            total = b.get("successes", 0) + b.get("failures", 0)
            if total > 0:
                s += b["successes"] / total * 10
            else:
                s += 5.0  # 未知模型给中等分
            # 成本倒数（便宜加分）
            cost = m.cost_for(input_tokens, output_tokens)
            if cost > 0:
                s += 0.01 / cost
            return s

        return max(candidates, key=score)

    def _build_fallback_chain(self, chosen: ModelConfig,
                              all_models: list[ModelConfig]) -> list[ModelConfig]:
        """构建 fallback 链 — 当前模型失败时依次尝试的模型。"""
        tier_order = [ModelTier.STANDARD, ModelTier.BUDGET, ModelTier.FALLBACK]
        chain: list[ModelConfig] = []

        for tier in tier_order:
            if tier == chosen.tier:
                continue
            for m in all_models:
                if m.tier == tier and m.name != chosen.name:
                    chain.append(m)
                    break
            if len(chain) >= 2:
                break

        return chain

    # ── 查询 ───────────────────────────────────────

    def get_model(self, name: str) -> ModelConfig | None:
        return self._models.get(name)

    def list_models(self, tier: ModelTier | None = None) -> list[ModelConfig]:
        models = list(self._models.values())
        if tier:
            models = [m for m in models if m.tier == tier]
        return sorted(models, key=lambda m: m.capability_score, reverse=True)

    def get_bandit_stats(self) -> dict[str, dict[str, float]]:
        return dict(self._bandit_state)
