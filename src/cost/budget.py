"""Budget Manager — 多层级 Token 预算管理。

ref: ParetoBandit — budget-paced adaptive routing
ref: Anthropic cost management — per-workspace budgets with hard/soft limits

层级:
  Tier 1: 全局每日预算（跨所有 agent + workflow）
  Tier 2: Agent 每日预算（per-agent）
  Tier 3: Workflow 单次预算（per-run）
  Tier 4: Session 预算（per-conversation）

每一层都可配置硬/软限制。软限制触发告警，硬限制阻断调用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class BudgetTier(str, Enum):
    GLOBAL = "global"
    AGENT = "agent"
    WORKFLOW = "workflow"
    SESSION = "session"


@dataclass
class BudgetConfig:
    """单层预算配置。"""
    tier: BudgetTier
    budget_id: str                      # "global" / agent_id / workflow_id
    daily_limit: int = 100_000          # tokens
    per_run_limit: int = 0              # 0 = unlimited
    hard_limit: bool = False            # True = 硬阻断, False = 软告警
    warn_threshold: float = 0.8         # 80% 时告警


@dataclass
class BudgetStatus:
    """当前预算状态。"""
    config: BudgetConfig
    tokens_used_today: int = 0
    tokens_used_this_run: int = 0
    last_reset: float = 0.0
    is_exceeded: bool = False
    is_warning: bool = False


class BudgetManager:
    """多层级 Token 预算管理器。

    用法:
        mgr = BudgetManager()
        mgr.configure(BudgetConfig(tier=BudgetTier.GLOBAL, budget_id="global",
                                    daily_limit=500_000))
        mgr.configure(BudgetConfig(tier=BudgetTier.AGENT, budget_id="researcher",
                                    daily_limit=50_000))

        status = mgr.check("researcher")  # 检查是否可调用
        if status.is_exceeded:
            raise BudgetExceededError(...)
        mgr.record("researcher", tokens=1500)
    """

    def __init__(self) -> None:
        self._configs: dict[str, BudgetConfig] = {}      # key: "tier:budget_id"
        self._usage: dict[str, BudgetStatus] = {}        # key: "tier:budget_id"

    # ── 配置 ───────────────────────────────────────

    def configure(self, cfg: BudgetConfig) -> None:
        key = f"{cfg.tier.value}:{cfg.budget_id}"
        self._configs[key] = cfg
        if key not in self._usage:
            self._usage[key] = BudgetStatus(config=cfg, last_reset=time.time())
        logger.debug("budget_configured", tier=cfg.tier.value,
                     budget_id=cfg.budget_id, daily_limit=cfg.daily_limit)

    def configure_defaults(self) -> None:
        """设置合理的默认预算。"""
        self.configure(BudgetConfig(
            tier=BudgetTier.GLOBAL, budget_id="global",
            daily_limit=500_000, hard_limit=True, warn_threshold=0.85,
        ))
        for agent in ["researcher", "copywriter", "data-analyst", "tech-dev",
                       "script-editor", "visual-designer", "product-designer",
                       "ops-monitor", "investment-analyst", "content-strategist"]:
            self.configure(BudgetConfig(
                tier=BudgetTier.AGENT, budget_id=agent,
                daily_limit=50_000, hard_limit=False, warn_threshold=0.8,
            ))
        # WORKFLOW tier: per-run 预算，用于控制反射轮次等成本
        self.configure(BudgetConfig(
            tier=BudgetTier.WORKFLOW, budget_id="reflection",
            per_run_limit=20_000, hard_limit=False, warn_threshold=0.9,
        ))

    # ── 检查 / 记录 ────────────────────────────────

    def check(self, agent_id: str = "", workflow_id: str = "",
              session_id: str = "") -> BudgetStatus:
        """检查所有相关预算层，返回最高优先级的阻断状态。

        优先级: GLOBAL > AGENT > WORKFLOW > SESSION
        """
        keys = self._resolve_keys(agent_id, workflow_id, session_id)
        worst: BudgetStatus | None = None

        for key in keys:
            status = self._get_status(key)
            if status is None:
                continue
            if status.is_exceeded and status.config.hard_limit:
                return status  # 硬阻断，立即返回
            if worst is None or (
                status.is_exceeded and not worst.is_exceeded
            ):
                worst = status

        return worst or BudgetStatus(
            config=BudgetConfig(tier=BudgetTier.GLOBAL, budget_id="fallback"))

    def record(self, tokens: int, agent_id: str = "",
               workflow_id: str = "", session_id: str = "") -> dict[str, BudgetStatus]:
        """记录 token 消耗，返回所有更新后的状态。"""
        keys = self._resolve_keys(agent_id, workflow_id, session_id)
        results: dict[str, BudgetStatus] = {}

        for key in keys:
            cfg = self._configs.get(key)
            if not cfg:
                continue
            status = self._get_status(key)
            if status is None:
                continue
            status.tokens_used_today += tokens
            status.tokens_used_this_run += tokens

            # 检测超标
            daily_pct = status.tokens_used_today / cfg.daily_limit if cfg.daily_limit else 0
            status.is_warning = daily_pct >= cfg.warn_threshold
            status.is_exceeded = (
                (cfg.daily_limit > 0 and status.tokens_used_today >= cfg.daily_limit)
                or (cfg.per_run_limit > 0 and status.tokens_used_this_run >= cfg.per_run_limit)
            )

            if status.is_exceeded:
                logger.warning("budget_exceeded", tier=cfg.tier.value,
                               budget_id=cfg.budget_id,
                               used=status.tokens_used_today,
                               limit=cfg.daily_limit)
            elif status.is_warning:
                logger.info("budget_warning", tier=cfg.tier.value,
                            budget_id=cfg.budget_id,
                            used=status.tokens_used_today,
                            pct=f"{daily_pct:.0%}")

            results[key] = status

        return results

    def can_proceed(self, tokens_needed: int, agent_id: str = "",
                    workflow_id: str = "", session_id: str = "") -> bool:
        """检查是否有足够预算执行指定 token 数的操作。"""
        for key in self._resolve_keys(agent_id, workflow_id, session_id):
            cfg = self._configs.get(key)
            s = self._usage.get(key)
            if not cfg or not s:
                continue
            if not cfg.hard_limit:
                continue
            # Check daily limit
            if cfg.daily_limit > 0 and s.tokens_used_today + tokens_needed > cfg.daily_limit:
                return False
            # Check per-run limit
            if cfg.per_run_limit > 0 and s.tokens_used_this_run + tokens_needed > cfg.per_run_limit:
                return False
        return True

    # ── 重置 ───────────────────────────────────────

    def reset_daily(self) -> int:
        """重置所有日预算，返回重置数量。"""
        now = time.time()
        count = 0
        for status in self._usage.values():
            if now - status.last_reset > 86400:
                status.tokens_used_today = 0
                status.last_reset = now
                status.is_exceeded = False
                status.is_warning = False
                count += 1
        return count

    def reset_run(self, workflow_id: str) -> None:
        """重置 workflow 运行计数（新 run 开始时调用）。"""
        key = f"{BudgetTier.WORKFLOW.value}:{workflow_id}"
        status = self._usage.get(key)
        if status:
            status.tokens_used_this_run = 0

    # ── 查询 ───────────────────────────────────────

    def get_usage_summary(self) -> dict[str, Any]:
        tiers: dict[str, dict] = {}
        for key, status in self._usage.items():
            tiers[key] = {
                "tier": status.config.tier.value,
                "budget_id": status.config.budget_id,
                "daily_limit": status.config.daily_limit,
                "tokens_used_today": status.tokens_used_today,
                "usage_pct": f"{status.tokens_used_today / status.config.daily_limit:.1%}"
                if status.config.daily_limit else "N/A",
                "is_exceeded": status.is_exceeded,
                "is_warning": status.is_warning,
            }
        return tiers

    # ── 内部 ───────────────────────────────────────

    def _resolve_keys(self, agent_id: str, workflow_id: str,
                      session_id: str) -> list[str]:
        keys = ["global:global"]
        if agent_id:
            keys.append(f"agent:{agent_id}")
        if workflow_id:
            keys.append(f"workflow:{workflow_id}")
        if session_id:
            keys.append(f"session:{session_id}")
        return keys

    def _get_status(self, key: str) -> BudgetStatus | None:
        """获取状态，自动日重置。"""
        if key not in self._configs:
            return None
        cfg = self._configs[key]
        if key not in self._usage:
            self._usage[key] = BudgetStatus(config=cfg, last_reset=time.time())
        status = self._usage[key]
        # 自动日重置
        if time.time() - status.last_reset > 86400:
            status.tokens_used_today = 0
            status.last_reset = time.time()
            status.is_exceeded = False
            status.is_warning = False
        return status
