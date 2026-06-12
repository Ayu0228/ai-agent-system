"""Rollback Manager — 自动回滚触发与安全策略。

ref: Argo Rollouts — automated rollback with configurable thresholds
ref: Flagger — metric-gated promotion with automatic rollback on SLO breach

回滚触发条件:
  - 错误率超过阈值
  - 延迟 P95 超过阈值
  - 幻觉检测分数下降
  - 成本异常（单次请求成本飙升 > 3x baseline）
  - 连续 N 次健康检查失败
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class RollbackTrigger(str, Enum):
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    HALLUCINATION = "hallucination"
    COST_SPIKE = "cost_spike"
    HEALTH_CHECK = "health_check"
    MANUAL = "manual"
    TIMEOUT = "timeout"


@dataclass
class RollbackPolicy:
    """回滚策略配置。"""
    agent_id: str
    error_rate_threshold: float = 0.05        # 5% 错误率触发回滚
    latency_p95_threshold_ms: float = 5000     # P95 延迟上限
    hallucination_score_min: float = 0.5       # 最低幻觉分数
    cost_spike_multiplier: float = 3.0         # 成本超过 baseline 的倍数
    consecutive_failures: int = 3              # 连续失败次数触发
    cooldown_s: int = 300                      # 回滚冷却时间（秒）
    enabled: bool = True


@dataclass
class RollbackEvent:
    """一次回滚事件。"""
    agent_id: str
    trigger: RollbackTrigger
    from_version: str
    to_version: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    success: bool = False


class RollbackManager:
    """自动回滚管理器。

    用法:
        mgr = RollbackManager()
        mgr.set_policy(RollbackPolicy(agent_id="researcher"))

        # 每个评估周期调用
        trigger = mgr.evaluate("researcher", metrics)
        if trigger:
            mgr.execute_rollback("researcher", from_ver="v1.2", to_ver="v1.1")
    """

    def __init__(self) -> None:
        self._policies: dict[str, RollbackPolicy] = {}
        self._history: list[RollbackEvent] = []
        self._consecutive_failures: dict[str, int] = {}
        self._last_rollback: dict[str, float] = {}
        self._rollback_handler: Callable[[str, str, str], bool] | None = None

    # ── 配置 ───────────────────────────────────────

    def set_policy(self, policy: RollbackPolicy) -> None:
        self._policies[policy.agent_id] = policy

    def set_handler(self, handler: Callable[[str, str, str], bool]) -> None:
        """设置回滚执行回调: async fn(agent_id, from_version, to_version) -> bool"""
        self._rollback_handler = handler

    def remove_policy(self, agent_id: str) -> None:
        self._policies.pop(agent_id, None)

    # ── 评估 ───────────────────────────────────────

    def evaluate(self, agent_id: str, metrics: dict[str, float],
                 baseline_cost: float = 0.0) -> list[RollbackTrigger]:
        """评估是否应该回滚。返回触发的触发器列表（空=安全）。"""
        policy = self._policies.get(agent_id)
        if not policy or not policy.enabled:
            return []

        # 冷却期检查
        last = self._last_rollback.get(agent_id, 0)
        if time.time() - last < policy.cooldown_s:
            return []

        triggers: list[RollbackTrigger] = []

        error_rate = metrics.get("error_rate", 0.0)
        latency = metrics.get("latency_p95_ms", 0.0)
        hallucination = metrics.get("hallucination_score", 1.0)
        cost = metrics.get("cost_per_request", 0.0)

        if error_rate > policy.error_rate_threshold:
            triggers.append(RollbackTrigger.ERROR_RATE)
            logger.warning("rollback_trigger_error_rate", agent=agent_id,
                           current=error_rate, threshold=policy.error_rate_threshold)

        if latency > policy.latency_p95_threshold_ms:
            triggers.append(RollbackTrigger.LATENCY)
            logger.warning("rollback_trigger_latency", agent=agent_id,
                           current=latency, threshold=policy.latency_p95_threshold_ms)

        if hallucination < policy.hallucination_score_min:
            triggers.append(RollbackTrigger.HALLUCINATION)
            logger.warning("rollback_trigger_hallucination", agent=agent_id,
                           current=hallucination, min=policy.hallucination_score_min)

        if baseline_cost > 0 and cost > baseline_cost * policy.cost_spike_multiplier:
            triggers.append(RollbackTrigger.COST_SPIKE)
            logger.warning("rollback_trigger_cost", agent=agent_id,
                           current=cost, baseline=baseline_cost)

        # 连续失败计数
        if triggers:
            self._consecutive_failures[agent_id] = \
                self._consecutive_failures.get(agent_id, 0) + 1
        else:
            self._consecutive_failures[agent_id] = 0

        # 连续 N 次触发
        if self._consecutive_failures.get(agent_id, 0) >= policy.consecutive_failures:
            if RollbackTrigger.HEALTH_CHECK not in triggers:
                triggers.append(RollbackTrigger.HEALTH_CHECK)

        return triggers

    def execute_rollback(self, agent_id: str, from_version: str,
                         to_version: str, trigger: RollbackTrigger = RollbackTrigger.MANUAL,
                         reason: str = "", metrics: dict[str, Any] | None = None) -> RollbackEvent:
        """执行回滚。"""
        event = RollbackEvent(
            agent_id=agent_id,
            trigger=trigger,
            from_version=from_version,
            to_version=to_version,
            reason=reason or f"auto-rollback triggered by {trigger.value}",
            metrics_snapshot=metrics or {},
        )

        success = True
        if self._rollback_handler:
            try:
                success = self._rollback_handler(agent_id, from_version, to_version)
            except Exception as e:
                logger.error("rollback_handler_error", agent=agent_id, error=str(e))
                success = False

        event.success = success
        self._history.append(event)
        self._last_rollback[agent_id] = time.time()
        self._consecutive_failures[agent_id] = 0

        logger.info("rollback_executed", agent=agent_id,
                    from_ver=from_version, to_ver=to_version,
                    trigger=trigger.value, success=success)
        return event

    # ── 查询 ───────────────────────────────────────

    def get_history(self, agent_id: str = "",
                    limit: int = 20) -> list[RollbackEvent]:
        events = self._history
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        return events[-limit:]

    def get_consecutive_failures(self, agent_id: str) -> int:
        return self._consecutive_failures.get(agent_id, 0)

    def is_in_cooldown(self, agent_id: str) -> bool:
        policy = self._policies.get(agent_id)
        if not policy:
            return False
        last = self._last_rollback.get(agent_id, 0)
        return (time.time() - last) < policy.cooldown_s
