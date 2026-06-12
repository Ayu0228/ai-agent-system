"""Canary Deployment — 金丝雀发布与指标门控升级。

ref: Argo Rollouts — canary deployment with weighted traffic splitting
ref: Flagger — metric-gated progressive delivery for K8s

流程:
  1. Deploy canary (5% traffic)
  2. Observe metrics for N minutes
  3. If metrics pass → increase to 25%
  4. Observe → 50% → 100%
  5. If metrics fail at any step → auto rollback

Each step is gated by SLO checks (error rate, latency, hallucination score, etc.).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class CanaryStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    OBSERVING = "observing"
    PROMOTING = "promoting"
    PROMOTED = "promoted"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CanaryStep:
    """金丝雀发布的一个步骤。"""
    traffic_weight: int             # 流量百分比: 5, 25, 50, 100
    observe_duration_s: int = 300   # 观察时间（秒）
    error_rate_threshold: float = 0.01      # 最大错误率
    latency_p95_threshold_ms: float = 3000  # P95 延迟上限
    hallucination_threshold: float = 0.3    # 最大幻觉分数 (lower=worse)
    min_requests: int = 100                 # 最小请求数才判定


@dataclass
class CanaryConfig:
    """金丝雀发布配置。"""
    agent_id: str
    new_version: str                # 新版本号
    previous_version: str = ""      # 旧版本号（回滚目标）
    steps: list[CanaryStep] = field(default_factory=list)
    auto_promote: bool = True       # 是否自动升级
    auto_rollback: bool = True      # 是否自动回滚
    max_duration_s: int = 3600      # 整个发布最大时长
    slo_namespace: str = ""         # SLO 指标命名空间

    @classmethod
    def default(cls, agent_id: str, new_version: str,
                previous_version: str = "") -> CanaryConfig:
        """默认 4 步金丝雀: 5% → 25% → 50% → 100%。"""
        return cls(
            agent_id=agent_id,
            new_version=new_version,
            previous_version=previous_version,
            steps=[
                CanaryStep(traffic_weight=5, observe_duration_s=180),
                CanaryStep(traffic_weight=25, observe_duration_s=300),
                CanaryStep(traffic_weight=50, observe_duration_s=300),
                CanaryStep(traffic_weight=100, observe_duration_s=180),
            ],
        )


@dataclass
class MetricSnapshot:
    """观测到的指标快照。"""
    error_rate: float = 0.0
    latency_p95_ms: float = 0.0
    hallucination_score: float = 1.0
    total_requests: int = 0
    cost_per_request: float = 0.0
    custom_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class CanaryDeployment:
    """一次金丝雀发布。"""
    config: CanaryConfig
    status: CanaryStatus = CanaryStatus.PENDING
    current_step_index: int = 0
    started_at: float = 0.0
    step_started_at: float = 0.0
    metrics_history: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # 回调 — 由外部注入
    _metric_collector: Callable[[], MetricSnapshot] | None = field(default=None, repr=False)
    _traffic_controller: Callable[[int], bool] | None = field(default=None, repr=False)
    _rollback_handler: Callable[[str, str], bool] | None = field(default=None, repr=False)

    # ── 控制 ───────────────────────────────────────

    def set_collector(self, fn: Callable[[], MetricSnapshot]) -> None:
        self._metric_collector = fn

    def set_traffic_controller(self, fn: Callable[[int], bool]) -> None:
        self._traffic_controller = fn

    def set_rollback_handler(self, fn: Callable[[str, str], bool]) -> None:
        self._rollback_handler = fn

    # ── 执行 ───────────────────────────────────────

    def start(self) -> bool:
        """开始金丝雀发布。"""
        if not self.config.steps:
            self.errors.append("no steps configured")
            self.status = CanaryStatus.FAILED
            return False

        self.status = CanaryStatus.DEPLOYING
        self.started_at = time.time()
        self.current_step_index = 0
        self.step_started_at = time.time()

        # 部署第一步
        step = self.config.steps[0]
        if self._traffic_controller:
            if not self._traffic_controller(step.traffic_weight):
                self.errors.append(f"failed to set traffic to {step.traffic_weight}%")
                self.status = CanaryStatus.FAILED
                return False

        self.status = CanaryStatus.OBSERVING
        logger.info("canary_started", agent=self.config.agent_id,
                    version=self.config.new_version, step=1,
                    traffic=step.traffic_weight)
        return True

    def tick(self) -> CanaryStatus:
        """每个 tick 检查一次（由外部调度器调用）。

        返回当前状态，调用者根据状态决定是否继续 tick。
        """
        if self.status in (CanaryStatus.PROMOTED, CanaryStatus.ROLLED_BACK,
                           CanaryStatus.FAILED):
            return self.status

        step = self.config.steps[self.current_step_index]
        elapsed = time.time() - self.step_started_at

        # 还在观察期
        if elapsed < step.observe_duration_s:
            return self.status

        # 观察期结束 — 收集指标并判断
        metrics = None
        if self._metric_collector:
            metrics = self._metric_collector()
        else:
            metrics = MetricSnapshot()

        self.metrics_history.append({
            "step": self.current_step_index,
            "traffic": step.traffic_weight,
            "timestamp": time.time(),
            "metrics": {
                "error_rate": metrics.error_rate,
                "latency_p95_ms": metrics.latency_p95_ms,
                "hallucination_score": metrics.hallucination_score,
                "total_requests": metrics.total_requests,
            },
        })

        # 检查是否通过 SLO gate
        if self._check_gate(step, metrics):
            # 升级
            return self._promote()
        else:
            # 回滚
            return self._rollback(
                f"step {self.current_step_index + 1} failed SLO gate: "
                f"error_rate={metrics.error_rate:.4f}, "
                f"latency_p95={metrics.latency_p95_ms:.0f}ms"
            )

    def run_sync(self) -> CanaryStatus:
        """同步执行（测试用）— 立即步进所有 step 而不等待。"""
        import time as _time
        if not self.start():
            return self.status

        for _ in self.config.steps:
            # 模拟观察期已过
            self.step_started_at = 0
            status = self.tick()
            if status in (CanaryStatus.ROLLED_BACK, CanaryStatus.FAILED):
                return status

        return self.status

    # ── 内部 ───────────────────────────────────────

    def _check_gate(self, step: CanaryStep, metrics: MetricSnapshot) -> bool:
        """检查指标是否通过门控。"""
        checks: list[tuple[bool, str]] = [
            (metrics.error_rate <= step.error_rate_threshold,
             f"error_rate {metrics.error_rate:.4f} > {step.error_rate_threshold}"),
            (metrics.latency_p95_ms <= step.latency_p95_threshold_ms,
             f"latency_p95 {metrics.latency_p95_ms:.0f} > {step.latency_p95_threshold_ms}"),
            (metrics.hallucination_score >= (1.0 - step.hallucination_threshold),
             f"hallucination_score {metrics.hallucination_score:.3f} < {1.0 - step.hallucination_threshold}"),
            (metrics.total_requests >= step.min_requests,
             f"requests {metrics.total_requests} < {step.min_requests}"),
        ]

        passed = True
        for ok, msg in checks:
            if not ok:
                logger.warning("canary_gate_failed", agent=self.config.agent_id, reason=msg)
                passed = False

        return passed

    def _promote(self) -> CanaryStatus:
        """升级到下一步或完成。"""
        self.current_step_index += 1

        if self.current_step_index >= len(self.config.steps):
            self.status = CanaryStatus.PROMOTED
            logger.info("canary_promoted", agent=self.config.agent_id,
                        version=self.config.new_version,
                        duration=f"{time.time() - self.started_at:.0f}s")
            return self.status

        self.status = CanaryStatus.PROMOTING
        step = self.config.steps[self.current_step_index]

        if self._traffic_controller:
            self._traffic_controller(step.traffic_weight)

        self.step_started_at = time.time()
        self.status = CanaryStatus.OBSERVING

        logger.info("canary_promoting", agent=self.config.agent_id,
                    step=self.current_step_index + 1,
                    traffic=step.traffic_weight)
        return self.status

    def _rollback(self, reason: str) -> CanaryStatus:
        """执行回滚。"""
        self.status = CanaryStatus.ROLLING_BACK
        self.errors.append(reason)

        if self._rollback_handler and self.config.auto_rollback:
            self._rollback_handler(
                self.config.agent_id,
                self.config.previous_version or "stable"
            )

        self.status = CanaryStatus.ROLLED_BACK
        logger.warning("canary_rolled_back", agent=self.config.agent_id,
                       from_version=self.config.new_version,
                       to_version=self.config.previous_version or "stable",
                       reason=reason)
        return self.status

    # ── 查询 ───────────────────────────────────────

    def current_traffic(self) -> int:
        if 0 <= self.current_step_index < len(self.config.steps):
            return self.config.steps[self.current_step_index].traffic_weight
        return 0

    @property
    def is_active(self) -> bool:
        return self.status in (CanaryStatus.DEPLOYING, CanaryStatus.OBSERVING,
                               CanaryStatus.PROMOTING)

    @property
    def elapsed_s(self) -> float:
        if self.started_at == 0:
            return 0.0
        return time.time() - self.started_at
