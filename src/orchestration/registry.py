"""Agent Registry — 服务发现、健康检查、能力注册。

ref: Microsoft Multi-Agent RA — Agent Registry pattern
ref: AWS Strands Agents 1.0 — agent identity & capabilities

每个 agent 向 Registry 注册自己的能力、当前负载和健康状态。
Orchestrator 通过 Registry 发现可用的 agent 并做能力路由。
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.shared.models import ToolDefinition, ToolValidationResult

logger = structlog.get_logger()


class AgentStatus(str, Enum):
    ONLINE = "online"
    BUSY = "busy"
    IDLE = "idle"
    OFFLINE = "offline"
    DEGRADED = "degraded"


@dataclass
class AgentInfo:
    """Agent 注册信息。"""

    agent_id: str
    name: str = ""
    capabilities: list[str] = field(default_factory=list)
    tools: list[str | ToolDefinition] = field(default_factory=list)
    model: str = ""
    status: AgentStatus = AgentStatus.OFFLINE
    max_concurrency: int = 3
    current_load: int = 0
    registered_at: float = 0.0
    last_heartbeat: float = 0.0
    health_score: float = 1.0         # 0-1，综合健康评分
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        return (
            self.status in (AgentStatus.ONLINE, AgentStatus.IDLE)
            and self.current_load < self.max_concurrency
            and self.health_score > 0.3
        )

    @property
    def capacity(self) -> float:
        """可用容量比例。"""
        if self.max_concurrency == 0:
            return 0.0
        return max(0.0, 1.0 - self.current_load / self.max_concurrency)


class AgentRegistry:
    """Agent 注册中心。

    用法:
        registry = AgentRegistry()
        registry.register(AgentInfo(agent_id="researcher", capabilities=["web_search", "fact_check"]))
        agents = registry.find_by_capability("web_search")
        best = registry.select_best("web_search")  # 健康度+容量加权
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentInfo] = {}
        # 能力倒排索引: capability -> [agent_id, ...]
        self._capability_index: dict[str, list[str]] = defaultdict(list)
        # 工具定义索引: agent_id:tool_name -> ToolDefinition
        self._tool_index: dict[str, ToolDefinition] = {}

    # ── 注册 / 注销 ────────────────────────────────

    def register(self, info: AgentInfo, validate_tools: bool = True) -> None:
        """注册或更新 agent 信息。"""
        old = self._agents.get(info.agent_id)
        if old:
            self._remove_from_index(old)

        info.registered_at = info.registered_at or time.time()
        info.last_heartbeat = info.last_heartbeat or time.time()
        self._agents[info.agent_id] = info
        self._add_to_index(info)
        self._index_tools(info)
        logger.debug("agent_registered", agent_id=info.agent_id,
                     capabilities=info.capabilities)

    def unregister(self, agent_id: str) -> None:
        info = self._agents.pop(agent_id, None)
        if info:
            self._remove_from_index(info)
            logger.debug("agent_unregistered", agent_id=agent_id)

    def heartbeat(self, agent_id: str) -> bool:
        """更新心跳时间，返回是否成功。"""
        info = self._agents.get(agent_id)
        if info:
            info.last_heartbeat = time.time()
            if info.status == AgentStatus.OFFLINE:
                info.status = AgentStatus.ONLINE
            return True
        return False

    # ── 状态管理 ───────────────────────────────────

    def update_status(self, agent_id: str, status: AgentStatus,
                      current_load: int | None = None) -> None:
        info = self._agents.get(agent_id)
        if info:
            info.status = status
            if current_load is not None:
                info.current_load = current_load
            logger.debug("agent_status_updated", agent_id=agent_id,
                         status=status.value)

    def update_health(self, agent_id: str, score: float) -> None:
        info = self._agents.get(agent_id)
        if info:
            info.health_score = max(0.0, min(1.0, score))
            if score < 0.3 and info.status != AgentStatus.OFFLINE:
                info.status = AgentStatus.DEGRADED

    def mark_offline_stale(self, heartbeat_timeout: float = 60.0) -> int:
        """将超时未心跳的 agent 标记为 offline，返回标记数量。"""
        now = time.time()
        count = 0
        for info in self._agents.values():
            if info.status != AgentStatus.OFFLINE and \
               (now - info.last_heartbeat) > heartbeat_timeout:
                info.status = AgentStatus.OFFLINE
                count += 1
                logger.warning("agent_heartbeat_timeout", agent_id=info.agent_id)
        return count

    # ── 查询 ───────────────────────────────────────

    def get(self, agent_id: str) -> AgentInfo | None:
        return self._agents.get(agent_id)

    def list_all(self) -> list[AgentInfo]:
        return list(self._agents.values())

    def list_available(self) -> list[AgentInfo]:
        return [a for a in self._agents.values() if a.is_available]

    def find_by_capability(self, capability: str) -> list[AgentInfo]:
        """通过能力关键字查找可用的 agent。"""
        agent_ids = self._capability_index.get(capability, [])
        return [self._agents[aid] for aid in agent_ids
                if aid in self._agents and self._agents[aid].is_available]

    def select_best(self, capability: str) -> AgentInfo | None:
        """选择最佳 agent — 健康度 × 容量 加权随机选择。

        权重 = health_score * capacity
        """
        candidates = self.find_by_capability(capability)
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # 加权选择
        weights = [a.health_score * a.capacity for a in candidates]
        total = sum(weights)
        if total == 0:
            return candidates[0]

        import random
        r = random.uniform(0, total)
        cumulative = 0.0
        for agent, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return agent
        return candidates[-1]

    # ── 内部 ───────────────────────────────────────

    def _add_to_index(self, info: AgentInfo) -> None:
        for cap in info.capabilities:
            if info.agent_id not in self._capability_index[cap]:
                self._capability_index[cap].append(info.agent_id)

    def _remove_from_index(self, info: AgentInfo) -> None:
        for cap in info.capabilities:
            lst = self._capability_index.get(cap, [])
            if info.agent_id in lst:
                lst.remove(info.agent_id)
        # 清理旧工具索引
        for tool in info.tools:
            name = tool.name if isinstance(tool, ToolDefinition) else tool
            key = f"{info.agent_id}:{name}"
            self._tool_index.pop(key, None)

    def _index_tools(self, info: AgentInfo) -> None:
        """将 agent 的工具定义索引到 _tool_index。"""
        for tool in info.tools:
            if isinstance(tool, ToolDefinition):
                key = f"{info.agent_id}:{tool.name}"
                self._tool_index[key] = tool

    def get_tool(self, agent_id: str, tool_name: str) -> ToolDefinition | None:
        """获取 agent 的工具定义。"""
        return self._tool_index.get(f"{agent_id}:{tool_name}")

    def validate_tool_params(self, agent_id: str, tool_name: str,
                              params: dict[str, Any]) -> ToolValidationResult:
        """校验工具参数。未注册的工具返回 valid=True（宽松模式）。"""
        tool_def = self.get_tool(agent_id, tool_name)
        if tool_def is None:
            return ToolValidationResult(valid=True, warnings=[f"Tool '{tool_name}' not registered for '{agent_id}' — skipping schema check"])
        return tool_def.validate(params)

    # ── 统计 ───────────────────────────────────────

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def online_count(self) -> int:
        return sum(1 for a in self._agents.values()
                   if a.status != AgentStatus.OFFLINE)

    def get_stats(self) -> dict[str, Any]:
        agents = self._agents.values()
        return {
            "total": len(agents),
            "online": sum(1 for a in agents if a.status != AgentStatus.OFFLINE),
            "available": sum(1 for a in agents if a.is_available),
            "degraded": sum(1 for a in agents if a.status == AgentStatus.DEGRADED),
            "total_load": sum(a.current_load for a in agents),
            "total_capacity": sum(a.max_concurrency for a in agents),
        }
