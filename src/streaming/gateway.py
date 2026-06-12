"""Edge Gateway — 边缘部署网关，优化延迟和带宽。

ref: Cloudflare Workers AI — edge-optimized inference
ref: Anthropic — edge deployment patterns for latency-sensitive agent apps
ref: Fly.io / Vercel Edge — global distribution with cold start optimization

功能:
  - 请求路由: 根据用户地理位置路由到最近节点
  - 响应缓存: 相同查询的缓存策略
  - 连接池: 复用 LLM API 连接
  - 速率限制: per-user / per-agent 限流
  - 冷启动优化: 预热模型 + lazy init
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class GatewayStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"


@dataclass
class GatewayConfig:
    """边缘网关配置。"""
    name: str = "default"
    region: str = "auto"                    # us-east / eu-west / ap-southeast / auto
    max_connections: int = 100
    connection_timeout_s: float = 30.0
    rate_limit_per_minute: int = 60
    cache_ttl_s: int = 300                  # 响应缓存时间
    enable_cache: bool = True
    enable_compression: bool = True
    warm_models: list[str] = field(default_factory=list)   # 预热的模型列表
    health_check_interval_s: int = 30


@dataclass
class EdgeNode:
    """边缘节点。"""
    name: str
    region: str
    endpoint: str
    status: GatewayStatus = GatewayStatus.HEALTHY
    load: int = 0                           # 当前连接数
    max_load: int = 100
    avg_latency_ms: float = 50.0
    last_health_check: float = 0.0


class EdgeGateway:
    """边缘部署网关。

    用法:
        gw = EdgeGateway(GatewayConfig(name="api-gw", region="auto"))
        gw.register_node(EdgeNode(name="node-1", region="us-east", endpoint="..."))
        gw.register_node(EdgeNode(name="node-2", region="ap-southeast", endpoint="..."))

        # 路由到最佳节点
        node = gw.route(user_region="ap-southeast", agent_id="researcher")

        # 速率限制
        if not gw.check_rate_limit(user_id="user-123"):
            raise RateLimitExceeded
    """

    def __init__(self, config: GatewayConfig | None = None) -> None:
        self.config = config or GatewayConfig()
        self._nodes: dict[str, EdgeNode] = {}
        self._regions: dict[str, list[str]] = defaultdict(list)

        # 速率限制状态
        self._rate_counters: dict[str, tuple[int, float]] = {}  # user_id → (count, window_start)

        # 响应缓存
        self._cache: dict[str, tuple[Any, float]] = {}  # cache_key → (response, expires_at)

        # 连接池（模拟）
        self._active_connections: dict[str, int] = defaultdict(int)

    # ── 节点管理 ───────────────────────────────────

    def register_node(self, node: EdgeNode) -> None:
        self._nodes[node.name] = node
        self._regions[node.region].append(node.name)
        logger.info("edge_node_registered", name=node.name, region=node.region)

    def unregister_node(self, name: str) -> None:
        node = self._nodes.pop(name, None)
        if node:
            self._regions[node.region].remove(name)

    # ── 路由 ───────────────────────────────────────

    def route(self, user_region: str = "", agent_id: str = "",
              prefer_low_latency: bool = True) -> EdgeNode | None:
        """路由到最佳边缘节点。

        策略:
          1. 同区域优先
          2. 负载最低
          3. 延迟最低
        """
        available = [n for n in self._nodes.values()
                     if n.status != GatewayStatus.OFFLINE
                     and n.load < n.max_load]

        if not available:
            return None

        # 同区域节点
        same_region = [n for n in available if n.region == user_region]
        candidates = same_region if same_region else available

        if prefer_low_latency:
            candidates.sort(key=lambda n: (n.load / max(1, n.max_load)) + n.avg_latency_ms / 1000)
        else:
            candidates.sort(key=lambda n: n.load / max(1, n.max_load))

        chosen = candidates[0]
        chosen.load += 1
        self._active_connections[chosen.name] += 1

        logger.debug("edge_routed", node=chosen.name, region=chosen.region,
                     user_region=user_region)
        return chosen

    # ── 速率限制 ───────────────────────────────────

    def check_rate_limit(self, user_id: str) -> bool:
        """检查用户速率限制。"""
        now = time.time()
        counter = self._rate_counters.get(user_id)

        if counter is None or (now - counter[1]) > 60:
            # 新窗口
            self._rate_counters[user_id] = (1, now)
            return True

        count, window_start = counter
        if count >= self.config.rate_limit_per_minute:
            logger.warning("rate_limit_exceeded", user_id=user_id, count=count)
            return False

        self._rate_counters[user_id] = (count + 1, window_start)
        return True

    def get_rate_limit_remaining(self, user_id: str) -> int:
        counter = self._rate_counters.get(user_id)
        if not counter:
            return self.config.rate_limit_per_minute
        count, window_start = counter
        if time.time() - window_start > 60:
            return self.config.rate_limit_per_minute
        return max(0, self.config.rate_limit_per_minute - count)

    # ── 缓存 ───────────────────────────────────────

    def cache_get(self, cache_key: str) -> Any | None:
        """获取缓存的响应。"""
        if not self.config.enable_cache:
            return None
        entry = self._cache.get(cache_key)
        if entry and time.time() < entry[1]:
            return entry[0]
        if entry:
            del self._cache[cache_key]
        return None

    def cache_set(self, cache_key: str, response: Any) -> None:
        if not self.config.enable_cache:
            return
        expires = time.time() + self.config.cache_ttl_s
        self._cache[cache_key] = (response, expires)

        # 过期清理
        if len(self._cache) > 1000:
            now = time.time()
            self._cache = {k: v for k, v in self._cache.items() if v[1] > now}

    # ── 健康检查 ───────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """检查所有节点的健康状态。"""
        now = time.time()
        for node in self._nodes.values():
            node.last_health_check = now
            if node.load > node.max_load * 0.9:
                node.status = GatewayStatus.OVERLOADED
            elif node.avg_latency_ms > 1000:
                node.status = GatewayStatus.DEGRADED
            else:
                node.status = GatewayStatus.HEALTHY

        return self.get_status()

    # ── 统计 ───────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        nodes = [{
            "name": n.name,
            "region": n.region,
            "status": n.status.value,
            "load": n.load,
            "load_pct": f"{n.load / max(1, n.max_load):.1%}",
            "avg_latency_ms": n.avg_latency_ms,
        } for n in self._nodes.values()]

        return {
            "gateway": self.config.name,
            "region": self.config.region,
            "nodes_total": len(self._nodes),
            "nodes_healthy": sum(1 for n in self._nodes.values()
                                 if n.status == GatewayStatus.HEALTHY),
            "nodes": nodes,
            "active_connections": sum(self._active_connections.values()),
            "cache_entries": len(self._cache),
            "rate_limited_users": sum(
                1 for uid in self._rate_counters
                if not self.check_rate_limit(uid)
            ),
        }
