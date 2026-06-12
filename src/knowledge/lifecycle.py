"""Knowledge Lifecycle — 知识新鲜度追踪、过期检测、自动归档。

ref: Lilian Weng — memory freshness and expiration in agent systems
ref: MemGPT — tiered memory with active management policies
ref: LangChain — knowledge lifecycle with refresh triggers

生命周期:
  FRESH → AGING → STALE → ARCHIVED
   ↳ 每次被引用时更新 last_accessed
   ↳ 到达 TTL 后标记为 STALE
   ↳ STALE 后 N 天自动 ARCHIVED
   ↳ 可通过 review 手动标记为 FRESH (刷新 TTL)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class FreshnessStatus(str, Enum):
    FRESH = "fresh"           # TTL 内，可信任
    AGING = "aging"           # 接近 TTL，标记关注
    STALE = "stale"           # 超过 TTL，需要审核
    ARCHIVED = "archived"     # 超过 TTL + grace，已归档
    REVIEWED = "reviewed"     # 人工审核确认过的


@dataclass
class KnowledgeItem:
    """知识条目，含生命周期元数据。"""

    id: str
    content: str
    source: str = ""                       # 来源: agent_id / url / document
    category: str = "general"              # 分类
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_days: int = 90                     # 新鲜期（天）
    stale_grace_days: int = 30             # 过期后宽限期
    status: FreshnessStatus = FreshnessStatus.FRESH
    importance: float = 0.5                # 0-1 重要性评分
    references: list[str] = field(default_factory=list)  # 引用此条目的其他 item ID
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    @property
    def days_since_access(self) -> float:
        return (time.time() - self.last_accessed) / 86400


class KnowledgeLifecycle:
    """知识生命周期管理器。

    用法:
        kl = KnowledgeLifecycle()
        kl.add_item(KnowledgeItem(id="k1", content="...", ttl_days=90))
        kl.access("k1")  # 记录访问

        # 定期执行
        stale = kl.check_freshness()
        archived = kl.archive_stale()

        # 搜索可用知识
        fresh = kl.get_fresh(category="research")
    """

    def __init__(self) -> None:
        self._items: dict[str, KnowledgeItem] = {}
        # 索引
        self._by_category: dict[str, list[str]] = defaultdict(list)
        self._by_tag: dict[str, list[str]] = defaultdict(list)
        self._by_source: dict[str, list[str]] = defaultdict(list)

    # ── CRUD ───────────────────────────────────────

    # 类型特定 TTL（天）
    _CATEGORY_TTL: dict[str, int] = {
        "rule": 30,
        "fact": 180,
    }

    def add_item(self, item: KnowledgeItem) -> None:
        """添加知识条目。根据 category 自动设置 TTL（rule=30d, fact=180d, 其他保持默认 90d）。"""
        if item.category in self._CATEGORY_TTL:
            item.ttl_days = self._CATEGORY_TTL[item.category]
        self._items[item.id] = item
        self._by_category[item.category].append(item.id)
        for tag in item.tags:
            self._by_tag[tag].append(item.id)
        if item.source:
            self._by_source[item.source].append(item.id)
        logger.debug("knowledge_added", id=item.id, category=item.category)

    def get(self, item_id: str) -> KnowledgeItem | None:
        return self._items.get(item_id)

    def update(self, item_id: str, **kwargs: Any) -> bool:
        item = self._items.get(item_id)
        if not item:
            return False
        for k, v in kwargs.items():
            if hasattr(item, k):
                setattr(item, k, v)
        item.updated_at = time.time()
        item.version += 1
        return True

    def remove(self, item_id: str) -> bool:
        item = self._items.pop(item_id, None)
        if not item:
            return False
        self._by_category[item.category].remove(item_id)
        for tag in item.tags:
            if item_id in self._by_tag[tag]:
                self._by_tag[tag].remove(item_id)
        if item.source and item_id in self._by_source[item.source]:
            self._by_source[item.source].remove(item_id)
        return True

    # ── 访问追踪 ───────────────────────────────────

    def access(self, item_id: str) -> KnowledgeItem | None:
        """记录一次访问。"""
        item = self._items.get(item_id)
        if item and item.status != FreshnessStatus.ARCHIVED:
            item.last_accessed = time.time()
            item.access_count += 1
            # 自动提升重要性
            item.importance = min(1.0, item.importance + 0.01)
        return item

    # ── 新鲜度检查 ─────────────────────────────────

    def check_freshness(self) -> dict[FreshnessStatus, list[str]]:
        """检查所有条目的新鲜度，更新状态。

        返回按状态分组的 item_id 列表。
        """
        now = time.time()
        results: dict[FreshnessStatus, list[str]] = defaultdict(list)

        for item in self._items.values():
            if item.status == FreshnessStatus.ARCHIVED:
                results[FreshnessStatus.ARCHIVED].append(item.id)
                continue
            if item.status == FreshnessStatus.REVIEWED:
                results[FreshnessStatus.REVIEWED].append(item.id)
                continue

            age_seconds = now - item.created_at
            ttl_seconds = item.ttl_days * 86400
            total_expire_seconds = (item.ttl_days + item.stale_grace_days) * 86400

            if age_seconds > total_expire_seconds:
                item.status = FreshnessStatus.ARCHIVED
                results[FreshnessStatus.ARCHIVED].append(item.id)
            elif age_seconds > ttl_seconds:
                item.status = FreshnessStatus.STALE
                results[FreshnessStatus.STALE].append(item.id)
            elif age_seconds > ttl_seconds * 0.8:
                item.status = FreshnessStatus.AGING
                results[FreshnessStatus.AGING].append(item.id)
            else:
                item.status = FreshnessStatus.FRESH
                results[FreshnessStatus.FRESH].append(item.id)

        logger.info("freshness_check", fresh=len(results[FreshnessStatus.FRESH]),
                    aging=len(results[FreshnessStatus.AGING]),
                    stale=len(results[FreshnessStatus.STALE]),
                    archived=len(results[FreshnessStatus.ARCHIVED]))
        return dict(results)

    def archive_stale(self) -> list[str]:
        """归档过期条目，返回归档的 id 列表。"""
        archived: list[str] = []
        for item in self._items.values():
            if item.status == FreshnessStatus.STALE:
                age_seconds = time.time() - item.created_at
                if age_seconds > (item.ttl_days + item.stale_grace_days) * 86400:
                    item.status = FreshnessStatus.ARCHIVED
                    archived.append(item.id)
        if archived:
            logger.info("knowledge_archived", count=len(archived))
        return archived

    def mark_reviewed(self, item_id: str) -> bool:
        """人工标记为已审核 — 刷新 TTL。"""
        item = self._items.get(item_id)
        if not item:
            return False
        item.status = FreshnessStatus.REVIEWED
        item.created_at = time.time()  # 重置时钟
        item.ttl_days = max(item.ttl_days, 90)
        logger.info("knowledge_reviewed", id=item_id)
        return True

    # ── 查询 ───────────────────────────────────────

    def get_fresh(self, category: str = "", min_importance: float = 0.0,
                  limit: int = 50) -> list[KnowledgeItem]:
        """获取新鲜的知识条目。"""
        ids = self._by_category.get(category, list(self._items.keys())) if category else list(self._items.keys())
        items = []
        for iid in ids:
            item = self._items.get(iid)
            if item and item.status in (FreshnessStatus.FRESH, FreshnessStatus.REVIEWED, FreshnessStatus.AGING):
                if item.importance >= min_importance:
                    items.append(item)
        items.sort(key=lambda x: (x.importance, x.access_count), reverse=True)
        return items[:limit]

    def get_stale(self, category: str = "") -> list[KnowledgeItem]:
        """获取过期/待审核的知识。"""
        ids = self._by_category.get(category, list(self._items.keys())) if category else list(self._items.keys())
        return [self._items[iid] for iid in ids
                if self._items[iid].status == FreshnessStatus.STALE]

    def search(self, query: str, category: str = "",
               limit: int = 20) -> list[KnowledgeItem]:
        """简单关键词搜索。"""
        results: list[tuple[KnowledgeItem, int]] = []
        query_lower = query.lower()
        ids = self._by_category.get(category, list(self._items.keys())) if category else list(self._items.keys())
        for iid in ids:
            item = self._items.get(iid)
            if not item or item.status == FreshnessStatus.ARCHIVED:
                continue
            score = item.content.lower().count(query_lower)
            for tag in item.tags:
                if query_lower in tag.lower():
                    score += 3
            if score > 0:
                results.append((item, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:limit]]

    # ── 统计 ───────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        statuses = defaultdict(int)
        for item in self._items.values():
            statuses[item.status.value] += 1
        return {
            "total_items": len(self._items),
            "by_status": dict(statuses),
            "by_category": {k: len(v) for k, v in self._by_category.items()},
            "avg_importance": sum(i.importance for i in self._items.values()) / max(1, len(self._items)),
            "total_accesses": sum(i.access_count for i in self._items.values()),
        }
