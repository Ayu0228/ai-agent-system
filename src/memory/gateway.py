"""MemoryGateway —— 记忆系统统一入口。五层记忆 + ACL 访问控制 + 检索路由 + 监控。

所有 Agent 通过此接口读写记忆，内部路由到对应的记忆子系统。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import structlog

from src.memory.long_term import LongTermMemory
from src.memory.short_term import ContextCompressor
from src.memory.write_policy import WritePolicy
from src.shared.models import (
    ExperienceCard,
    ExperienceQuery,
    KnowledgeQuery,
    KnowledgeTriple,
    MemoryEntry,
    MemoryMetrics,
    MemoryProposal,
    MemoryQuery,
    MemorySearchResult,
    MemoryType,
    TaskCognitiveState,
    TaskRecord,
    UserFact,
    UserMemoryQuery,
)

logger = structlog.get_logger()


class MemoryGateway:
    """统一记忆网关。封装五类记忆的读写、ACL 检查、检索路由、监控。"""

    def __init__(self) -> None:
        self._long_term = LongTermMemory()
        self._compressor = ContextCompressor()
        self._write_policy = WritePolicy()
        self._db = self._long_term._db

        # 延迟导入避免循环依赖
        from src.memory.task_memory import TaskMemory
        from src.memory.knowledge_memory import KnowledgeMemory
        from src.memory.experience_memory import ExperienceMemory
        from src.memory.user_memory import UserMemory
        from src.memory.retrieval_router import RetrievalRouter

        self._task = TaskMemory(db=self._db)
        self._knowledge = KnowledgeMemory(db=self._db)
        self._experience = ExperienceMemory(db=self._db)
        self._user = UserMemory(db=self._db)
        self._router = RetrievalRouter()

        # 监控
        self._metrics = MemoryMetrics()
        self._last_latency: dict[str, float] = {}

    # ── Context Memory (L1 Hot) ───────────────────────────

    def compress_context(self, messages: list[dict]) -> list[dict]:
        t0 = time.monotonic()
        result = self._compressor.compress(messages)
        self._last_latency["L1"] = (time.monotonic() - t0) * 1000
        return result

    def should_compress(self, messages: list[dict]) -> bool:
        return self._compressor.should_compress(messages)

    @property
    def context_stats(self) -> dict[str, Any]:
        return {
            "max_tokens": self._compressor.max_tokens,
            "recent_turns": self._compressor.recent_turns,
            "system_prompt_reserved": self._compressor.system_prompt_reserved,
            "threshold": self._compressor.compress_threshold,
            "compression_count": self._compressor._compression_count,
        }

    # ── Task Memory (L2 Warm) ─────────────────────────────

    def create_task(self, owner_agent_id: str, **kwargs: Any) -> TaskRecord:
        t0 = time.monotonic()
        result = self._task.create_task(owner_agent_id, **kwargs)
        self._last_latency["L2"] = (time.monotonic() - t0) * 1000
        return result

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._task.get_task(task_id)

    def get_active_tasks(self, agent_id: str | None = None) -> list[TaskRecord]:
        return self._task.get_active_tasks(agent_id)

    def update_task_status(self, task_id: str, new_status: str, **kwargs: Any) -> TaskRecord | None:
        return self._task.update_task_status(task_id, new_status, **kwargs)

    def mark_task_done(self, task_id: str, **kwargs: Any) -> TaskRecord | None:
        return self._task.mark_done(task_id, **kwargs)

    def check_task_ttl(self) -> list[str]:
        return self._task.check_ttl_and_archive()

    # ── Knowledge Memory (L3 Cold) ────────────────────────

    def search_knowledge(self, query: KnowledgeQuery) -> list[MemorySearchResult]:
        t0 = time.monotonic()
        result = self._knowledge.search(query)
        self._last_latency["L3"] = (time.monotonic() - t0) * 1000
        return result

    def add_knowledge_triple(self, triple: KnowledgeTriple) -> KnowledgeTriple:
        return self._knowledge.add_triple(triple)

    def add_knowledge_triples(self, triples: list[KnowledgeTriple]) -> int:
        return self._knowledge.add_triples_batch(triples)

    def query_knowledge_triples(self, **kwargs: Any) -> list[KnowledgeTriple]:
        return self._knowledge.query_triples(**kwargs)

    def get_pending_conflicts(self):
        return self._knowledge.get_pending_conflicts()

    def resolve_conflict(self, conflict_id: str, resolved_by: str, winner: str) -> None:
        self._knowledge.resolve_conflict(conflict_id, resolved_by, winner)

    # ── Experience Memory (L3 Cold) ───────────────────────

    def store_experience(self, card: ExperienceCard) -> str:
        return self._experience.store(card)

    def search_experience(self, query: ExperienceQuery) -> list[ExperienceCard]:
        t0 = time.monotonic()
        result = self._experience.search(query)
        self._last_latency["L3_exp"] = (time.monotonic() - t0) * 1000
        return result

    def get_experience(self, experience_id: str) -> ExperienceCard | None:
        return self._experience.get(experience_id)

    def use_experience(self, experience_id: str) -> None:
        self._experience.update_usage(experience_id)

    def share_experience(self, experience_id: str) -> bool:
        return self._experience.mark_shareable(experience_id)

    def memify(self) -> dict[str, int]:
        return self._experience.memify()

    # ── User Memory (L3 Warm Priority) ────────────────────

    def write_user_fact(self, fact: UserFact, **kwargs: Any) -> UserFact | None:
        return self._user.write_fact(fact, **kwargs)

    def submit_user_proposal(self, proposal: MemoryProposal) -> str:
        return self._user.submit_proposal(proposal)

    def get_pending_proposals(self) -> list[MemoryProposal]:
        return self._user.get_pending_proposals()

    def review_proposal(self, proposal_id: str, approved: bool) -> MemoryProposal | None:
        return self._user.review_proposal(proposal_id, approved)

    def degrade_stale_proposals(self) -> int:
        return self._user.degrade_stale_proposals()

    def search_user_facts(self, query: UserMemoryQuery) -> list[UserFact]:
        return self._user.search(query)

    def get_user_profile(self, user_id: str) -> list[UserFact]:
        return self._user.get_user_profile(user_id)

    def delete_user_fact(self, fact_id: str) -> bool:
        return self._user.delete_fact(fact_id)

    # ── Generic Write (legacy compatibility) ──────────────

    async def write(self, entry: MemoryEntry, *, trace_id: str = "") -> str:
        """写入记忆（带去重检查），兼容旧 API。"""
        existing = await asyncio.to_thread(
            self._long_term.search,
            MemoryQuery(query_text=entry.content, top_k=3),
        )
        evaluation = self._write_policy.evaluate(entry, existing)

        if evaluation.decision.value == "skip":
            logger.debug("memory_write_skipped", id=entry.id, reason=evaluation.reason)
            return entry.id
        if evaluation.decision.value == "merge" and existing:
            merged = existing[0].entry
            merged.last_accessed_at = entry.created_at
            merged.access_count += 1
            merged.importance = max(merged.importance, entry.importance)
            return await asyncio.to_thread(self._long_term.store, merged)

        return await asyncio.to_thread(self._long_term.store, entry)

    async def read(self, entry_id: str, *, agent_id: str = "", trace_id: str = "") -> MemoryEntry | None:
        """按 ID 读取，兼容旧 API。"""
        results = await asyncio.to_thread(
            self._long_term.search,
            MemoryQuery(query_text="", agent_id=agent_id, top_k=1),
        )
        for r in results:
            if r.entry.id == entry_id:
                return r.entry
        return None

    async def search(self, query: MemoryQuery, *, trace_id: str = "") -> list[MemorySearchResult]:
        """语义搜索记忆，兼容旧 API。"""
        return await asyncio.to_thread(self._long_term.search, query)

    async def consolidate(
        self, *, agent_id: str, session_id: str, trace_id: str = ""
    ) -> list[str]:
        """整合工作记忆到长期记忆。"""
        results = await asyncio.to_thread(
            self._long_term.search,
            MemoryQuery(agent_id=agent_id, min_importance=0.6, top_k=20),
        )
        ids: list[str] = []
        for r in results:
            if r.entry.importance >= 0.6:
                ids.append(r.entry.id)
        logger.info("memory_consolidated", count=len(ids), session_id=session_id)
        return ids

    async def decay(self, *, trace_id: str = "") -> int:
        return await asyncio.to_thread(self._long_term.decay)

    async def share(
        self, entry_id: str, from_agent: str, to_agents: list[str], *, trace_id: str = ""
    ) -> bool:
        entry = await self.read(entry_id, agent_id=from_agent)
        if entry is None:
            return False
        for target in to_agents:
            shared = MemoryEntry(
                agent_id=target, content=entry.content, memory_type=entry.memory_type,
                tags=[*entry.tags, "shared", f"from:{from_agent}"],
                importance=entry.importance, source_trace_id=trace_id,
            )
            await self.write(shared, trace_id=trace_id)
        return True

    # ── ACL Matrix (跨 Agent 访问控制) ────────────────────

    def check_acl(
        self, memory_type: MemoryType, accessor_agent_id: str, owner_agent_id: str = "",
        *, task_id: str = "", is_same_task: bool = False, is_completed: bool = False,
    ) -> bool:
        """跨 agent 访问矩阵检查。

        返回 True 表示允许访问。
        """
        if memory_type == MemoryType.TASK_CONTEXT:
            # 上下文完全私有，只有自己可见
            return accessor_agent_id == owner_agent_id

        if memory_type == MemoryType.TASK_STATE:
            # 同任务 rw，非同任务 completed 后只读
            if accessor_agent_id == "main":
                return True
            if is_same_task:
                return True
            return is_completed

        if memory_type in (MemoryType.KNOWLEDGE, MemoryType.FACT):
            # 知识/事实全局 rw
            return True

        if memory_type == MemoryType.EXPERIENCE:
            # 自己 rw，main 可读全部，其他只读 shareable
            # 注意：此处只做第一层 ACL 判断，shareable 的细粒度过滤
            # 由 ExperienceMemory.search() 在检索层完成
            if accessor_agent_id == owner_agent_id:
                return True
            if accessor_agent_id == "main":
                return True
            return True  # 通过，具体过滤在查询层

        if memory_type == MemoryType.USER_FACT:
            # 全员 r，main 独占 w
            return True  # read 允许，write 由 UserMemory 内部控

        return False

    # ── Route ─────────────────────────────────────────────

    def route_query(self, query: str, *, agent_id: str = "") -> dict[str, Any]:
        """根据查询意图自动路由。返回路由计划。"""
        has_task = len(self._task.get_active_tasks(agent_id)) > 0 if agent_id else False
        plan = self._router.route(query, agent_id=agent_id, has_active_task=has_task)

        result: dict[str, Any] = {
            "intent": plan.query_intent,
            "token_budget_per_target": plan.token_budget,
            "targets": [
                {"type": t[0].value, "weight": t[1], "layer": t[2].value, "hints": t[3]}
                for t in plan.targets
            ],
        }

        # 自动执行路由（同步部分）
        if agent_id:
            # 用户画像
            for t in plan.targets:
                if t[0] == MemoryType.USER_FACT:
                    facts = self._user.search(UserMemoryQuery(user_id=agent_id, active_only=True))
                    result["user_profile"] = [
                        {"entity": f.entity, "predicate": f.predicate, "object": f.object}
                        for f in facts[:10]
                    ]
                    break

        return result

    # ── Monitoring ────────────────────────────────────────

    def get_metrics(self) -> MemoryMetrics:
        """获取当前监控指标。"""
        self._metrics.retrieval_latency_l1_ms = self._last_latency.get("L1", 0)
        self._metrics.retrieval_latency_l2_ms = self._last_latency.get("L2", 0)
        self._metrics.retrieval_latency_l3_ms = self._last_latency.get("L3", 0)

        # 总数统计
        total_mem = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        total_tasks = self._db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        total_exp = self._db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        total_uf = self._db.execute("SELECT COUNT(*) FROM user_facts WHERE valid_to IS NULL").fetchone()[0]
        pending_props = self._db.execute("SELECT COUNT(*) FROM memory_proposals WHERE status='pending'").fetchone()[0]

        self._metrics.total_memories = total_mem
        self._metrics.total_tasks = total_tasks
        self._metrics.total_experiences = total_exp
        self._metrics.total_user_facts = total_uf
        self._metrics.write_queue_length = pending_props

        # 幂等冲突率
        idemp_stats = self._task.get_idempotency_stats()
        total_records = idemp_stats.get("total_records", 1)
        conflicts = idemp_stats.get("conflicts", 0)
        self._metrics.idempotency_conflict_rate = conflicts / max(total_records, 1)

        return self._metrics

    def get_stats_summary(self) -> dict[str, Any]:
        """获取记忆系统统计摘要。"""
        m = self.get_metrics()
        return {
            "storage": {
                "total_memories": m.total_memories,
                "total_tasks": m.total_tasks,
                "total_experiences": m.total_experiences,
                "total_user_facts": m.total_user_facts,
            },
            "latency_ms": {
                "L1_context": m.retrieval_latency_l1_ms,
                "L2_task": m.retrieval_latency_l2_ms,
                "L3_knowledge": m.retrieval_latency_l3_ms,
            },
            "health": {
                "pending_proposals": m.write_queue_length,
                "idempotency_conflict_rate": m.idempotency_conflict_rate,
            },
            "limits": {
                "task_max_concurrent": self._task._settings.task_max_concurrent,
                "user_max_facts": self._user._settings.user_max_facts,
                "context_max_tokens": self._compressor.max_tokens,
            },
        }

    # ── Lifecycle ─────────────────────────────────────────

    def close(self) -> None:
        self._long_term.close()


# 模块级单例（双重检查锁）
_gateway: MemoryGateway | None = None
_gateway_lock = threading.Lock()


def get_gateway() -> MemoryGateway:
    """获取 MemoryGateway 单例。"""
    global _gateway
    if _gateway is None:
        with _gateway_lock:
            if _gateway is None:
                _gateway = MemoryGateway()
    return _gateway
