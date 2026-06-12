"""记忆生命周期管理 —— 分层衰减、归档、清理。

对齐文档规格：
- 上下文：会话结束丢弃
- 任务：72h 后归档到经验记忆
- 知识：永久，冲突合并
- 经验：30 天衰减 → 90 天归档 → 低权重删除
- 用户：永久，LRU+低置信度淘汰
- 幂等记录：7 天后归档
"""

from __future__ import annotations

import json as _json
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from src.memory.experience_memory import ExperienceMemory
from src.shared.config import get_settings

logger = structlog.get_logger()

# 经验卡片 JSON 提取 prompt（用于从 LLM 响应中解析）
_EXTRACT_SYSTEM_PROMPT = (
    "你是一个 Agent 经验分析师。请严格输出 JSON，不要包含其他文字。"
)


class MemoryLifecycle:
    """分层生命周期管理器。定期运行维护任务。"""

    def __init__(self, gateway) -> None:
        self._gw = gateway  # MemoryGateway instance
        self._settings = get_settings()

    # ── Full Maintenance Cycle ────────────────────────────

    async def run_maintenance(self) -> dict[str, Any]:
        """执行完整维护周期（建议每 24h 运行一次）。"""
        report: dict[str, Any] = {}

        # 1. 任务 TTL 检查与归档（含 LLM 经验卡片生成）
        archived = await self.archive_tasks()
        report["tasks_archived"] = len(archived)

        # 2. 经验衰减与 memify
        memify_result = self._gw.memify()
        report["experience"] = memify_result

        # 3. 知识冲突清理
        conflicts = self.clean_conflicts()
        report["conflicts_resolved"] = conflicts

        # 4. 幂等记录归档
        idemp_archived = self.archive_idempotency()
        report["idempotency_archived"] = idemp_archived

        # 5. 用户提案降级检查
        degraded = self._gw.degrade_stale_proposals()
        report["proposals_degraded"] = degraded

        # 6. 通用记忆衰减
        decayed = self._gw._long_term.decay()
        report["memories_decayed"] = decayed

        logger.info("maintenance_cycle_complete", **report)
        return report

    # ── Task Archival ─────────────────────────────────────

    async def archive_tasks(self) -> list[str]:
        """归档到期任务 → 调用 LLM 生成经验卡片并写入 ExperienceMemory。"""
        archived = self._gw.check_task_ttl()
        settings = get_settings()

        for task_id in archived:
            task = self._gw.get_task(task_id)
            if task is None:
                continue

            prompt = ExperienceMemory.get_extract_prompt(
                task_goal=f"Task {task_id}",
                steps=_json.dumps(task.artifacts),
                result=task.status.value if hasattr(task.status, 'value') else str(task.status),
            )

            # 调用 LLM 提取经验卡片
            if settings.llm_api_key:
                try:
                    from src.shared.llm import get_llm
                    llm = get_llm()
                    response = await llm.chat(
                        messages=[
                            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=1024,
                    )
                    card_data = _json.loads(response)
                    from src.shared.models import ExperienceCard
                    card = ExperienceCard(
                        owner_agent_id=task.owner_agent_id,
                        scenario=card_data.get("scenario", ""),
                        approach=card_data.get("approach", ""),
                        result=card_data.get("result", ""),
                        lesson=card_data.get("lesson", ""),
                        tags=card_data.get("tags", []),
                        task_id=task_id,
                    )
                    if card_data.get("is_success"):
                        card.success_rate = 0.9
                    self._gw.store_experience(card)
                    logger.info("experience_card_generated", task_id=task_id, exp_id=card.experience_id)
                except Exception as e:
                    logger.warning("experience_extraction_failed", task_id=task_id, error=str(e))
            else:
                logger.debug("task_archive_skipped_llm", task_id=task_id, reason="no_api_key")

        return archived

    # ── Conflict Cleanup ──────────────────────────────────

    def clean_conflicts(self) -> int:
        """自动裁决冲突：duplicate 自动合并，contradiction 启发式选置信度高的。
        胜负不明（confidence_weight 相同）才留待 LLM 裁决。"""
        pending = self._gw.get_pending_conflicts()
        resolved = 0

        for conflict in pending:
            if conflict.conflict_type == "duplicate":
                self._gw.resolve_conflict(conflict.id, "auto", conflict.triple_a_id)
                resolved += 1
            elif conflict.conflict_type == "contradiction":
                # 启发式裁决：选 confidence_weight 高的
                a = self._gw._knowledge._db.execute(
                    "SELECT * FROM knowledge_triples WHERE id = ?", (conflict.triple_a_id,)
                ).fetchone()
                b = self._gw._knowledge._db.execute(
                    "SELECT * FROM knowledge_triples WHERE id = ?", (conflict.triple_b_id,)
                ).fetchone()
                if a and b:
                    w_a = a["confidence_weight"] if a["confidence_weight"] else 1.0
                    w_b = b["confidence_weight"] if b["confidence_weight"] else 1.0
                    if w_a != w_b:
                        winner = a["id"] if w_a > w_b else b["id"]
                        self._gw.resolve_conflict(conflict.id, "auto_heuristic", winner)
                        resolved += 1
                    # confidence_weight 相同 → 跳过，留待 LLM 裁决

        return resolved

    # ── Idempotency Archival ──────────────────────────────

    def archive_idempotency(self) -> int:
        """归档 7 天前的幂等记录。"""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._settings.idempotency_archive_days)).isoformat()
        db = self._gw._db

        # 先确认无活跃的重放请求
        active = db.execute(
            "SELECT COUNT(*) FROM agent_idempotency WHERE created_at >= ? AND status = 'processing'",
            (cutoff,),
        ).fetchone()[0]

        if active > 0:
            logger.warning("idempotency_archive_blocked", active_count=active)
            return 0

        cursor = db.execute("DELETE FROM agent_idempotency WHERE created_at < ?", (cutoff,))
        db.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info("idempotency_archived", count=deleted)
        return deleted

    # ── Health Check ──────────────────────────────────────

    def health_check(self) -> dict[str, Any]:
        """系统健康检查。"""
        metrics = self._gw.get_metrics()
        alerts: list[str] = []

        # 写入积压检查
        if metrics.write_queue_length > 100:
            alerts.append(f"写入积压: {metrics.write_queue_length} > 100")

        # 延迟检查
        if metrics.retrieval_latency_l3_ms > 2000:
            alerts.append(f"L3 检索延迟: {metrics.retrieval_latency_l3_ms:.0f}ms > 2000ms")
        if metrics.retrieval_latency_l2_ms > 500:
            alerts.append(f"L2 检索延迟: {metrics.retrieval_latency_l2_ms:.0f}ms > 500ms")

        # 幂等冲突率检查：重复请求过半才告警（正常系统大部分请求唯一，冲突率接近 0）
        if metrics.idempotency_conflict_rate > 0.5:
            alerts.append(f"幂等冲突率异常偏高: {metrics.idempotency_conflict_rate:.1%} > 50%")

        # 存储空间检查（简化版）
        storage_pct = metrics.storage_usage_pct
        if storage_pct > 0.8:
            alerts.append(f"存储使用率: {storage_pct:.1%} > 80%")

        # 任务上限
        if metrics.total_tasks > 10:
            alerts.append(f"活跃任务: {metrics.total_tasks} > 10 上限")

        return {
            "healthy": len(alerts) == 0,
            "alerts": alerts,
            "metrics": {
                "write_queue": metrics.write_queue_length,
                "l1_latency_ms": metrics.retrieval_latency_l1_ms,
                "l2_latency_ms": metrics.retrieval_latency_l2_ms,
                "l3_latency_ms": metrics.retrieval_latency_l3_ms,
                "idempotency_rate": metrics.idempotency_conflict_rate,
            },
        }
