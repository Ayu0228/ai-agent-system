"""经验记忆（L3 Cold）—— 时间衰减 + 自进化双轨。

对齐文档规格：
- 30 天半衰期，7 天 recency boost ×1.5，success bonus ×1.3
- 检索 Top3，默认私有，shareable 阈值 0.9（复用≥3 次 且 成功率≥90%）
- 经验卡片生成：任务 done 后异步 LLM prompt（~500 token）
- 每周 memify：合并相似事件、删除低权重、标记高复用
- 自进化双轨：self-improving(P1) + AutoScale(P2)
"""

from __future__ import annotations

import json as _json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from src.shared.config import get_settings
from src.shared.models import ExperienceCard, ExperienceQuery, SelfImprovingTrack

logger = structlog.get_logger()

# 经验卡片 LLM 提取 prompt（~500 tokens）
EXPERIENCE_EXTRACT_PROMPT = """你是一个 Agent 经验分析师。基于以下任务的执行记录，生成一条经验卡片。

输出 JSON 格式（不要其他文字）：
{
  "scenario": "一句话描述任务场景",
  "approach": "采用了什么方法/步骤",
  "result": "执行结果（成功/失败/部分成功）",
  "lesson": "核心教训或最佳实践",
  "tags": ["标签1", "标签2"],
  "is_success": true
}

任务记录：
"""


class ExperienceMemory:
    """L3 经验记忆管理器。操作 SQLite experiences 表 + ChromaDB。"""

    def __init__(self, db: sqlite3.Connection | None = None) -> None:
        self._settings = get_settings()
        if db is not None:
            self._db = db
            self._chroma = None
        else:
            from src.memory.long_term import LongTermMemory
            ltm = LongTermMemory()
            self._db = ltm._db
            self._chroma = ltm._collection

    # ── Experience CRUD ───────────────────────────────────

    def store(self, card: ExperienceCard) -> str:
        """写入经验卡片。"""
        now = datetime.now(timezone.utc).isoformat()
        if not card.experience_id:
            card.experience_id = f"exp_{uuid.uuid4().hex[:12]}"

        self._db.execute(
            """INSERT OR REPLACE INTO experiences
               (experience_id, owner_agent_id, scenario, approach, result, lesson,
                tags, weight, shareable, usage_count, success_rate,
                self_improving_track, autoscale_eligible, created_at, last_accessed_at, task_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (card.experience_id, card.owner_agent_id, card.scenario, card.approach,
             card.result, card.lesson, _json.dumps(card.tags), card.weight,
             int(card.shareable), card.usage_count, card.success_rate,
             card.self_improving_track.value if isinstance(card.self_improving_track, SelfImprovingTrack) else card.self_improving_track,
             int(card.autoscale_eligible), card.created_at or now, now, card.task_id),
        )

        # 同步到 ChromaDB
        if self._chroma:
            try:
                existing = self._chroma.get(ids=[card.experience_id])
                doc = f"{card.scenario}\n{card.approach}\n{card.lesson}"
                if existing and existing["ids"]:
                    self._chroma.update(ids=[card.experience_id], documents=[doc])
                else:
                    self._chroma.add(ids=[card.experience_id], documents=[doc],
                                     metadatas=[{"type": "experience", "agent": card.owner_agent_id}])
            except Exception as e:
                logger.warning("experience_chroma_sync_failed", error=str(e))

        self._db.commit()
        logger.debug("experience_stored", id=card.experience_id, agent=card.owner_agent_id)
        return card.experience_id

    def get(self, experience_id: str) -> ExperienceCard | None:
        """按 ID 获取经验卡片。"""
        row = self._db.execute(
            "SELECT * FROM experiences WHERE experience_id = ?", (experience_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_card(row)

    def search(self, query: ExperienceQuery) -> list[ExperienceCard]:
        """检索经验：向量搜索 → 按 weight×recency_boost 排序 → TopK。"""
        results: list[tuple[float, ExperienceCard]] = []

        if self._chroma and query.query_text:
            try:
                raw = self._chroma.query(query_texts=[query.query_text], n_results=10)
                if raw and raw.get("ids") and raw["ids"][0]:
                    for i, doc_id in enumerate(raw["ids"][0]):
                        row = self._db.execute(
                            "SELECT * FROM experiences WHERE experience_id = ?", (doc_id,)
                        ).fetchone()
                        if row is None:
                            continue
                        card = self._row_to_card(row)

                        # ACL 过滤
                        if query.agent_id:
                            if card.owner_agent_id != query.agent_id and not card.shareable:
                                continue
                        if not query.include_shared and card.shareable:
                            continue
                        if query.is_success is not None:
                            is_ok = card.success_rate >= 0.7
                            if is_ok != query.is_success:
                                continue

                        # 加权排序
                        distance = raw.get("distances", [[1.0]])[0][i] if raw.get("distances") else 1.0
                        similarity = max(0.0, 1.0 - distance)
                        effective_weight = self._calc_weight(card)
                        score = similarity * 0.6 + effective_weight * 0.4
                        results.append((score, card))
            except Exception as e:
                logger.error("experience_search_failed", error=str(e))
                return self._keyword_fallback(query)

        # 排序
        results.sort(key=lambda x: x[0], reverse=True)
        return [card for _, card in results[: query.top_k]]

    # ── Weight Calculation ────────────────────────────────

    def _calc_weight(self, card: ExperienceCard) -> float:
        """计算经验当前有效权重。"""
        now = datetime.now(timezone.utc)
        try:
            created = datetime.fromisoformat(card.created_at)
        except (ValueError, TypeError):
            created = now - timedelta(days=30)

        days_ago = (now - created).days
        half_life = self._settings.experience_decay_half_life_days

        # 时间衰减
        decay = 0.5 ** (days_ago / half_life)
        # Recency boost
        recency = 1.5 if days_ago <= self._settings.experience_recency_boost_days else 1.0
        # Success bonus
        success = self._settings.experience_success_bonus if card.success_rate >= 0.7 else 1.0

        weight = decay * recency * success
        return min(weight, 2.0)

    def update_usage(self, experience_id: str) -> None:
        """复用经验：usage_count+1，更新 last_accessed_at。"""
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            """UPDATE experiences SET usage_count = usage_count + 1,
               last_accessed_at = ? WHERE experience_id = ?""",
            (now, experience_id),
        )
        self._db.commit()

        # 检查是否满足 AutoScale 条件
        card = self.get(experience_id)
        if card and card.usage_count >= 3 and card.success_rate >= 0.9:
            self._db.execute(
                "UPDATE experiences SET autoscale_eligible = 1 WHERE experience_id = ?",
                (experience_id,),
            )
            self._db.commit()

    # ── Share ─────────────────────────────────────────────

    def mark_shareable(self, experience_id: str) -> bool:
        """标记经验为可共享（需满足阈值条件）。"""
        card = self.get(experience_id)
        if card is None:
            return False
        if card.usage_count < 3 or card.success_rate < self._settings.experience_shareable_threshold:
            return False
        # 满足阈值
        self._db.execute(
            "UPDATE experiences SET shareable = 1 WHERE experience_id = ?",
            (experience_id,),
        )
        self._db.commit()
        logger.info("experience_shared", id=experience_id, usage=card.usage_count, success=card.success_rate)
        return True

    def get_shared(self, limit: int = 10) -> list[ExperienceCard]:
        """获取全局共享经验。"""
        rows = self._db.execute(
            "SELECT * FROM experiences WHERE shareable = 1 ORDER BY weight DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_card(r) for r in rows]

    # ── Memify ────────────────────────────────────────────

    def memify(self, *, weight_threshold: float = 0.2) -> dict[str, int]:
        """每周清理：删除低权重、合并相似事件、标记高复用。

        返回 {deleted: N, merged: N, promoted: N}。批量操作，避免 N+1 查询。
        """
        stats = {"deleted": 0, "merged": 0, "promoted": 0}
        now = datetime.now(timezone.utc).isoformat()

        # 一次性 JOIN 查询所有字段，批量删除低权重经验
        rows = self._db.execute(
            "SELECT * FROM experiences"
        ).fetchall()
        to_delete: list[str] = []
        for row in rows:
            card = self._row_to_card(row)
            effective = self._calc_weight(card)
            if effective < weight_threshold:
                to_delete.append(row["experience_id"])

        if to_delete:
            placeholders = ",".join("?" for _ in to_delete)
            self._db.execute(
                f"DELETE FROM experiences WHERE experience_id IN ({placeholders})",
                to_delete,
            )
            if self._chroma:
                try:
                    self._chroma.delete(ids=to_delete)
                except Exception:
                    pass
            stats["deleted"] = len(to_delete)

        # 批量标记高复用为 shareable
        shareable_rows = self._db.execute(
            """SELECT experience_id FROM experiences
               WHERE shareable = 0 AND usage_count >= 3 AND success_rate >= ?""",
            (self._settings.experience_shareable_threshold,),
        ).fetchall()
        if shareable_rows:
            s_ids = [r["experience_id"] for r in shareable_rows]
            placeholders = ",".join("?" for _ in s_ids)
            self._db.execute(
                f"UPDATE experiences SET shareable = 1 WHERE experience_id IN ({placeholders})",
                s_ids,
            )
            stats["promoted"] += len(s_ids)

        # 批量检查 autoscale_eligible
        eligible = self._db.execute(
            """SELECT experience_id FROM experiences
               WHERE self_improving_track = 'self-improving'
               AND usage_count >= 3 AND success_rate >= 0.9 AND autoscale_eligible = 0"""
        ).fetchall()
        if eligible:
            e_ids = [r["experience_id"] for r in eligible]
            placeholders = ",".join("?" for _ in e_ids)
            self._db.execute(
                f"UPDATE experiences SET autoscale_eligible = 1 WHERE experience_id IN ({placeholders})",
                e_ids,
            )
            stats["promoted"] += len(e_ids)

        self._db.commit()
        logger.info("memify_complete", **stats)
        return stats

    # ── Experience Card Generation ────────────────────────

    @staticmethod
    def get_extract_prompt(task_goal: str, steps: str, result: str) -> str:
        """生成经验卡片提取 prompt。"""
        return f"""{EXPERIENCE_EXTRACT_PROMPT}

目标：{task_goal}
执行步骤：{steps}
最终结果：{result}
"""

    # ── Keyword Fallback ──────────────────────────────────

    def _keyword_fallback(self, query: ExperienceQuery) -> list[ExperienceCard]:
        words = query.query_text.split()
        if not words:
            return []
        clauses = " OR ".join(["scenario LIKE ? OR lesson LIKE ?" for _ in words])
        params: list[Any] = []
        for w in words:
            params.extend([f"%{w}%", f"%{w}%"])
        if query.agent_id:
            clauses = f"({clauses}) AND (owner_agent_id = ? OR shareable = 1)"
            params.append(query.agent_id)

        rows = self._db.execute(
            f"SELECT * FROM experiences WHERE {clauses} ORDER BY weight DESC LIMIT ?",
            [*params, query.top_k],
        ).fetchall()
        return [self._row_to_card(r) for r in rows]

    # ── Helpers ───────────────────────────────────────────

    def _row_to_card(self, row: sqlite3.Row) -> ExperienceCard:
        return ExperienceCard(
            experience_id=row["experience_id"],
            owner_agent_id=row["owner_agent_id"],
            scenario=row["scenario"],
            approach=row["approach"],
            result=row["result"],
            lesson=row["lesson"],
            tags=_json.loads(row["tags"]) if row["tags"] else [],
            weight=row["weight"],
            shareable=bool(row["shareable"]),
            usage_count=row["usage_count"],
            success_rate=row["success_rate"],
            self_improving_track=SelfImprovingTrack(row["self_improving_track"]),
            autoscale_eligible=bool(row["autoscale_eligible"]),
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            task_id=row["task_id"],
        )
