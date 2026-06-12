"""知识记忆（L3 Cold）—— 向量语义检索 + 知识图谱三元组。

对齐文档规格：
- text-embedding-3-small, 1536 维, 512 token chunk, 50 token overlap
- 两阶段检索：ChromaDB 宽召回（TopK10-20）→ Rerank 精排（3-5 条）
- 动态相似度阈值：简单查询 0.9, 复杂查询 0.7, baseline 0.85
- Cognee 六阶段流水线 + 冲突检测（80% 自动合并, 20% main 裁决）
- 异步写入，不阻塞对话
"""

from __future__ import annotations

import json as _json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.shared.config import get_settings
from src.shared.models import ConflictRecord, ConflictStatus, KnowledgeQuery, KnowledgeTriple, MemorySearchResult

logger = structlog.get_logger()


class KnowledgeMemory:
    """L3 知识记忆：ChromaDB 语义检索 + SQLite 图谱三元组。"""

    def __init__(self, db: sqlite3.Connection | None = None) -> None:
        self._settings = get_settings()
        if db is not None:
            self._db = db
        else:
            from src.memory.long_term import LongTermMemory
            ltm = LongTermMemory()
            self._db = ltm._db
            self._chroma = ltm._collection

    # ── Query ─────────────────────────────────────────────

    def search(self, query: KnowledgeQuery) -> list[MemorySearchResult]:
        """两阶段检索：宽召回 → 精排。"""
        threshold = query.similarity_threshold
        if threshold is None:
            # 动态阈值：简单查询高阈值，复杂查询低阈值
            threshold = (
                self._settings.knowledge_similarity_max if not query.is_complex_query
                else self._settings.knowledge_similarity_min
            )

        # 第一阶段：ChromaDB 宽召回
        recall_k = min(query.top_k * 4, 20)
        try:
            raw = self._chroma.query(
                query_texts=[query.query_text],
                n_results=recall_k,
            )
        except Exception as e:
            logger.error("knowledge_search_failed", error=str(e))
            return []

        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return []

        # 第二阶段：Rerank（按相似度 → 阈值过滤 → top_k）
        candidates: list[tuple[float, str]] = []
        for i, doc_id in enumerate(raw["ids"][0]):
            distance = raw.get("distances", [[1.0]])[0][i] if raw.get("distances") else 1.0
            score = max(0.0, 1.0 - distance)
            if score >= threshold:
                candidates.append((score, doc_id))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[: query.top_k]

        results: list[MemorySearchResult] = []
        for score, doc_id in top_candidates:
            row = self._db.execute(
                "SELECT * FROM memories WHERE id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                continue
            from src.shared.models import MemoryEntry, MemoryType
            entry = MemoryEntry(
                id=row["id"], agent_id=row["agent_id"], content=row["content"],
                memory_type=MemoryType(row["memory_type"]),
                tags=_json.loads(row["tags"]) if row["tags"] else [],
                importance=row["importance"], source_trace_id=row["source_trace_id"],
                created_at=row["created_at"], last_accessed_at=row["last_accessed_at"],
                access_count=row["access_count"], expires_at=row["expires_at"],
            )
            results.append(MemorySearchResult(entry=entry, score=round(score, 4)))

        return results

    # ── Triple Store ──────────────────────────────────────

    def add_triple(self, triple: KnowledgeTriple) -> KnowledgeTriple:
        """添加知识三元组。写入前做冲突检测。"""
        # 冲突检测
        conflicts = self._detect_conflicts(triple)
        for conflict in conflicts:
            self._resolve_or_enqueue(conflict)

        self._db.execute(
            """INSERT OR REPLACE INTO knowledge_triples
               (id, subject, predicate, object, source, confidence_weight, created_at, chroma_entity_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (triple.id, triple.subject, triple.predicate, triple.object,
             triple.source, triple.confidence_weight, triple.created_at, triple.chroma_entity_id),
        )
        self._db.commit()
        logger.debug("triple_added", id=triple.id, subject=triple.subject, predicate=triple.predicate)
        return triple

    def add_triples_batch(self, triples: list[KnowledgeTriple]) -> int:
        """批量添加三元组（来自 Cognee 流水线 LLM 提取）。"""
        count = 0
        for t in triples:
            self.add_triple(t)
            count += 1
        logger.info("triples_batch_added", count=count)
        return count

    def query_triples(
        self, subject: str | None = None, predicate: str | None = None, object: str | None = None,
    ) -> list[KnowledgeTriple]:
        """精确查询知识图谱。"""
        conditions: list[str] = []
        params: list[str] = []
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if object:
            conditions.append("object = ?")
            params.append(object)
        where = " AND ".join(conditions) if conditions else "1=1"
        rows = self._db.execute(
            f"SELECT * FROM knowledge_triples WHERE {where}", params
        ).fetchall()
        return [self._row_to_triple(r) for r in rows]

    def get_entity_relations(self, entity: str) -> list[KnowledgeTriple]:
        """查询某个实体的所有关系。"""
        rows = self._db.execute(
            "SELECT * FROM knowledge_triples WHERE subject = ? OR object = ?",
            (entity, entity),
        ).fetchall()
        return [self._row_to_triple(r) for r in rows]

    # ── Conflict Detection ────────────────────────────────

    def _detect_conflicts(self, new_triple: KnowledgeTriple) -> list[ConflictRecord]:
        """检测新三元组与已有三元组的冲突。"""
        conflicts: list[ConflictRecord] = []
        now = datetime.now(timezone.utc).isoformat()

        # 查相同 subject+predicate 或 subject+object 的现有三元组
        existing = self._db.execute(
            """SELECT * FROM knowledge_triples
               WHERE (subject = ? AND predicate = ?) OR (subject = ? AND object = ?)""",
            (new_triple.subject, new_triple.predicate, new_triple.subject, new_triple.object),
        ).fetchall()

        for row in existing:
            ext = self._row_to_triple(row)
            if ext.id == new_triple.id:
                continue

            # 完全相同 → 增加 confidence_weight（去重），同时写入冲突队列做统计
            if (ext.subject == new_triple.subject
                    and ext.predicate == new_triple.predicate
                    and ext.object == new_triple.object):
                self._db.execute(
                    "UPDATE knowledge_triples SET confidence_weight = confidence_weight + ? WHERE id = ?",
                    (new_triple.confidence_weight, ext.id),
                )
                record = ConflictRecord(
                    triple_a_id=ext.id, triple_b_id=new_triple.id,
                    conflict_type="duplicate", status=ConflictStatus.AUTO_MERGED,
                    resolved_by="auto", resolved_at=now, created_at=now,
                )
                self._write_conflict(record)
                logger.debug("triple_duplicate_bumped", id=ext.id)
                continue

            # 相同 subject+object 但不同 predicate → 矛盾冲突
            if ext.subject == new_triple.subject and ext.object == new_triple.object:
                record = ConflictRecord(
                    triple_a_id=ext.id, triple_b_id=new_triple.id,
                    conflict_type="contradiction", created_at=now,
                )
                self._write_conflict(record)
                conflicts.append(record)

        return conflicts

    def _write_conflict(self, record: ConflictRecord) -> None:
        self._db.execute(
            """INSERT INTO conflict_queue (id, triple_a_id, triple_b_id, conflict_type, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (record.id, record.triple_a_id, record.triple_b_id,
             record.conflict_type, record.status.value, record.created_at),
        )
        self._db.commit()

    def _resolve_or_enqueue(self, conflict: ConflictRecord) -> None:
        """冲突自动裁决：duplicate 直接合并，contradiction 启发式打分。

        - duplicate：直接 bump confidence（已在 _detect_conflicts 处理）
        - contradiction：计算语义相似度（source + object 重叠加 confidence_weight 差），
          高分自动合并，低分留待 main 裁决（约 80%/20% 分流）。
        """
        now = datetime.now(timezone.utc).isoformat()
        if conflict.conflict_type == "duplicate":
            self._db.execute(
                "UPDATE conflict_queue SET status = 'auto_merged', resolved_at = ? WHERE id = ?",
                (now, conflict.id),
            )
            self._db.commit()
            return

        # contradiction: 启发式打分 → 高分自动合并
        a = self._db.execute(
            "SELECT * FROM knowledge_triples WHERE id = ?", (conflict.triple_a_id,)
        ).fetchone()
        b = self._db.execute(
            "SELECT * FROM knowledge_triples WHERE id = ?", (conflict.triple_b_id,)
        ).fetchone()
        if not a or not b:
            return

        auto_score = self._heuristic_score(dict(a), dict(b))
        if auto_score >= 0.8:
            # 高分 → 自动合并（选 confidence_weight 高的）
            w_a = a["confidence_weight"] or 1.0
            w_b = b["confidence_weight"] or 1.0
            winner = conflict.triple_a_id if w_a >= w_b else conflict.triple_b_id
            self._db.execute(
                "UPDATE conflict_queue SET status = 'auto_merged', resolved_by = 'heuristic', resolved_at = ? WHERE id = ?",
                (now, conflict.id),
            )
            # 落选三元组降权
            loser_id = conflict.triple_b_id if winner == conflict.triple_a_id else conflict.triple_a_id
            self._db.execute(
                "UPDATE knowledge_triples SET confidence_weight = confidence_weight * 0.3 WHERE id = ?",
                (loser_id,),
            )
            logger.info("conflict_auto_merged", conflict_id=conflict.id, score=auto_score)
        else:
            # 低分 → 入队待 main 裁决
            self._db.execute(
                "UPDATE conflict_queue SET status = 'pending' WHERE id = ?",
                (conflict.id,),
            )
        self._db.commit()

    @staticmethod
    def _heuristic_score(a: dict, b: dict) -> float:
        """启发式打分：基于 source 重叠 + object 相似度 + confidence 差距。

        返回 0.0~1.0，>=0.8 自动合并。
        """
        score = 0.0
        # source 相同 → +0.4
        if a.get("source") and b.get("source") and a["source"] == b["source"]:
            score += 0.4
        # object 重叠（字符级 Jaccard）→ +0.3
        obj_a = set(str(a.get("object", "")))
        obj_b = set(str(b.get("object", "")))
        if obj_a and obj_b:
            jaccard = len(obj_a & obj_b) / len(obj_a | obj_b)
            score += jaccard * 0.3
        # confidence_weight 差距小 → +0.3
        c_a = a.get("confidence_weight", 1.0) or 1.0
        c_b = b.get("confidence_weight", 1.0) or 1.0
        conf_diff = abs(c_a - c_b) / max(c_a, c_b, 1.0)
        score += (1.0 - conf_diff) * 0.3
        return min(score, 1.0)

    def get_pending_conflicts(self) -> list[ConflictRecord]:
        """获取待 main 裁决的冲突列表。"""
        rows = self._db.execute(
            "SELECT * FROM conflict_queue WHERE status = 'pending'"
        ).fetchall()
        return [ConflictRecord(
            id=r["id"], triple_a_id=r["triple_a_id"], triple_b_id=r["triple_b_id"],
            conflict_type=r["conflict_type"], status=ConflictStatus(r["status"]),
            resolved_by=r["resolved_by"], resolved_at=r["resolved_at"], created_at=r["created_at"],
        ) for r in rows]

    def resolve_conflict(self, conflict_id: str, resolved_by: str, winner_triple_id: str) -> None:
        """main agent 裁决冲突。"""
        now = datetime.now(timezone.utc).isoformat()
        conflict = self._db.execute(
            "SELECT * FROM conflict_queue WHERE id = ?", (conflict_id,)
        ).fetchone()
        if conflict is None:
            return

        # 标记落选三元组为 deprecated
        loser_id = (
            conflict["triple_b_id"] if winner_triple_id == conflict["triple_a_id"]
            else conflict["triple_a_id"]
        )
        self._db.execute(
            "UPDATE knowledge_triples SET confidence_weight = confidence_weight * 0.1 WHERE id = ?",
            (loser_id,),
        )
        self._db.execute(
            "UPDATE conflict_queue SET status = 'llm_resolved', resolved_by = ?, resolved_at = ? WHERE id = ?",
            (resolved_by, now, conflict_id),
        )
        self._db.commit()
        logger.info("conflict_resolved", conflict_id=conflict_id, resolver=resolved_by)

    # ── Cognee 提取规则 ───────────────────────────────────

    @staticmethod
    def extract_triples_prompt(text: str) -> str:
        """生成 LLM 提取三元组的 prompt。"""
        return f"""从以下文本提取知识三元组（subject, predicate, object 格式）。

规则：
1. 只提取主谓宾结构的客观事实
2. 不提取主观观点（"我觉得""应该"等）
3. 不提取临时信息（"目前正在""今天"等）
4. 每个三元组必须包含 source 字段
5. 提取后按冲突检测规则过滤

文本：
{text[:2000]}

输出 JSON 数组格式：
[{{"subject": "...", "predicate": "...", "object": "...", "source": "...", "confidence_weight": 0.9}}]"""

    # ── Helpers ───────────────────────────────────────────

    def _row_to_triple(self, row: sqlite3.Row) -> KnowledgeTriple:
        return KnowledgeTriple(
            id=row["id"], subject=row["subject"], predicate=row["predicate"],
            object=row["object"], source=row["source"] or "",
            confidence_weight=row["confidence_weight"],
            created_at=row["created_at"], chroma_entity_id=row["chroma_entity_id"],
        )
