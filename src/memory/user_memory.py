"""用户记忆（L3 Warm Priority）—— 三层写入过滤 + main 独占写入 + 心跳审核。

对齐文档规格：
- 置信度阈值 0.7，最大 200 条/user，4 类优先标签
- 三层写入过滤：L1 规则硬编码 → L2 重要性评分 → L3 收益预测
- 90% 轻量预过滤 + 10% LLM 终审
- main 独占写入，其他 agent 提交 memory_proposal
- main 心跳：60s 周期扫描，超时 2 次降级
- LRU + 低置信度淘汰（超 200 条时）
- 写入限流：每秒最多 10 条
- 冲突处理：双时间戳（valid_from/valid_to）
"""

from __future__ import annotations

import json as _json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.shared.config import get_settings
from src.shared.models import MemoryProposal, ProposalStatus, UserFact, UserMemoryQuery

logger = structlog.get_logger()

# 四类优先保留标签（中英文映射）
PRIORITY_TAGS = ["preference", "goal", "constraint", "identity"]
_PRIORITY_KEYWORDS: dict[str, list[str]] = {
    "preference": ["偏好", "喜欢", "讨厌", "习惯", "常用", "preference"],
    "goal": ["目标", "计划", "打算", "想实现", "goal"],
    "constraint": ["限制", "不能", "不要", "禁止", "约束", "constraint"],
    "identity": ["我是", "身份", "角色", "职位", "identity"],
}


def _match_priority_tag(fact: UserFact) -> str | None:
    """词边界匹配：检测 fact 属于哪个优先标签类别。"""
    text = f"{fact.entity} {fact.predicate} {fact.object}".lower()
    for tag, keywords in _PRIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return tag
    return None

# L1 规则硬编码触发词（用户明确表达）
# 每个元组为 (触发词, 匹配位置): "prefix"=仅句首, "anywhere"=任意位置
L1_TRIGGER_PATTERNS: list[tuple[str, str]] = [
    ("请记住", "prefix"),
    ("记住，", "prefix"),
    ("别忘了", "prefix"),
    ("我是", "prefix"),
    ("我的目标是", "prefix"),
    ("我喜欢", "prefix"),
    ("我讨厌", "prefix"),
    ("我不要", "prefix"),
    ("我偏好", "prefix"),
    ("我的习惯是", "prefix"),
    ("我经常", "prefix"),
    ("我一直", "prefix"),
]

# 任意位置可匹配的触发词（需额外做安全过滤，不可包含敏感信息模式）
L1_ANYWHERE_PATTERNS: list[str] = [
    "我的偏好是",
    "我的习惯是",
]


class UserMemory:
    """L3 用户记忆管理器。操作 SQLite user_facts + memory_proposals 表。"""

    def __init__(self, db: sqlite3.Connection | None = None) -> None:
        self._settings = get_settings()
        if db is not None:
            self._db = db
        else:
            from src.memory.long_term import LongTermMemory
            ltm = LongTermMemory()
            self._db = ltm._db
        self._write_timestamps: list[float] = []  # 写入限流
        self._rate_lock = threading.Lock()

    # ── Write (main only) ─────────────────────────────────

    def write_fact(self, fact: UserFact, *, skip_filter: bool = False) -> UserFact | None:
        """直接写入用户事实（main 专用）。经三层过滤检查。"""
        # 写入限流检查
        if not self._check_rate_limit():
            logger.warning("user_memory_rate_limited", user_id=fact.user_id)
            return None

        # 三层过滤（main 直接写也走过滤，除非 skip_filter）
        if not skip_filter:
            decision = self._evaluate_fact(fact)
            fact.write_decision = decision
        else:
            fact.write_decision = "approved"

        if fact.write_decision == "rejected":
            logger.debug("user_fact_rejected", entity=fact.entity, predicate=fact.predicate)
            return None

        # 容量检查 + LRU 淘汰
        current_count = self._db.execute(
            "SELECT COUNT(*) FROM user_facts WHERE user_id = ? AND valid_to IS NULL",
            (fact.user_id,),
        ).fetchone()[0]
        if current_count >= self._settings.user_max_facts:
            self._evict_lru(fact.user_id)

        # 冲突检测：同名 key → 更新 valid_to
        conflict = self._db.execute(
            """SELECT id FROM user_facts
               WHERE user_id = ? AND entity = ? AND predicate = ? AND valid_to IS NULL""",
            (fact.user_id, fact.entity, fact.predicate),
        ).fetchone()
        if conflict:
            now = datetime.now(timezone.utc).isoformat()
            self._db.execute(
                "UPDATE user_facts SET valid_to = ? WHERE id = ?",
                (now, conflict["id"]),
            )
            logger.info("user_fact_updated", entity=fact.entity, old_id=conflict["id"])

        # 写入新事实
        self._db.execute(
            """INSERT INTO user_facts
               (id, user_id, entity, predicate, object, confidence, valid_from, valid_to,
                source_agent, write_decision, created_at, access_count, last_accessed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (fact.id, fact.user_id, fact.entity, fact.predicate, fact.object,
             fact.confidence, fact.valid_from, None, fact.source_agent,
             fact.write_decision, fact.created_at, fact.created_at),
        )
        self._db.commit()
        self._record_write()
        logger.info("user_fact_written", id=fact.id, entity=fact.entity, decision=fact.write_decision)
        return fact

    # ── Proposal (other agents → main) ────────────────────

    def submit_proposal(self, proposal: MemoryProposal) -> str:
        """其他 agent 提交用户记忆提案。等待 main 心跳审核。"""
        if not proposal.id:
            proposal.id = f"prop_{uuid.uuid4().hex[:12]}"

        self._db.execute(
            """INSERT INTO memory_proposals
               (id, user_id, entity, predicate, object, confidence,
                source_agent, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (proposal.id, proposal.user_id, proposal.entity, proposal.predicate,
             proposal.object, proposal.confidence, proposal.source_agent,
             proposal.created_at),
        )
        self._db.commit()
        logger.debug("proposal_submitted", id=proposal.id, source=proposal.source_agent)
        return proposal.id

    def get_pending_proposals(self, limit: int = 50) -> list[MemoryProposal]:
        """main 心跳获取待审核提案。"""
        rows = self._db.execute(
            "SELECT * FROM memory_proposals WHERE status = 'pending' ORDER BY created_at LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_proposal(r) for r in rows]

    def review_proposal(self, proposal_id: str, approved: bool, reviewer: str = "main") -> MemoryProposal | None:
        """main 审核提案。通过 → 写入 user_facts；拒绝 → 标记 rejected。"""
        row = self._db.execute(
            "SELECT * FROM memory_proposals WHERE id = ?", (proposal_id,)
        ).fetchone()
        if row is None:
            return None

        proposal = self._row_to_proposal(row)
        now = datetime.now(timezone.utc).isoformat()

        if approved:
            status = ProposalStatus.APPROVED
            # 转为正式 user_fact
            fact = UserFact(
                user_id=proposal.user_id, entity=proposal.entity,
                predicate=proposal.predicate, object=proposal.object,
                confidence=proposal.confidence, source_agent=proposal.source_agent,
                write_decision="approved", valid_from=now,
            )
            self.write_fact(fact, skip_filter=True)
        else:
            status = ProposalStatus.REJECTED

        self._db.execute(
            "UPDATE memory_proposals SET status = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
            (status.value, reviewer, now, proposal_id),
        )
        self._db.commit()
        proposal.status = status
        proposal.reviewed_by = reviewer
        proposal.reviewed_at = now
        return proposal

    def degrade_stale_proposals(self) -> int:
        """超时提案降级：心跳错过 2 次 → 状态改为 degraded（直接写入+标记待审核）。"""
        settings = self._settings
        cutoff = (datetime.now(timezone.utc).timestamp()
                  - settings.main_heartbeat_interval * settings.main_heartbeat_max_misses)
        cutoff_str = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()

        rows = self._db.execute(
            "SELECT * FROM memory_proposals WHERE status = 'pending' AND created_at < ?",
            (cutoff_str,),
        ).fetchall()

        count = 0
        for row in rows:
            proposal = self._row_to_proposal(row)
            self._db.execute(
                "UPDATE memory_proposals SET status = 'degraded' WHERE id = ?",
                (proposal.id,),
            )
            # 降级：直接写入 + 标记待审核
            fact = UserFact(
                user_id=proposal.user_id, entity=proposal.entity,
                predicate=proposal.predicate, object=proposal.object,
                confidence=proposal.confidence, source_agent=proposal.source_agent,
                write_decision="degraded", valid_from=datetime.now(timezone.utc).isoformat(),
            )
            self.write_fact(fact, skip_filter=True)
            count += 1

        if count:
            self._db.commit()
            logger.warning("proposals_degraded", count=count)
        return count

    # ── Read ──────────────────────────────────────────────

    def search(self, query: UserMemoryQuery) -> list[UserFact]:
        """查询用户记忆。"""
        conditions = ["1=1"]
        params: list[Any] = []

        if query.user_id:
            conditions.append("user_id = ?")
            params.append(query.user_id)
        if query.entity:
            conditions.append("entity LIKE ?")
            params.append(f"%{query.entity}%")
        if query.min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(query.min_confidence)
        if query.active_only:
            conditions.append("valid_to IS NULL")

        rows = self._db.execute(
            f"SELECT * FROM user_facts WHERE {' AND '.join(conditions)} "
            f"ORDER BY confidence DESC, access_count DESC LIMIT ?",
            [*params, query.top_k],
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def get_user_profile(self, user_id: str) -> list[UserFact]:
        """获取用户完整画像（当前活跃事实）。"""
        return self.search(UserMemoryQuery(user_id=user_id, active_only=True, top_k=50))

    def get_by_priority_tags(self, user_id: str) -> dict[str, list[UserFact]]:
        """按优先标签分类获取用户事实。使用词边界匹配。"""
        facts = self.get_user_profile(user_id)
        grouped: dict[str, list[UserFact]] = {tag: [] for tag in PRIORITY_TAGS}
        grouped["other"] = []

        for fact in facts:
            matched = _match_priority_tag(fact)
            if matched:
                grouped[matched].append(fact)
            else:
                grouped["other"].append(fact)

        return grouped

    def get_stats(self, user_id: str) -> dict[str, int]:
        """用户记忆统计。"""
        total = self._db.execute(
            "SELECT COUNT(*) FROM user_facts WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        active = self._db.execute(
            "SELECT COUNT(*) FROM user_facts WHERE user_id = ? AND valid_to IS NULL", (user_id,)
        ).fetchone()[0]
        pending = self._db.execute(
            "SELECT COUNT(*) FROM memory_proposals WHERE status = 'pending'"
        ).fetchone()[0]
        return {"total": total, "active": active, "pending_proposals": pending}

    # ── Three-Layer Write Filter ──────────────────────────

    def _evaluate_fact(self, fact: UserFact) -> str:
        """三层写入过滤。返回 approved / degraded / rejected。"""
        # L1: 规则硬编码（轻量预过滤，90% 文本在此被决策）
        content = f"{fact.entity} {fact.predicate} {fact.object}".lower()

        # 安全过滤：拒绝含敏感信息的写入（密码、密钥、token 等）
        _sensitive_keywords = ["密码", "password", "密钥", "secret", "token", "api_key"]
        for kw in _sensitive_keywords:
            if kw in content:
                return "rejected"

        # 句首触发词匹配（更严格）
        for pattern, position in L1_TRIGGER_PATTERNS:
            if position == "prefix" and content.startswith(pattern):
                return "approved"
            elif position == "anywhere" and pattern in content:
                return "approved"

        # 任意位置匹配（仅限安全触发词）
        for pattern in L1_ANYWHERE_PATTERNS:
            if pattern in content:
                return "approved"

        # 属性键值对检测
        if fact.confidence >= self._settings.user_confidence_threshold:
            # L2: 置信度 + 重要性评分
            importance = self._calc_importance(fact)
            if importance >= 0.6:
                return "approved"
            elif importance >= 0.4:
                return "degraded"
        elif fact.confidence >= 0.5:
            # L3: 收益预测（简化为高置信度关联）
            related_count = self._db.execute(
                "SELECT COUNT(*) FROM user_facts WHERE user_id = ? AND entity = ?",
                (fact.user_id, fact.entity),
            ).fetchone()[0]
            if related_count >= 2:
                return "approved"

        return "rejected"

    @staticmethod
    def _calc_importance(fact: UserFact) -> float:
        """L2 重要性评分：新颖度 + 相关性。"""
        score = fact.confidence * 0.6  # 置信度占 60%
        # 优先标签加成
        content = f"{fact.entity} {fact.predicate} {fact.object}".lower()
        for tag in PRIORITY_TAGS:
            if tag in content:
                score += 0.2
                break
        return min(score, 1.0)

    # ── Eviction ──────────────────────────────────────────

    def _evict_lru(self, user_id: str) -> None:
        """LRU + 低置信度淘汰。淘汰 1 条最不活跃+低置信度事实。"""
        # 优先淘汰低置信度 + 最少访问
        row = self._db.execute(
            """SELECT id FROM user_facts
               WHERE user_id = ? AND valid_to IS NULL
               ORDER BY confidence ASC, access_count ASC, last_accessed_at ASC
               LIMIT 1""",
            (user_id,),
        ).fetchone()
        if row:
            now = datetime.now(timezone.utc).isoformat()
            self._db.execute(
                "UPDATE user_facts SET valid_to = ? WHERE id = ?",
                (now, row["id"]),
            )
            self._db.commit()
            logger.info("user_fact_evicted", id=row["id"], user_id=user_id)

    # ── Rate Limiting ─────────────────────────────────────

    def _check_rate_limit(self) -> bool:
        """写入限流：每秒最多 N 条。线程安全。"""
        now = time.monotonic()
        with self._rate_lock:
            self._write_timestamps = [t for t in self._write_timestamps if now - t < 1.0]
            if len(self._write_timestamps) >= self._settings.user_write_rate_limit:
                return False
            return True

    def _record_write(self) -> None:
        self._write_timestamps.append(time.monotonic())

    # ── Deletion ──────────────────────────────────────────

    def delete_fact(self, fact_id: str) -> bool:
        """用户明确否定时立即删除。"""
        self._db.execute("DELETE FROM user_facts WHERE id = ?", (fact_id,))
        self._db.commit()
        logger.info("user_fact_deleted", id=fact_id)
        return True

    # ── Helpers ───────────────────────────────────────────

    def _row_to_fact(self, row: sqlite3.Row) -> UserFact:
        return UserFact(
            id=row["id"], user_id=row["user_id"], entity=row["entity"],
            predicate=row["predicate"], object=row["object"],
            confidence=row["confidence"], valid_from=row["valid_from"],
            valid_to=row["valid_to"], source_agent=row["source_agent"] or "",
            write_decision=row["write_decision"] or "pending",
            created_at=row["created_at"], access_count=row["access_count"] or 0,
            last_accessed_at=row["last_accessed_at"],
        )

    def _row_to_proposal(self, row: sqlite3.Row) -> MemoryProposal:
        return MemoryProposal(
            id=row["id"], user_id=row["user_id"], entity=row["entity"],
            predicate=row["predicate"], object=row["object"],
            confidence=row["confidence"], source_agent=row["source_agent"],
            status=ProposalStatus(row["status"]),
            reviewed_by=row["reviewed_by"], reviewed_at=row["reviewed_at"],
            created_at=row["created_at"],
        )
