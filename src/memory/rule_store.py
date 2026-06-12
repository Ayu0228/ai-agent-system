"""Rule Store — SQLite-backed storage for procedural rules (WHEN-THEN).

Rules are extracted from experiences and consolidated memories. They represent
reusable if-then patterns that guide agent behavior.

Lifecycle: ACTIVE → (fired) → COOLDOWN → ACTIVE | DEPRECATED
"""

from __future__ import annotations

import json as _json
import time
from pathlib import Path

import structlog

from src.shared.models import Rule, RuleStatus, ScopeLevel

logger = structlog.get_logger()


class RuleStore:
    """SQLite-backed procedural rule storage.

    用法:
        store = RuleStore(":memory:")  # or Path to file
        store.add_rule(Rule(trigger_condition="api error 429", action="wait 5s and retry"))
        rules = store.list_active("agent-1", ScopeLevel.AGENT)
        store.record_fire("rule-id", success=True)
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        import sqlite3

        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            trigger_condition TEXT NOT NULL DEFAULT '',
            action TEXT NOT NULL DEFAULT '',
            scope TEXT NOT NULL DEFAULT 'agent',
            agent_id TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.5,
            success_rate REAL NOT NULL DEFAULT 0.5,
            fire_count INTEGER NOT NULL DEFAULT 0,
            last_fired_at TEXT,
            cooldown_seconds INTEGER NOT NULL DEFAULT 300,
            status TEXT NOT NULL DEFAULT 'active',
            source_experience_id TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_rules_agent ON rules(agent_id);
        CREATE INDEX IF NOT EXISTS idx_rules_status ON rules(status);
        CREATE INDEX IF NOT EXISTS idx_rules_scope ON rules(scope);
        """)

    # ── CRUD ──────────────────────────────────────────

    def add_rule(self, rule: Rule) -> str:
        """插入或替换规则，返回 id。"""
        self._db.execute(
            """INSERT OR REPLACE INTO rules
               (id, name, description, trigger_condition, action, scope, agent_id,
                confidence, success_rate, fire_count, last_fired_at,
                cooldown_seconds, status, source_experience_id,
                created_at, expires_at, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rule.id, rule.name, rule.description,
                rule.trigger_condition, rule.action,
                rule.scope.value, rule.agent_id,
                rule.confidence, rule.success_rate, rule.fire_count,
                rule.last_fired_at, rule.cooldown_seconds,
                rule.status.value, rule.source_experience_id,
                rule.created_at, rule.expires_at,
                _json.dumps(rule.tags),
            ),
        )
        self._db.commit()
        logger.debug("rule_added", id=rule.id, name=rule.name)
        return rule.id

    def get_rule(self, rule_id: str) -> Rule | None:
        """按 ID 查询规则。"""
        row = self._db.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)).fetchone()
        return self._row_to_rule(row) if row else None

    def list_active(self, agent_id: str = "",
                    scope: ScopeLevel | None = None) -> list[Rule]:
        """列出活跃（非 DEPRECATED/COOLDOWN）的规则。"""
        query = "SELECT * FROM rules WHERE status = ?"
        params: list = [RuleStatus.ACTIVE.value]

        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if scope is not None:
            query += " AND scope = ?"
            params.append(scope.value)

        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_rule(r) for r in rows]

    def list_by_scope(self, scope: ScopeLevel) -> list[Rule]:
        """按作用域查询所有活跃规则。"""
        rows = self._db.execute(
            "SELECT * FROM rules WHERE scope = ? AND status = ?",
            (scope.value, RuleStatus.ACTIVE.value),
        ).fetchall()
        return [self._row_to_rule(r) for r in rows]

    def update_status(self, rule_id: str, status: RuleStatus) -> bool:
        """更新规则状态。"""
        self._db.execute(
            "UPDATE rules SET status = ? WHERE id = ?",
            (status.value, rule_id),
        )
        self._db.commit()
        return self._db.total_changes > 0

    # ── 触发追踪 ────────────────────────────────────

    def record_fire(self, rule_id: str, success: bool) -> None:
        """记录一次规则触发。"""
        now = time.time()
        self._db.execute(
            """UPDATE rules SET
               fire_count = fire_count + 1,
               last_fired_at = ?,
               success_rate = (success_rate * fire_count + ?) / (fire_count + 1.0)
               WHERE id = ?""",
            (now, 1.0 if success else 0.0, rule_id),
        )
        self._db.commit()

    def check_cooldown(self, rule_id: str) -> bool:
        """检查规则是否在冷却期。返回 True = 冷却中。"""
        row = self._db.execute(
            "SELECT last_fired_at, cooldown_seconds FROM rules WHERE id = ?",
            (rule_id,),
        ).fetchone()
        if not row or not row["last_fired_at"]:
            return False

        elapsed = time.time() - float(row["last_fired_at"])
        return elapsed < row["cooldown_seconds"]

    # ── 冲突 ────────────────────────────────────────

    def resolve_conflicts(self, rules: list[Rule]) -> list[Rule]:
        """冲突规则按 confidence × success_rate 排序，返回去重后列表。"""
        if len(rules) <= 1:
            return list(rules)

        # 按得分降序排序
        scored = [(r, r.confidence * r.success_rate) for r in rules]
        scored.sort(key=lambda x: x[1], reverse=True)

        # 去重：相同 trigger_condition 只保留最高分
        seen: set[str] = set()
        result: list[Rule] = []
        for rule, score in scored:
            key = rule.trigger_condition.lower().strip()
            if key not in seen:
                seen.add(key)
                result.append(rule)

        return result

    # ── 衰减 ────────────────────────────────────────

    def decay_rules(self, success_rate_threshold: float = 0.3) -> int:
        """标记低成功率的规则为 DEPRECATED，返回标记数量。"""
        rows = self._db.execute(
            "SELECT id, success_rate, fire_count FROM rules WHERE status = ?",
            (RuleStatus.ACTIVE.value,),
        ).fetchall()

        deprecated = 0
        for row in rows:
            # 至少触发过 3 次才考虑衰减
            if row["fire_count"] < 3:
                continue
            if row["success_rate"] < success_rate_threshold:
                self._db.execute(
                    "UPDATE rules SET status = ? WHERE id = ?",
                    (RuleStatus.DEPRECATED.value, row["id"]),
                )
                deprecated += 1

        if deprecated > 0:
            self._db.commit()
            logger.info("rules_decayed", deprecated=deprecated)

        return deprecated

    # ── 统计 ────────────────────────────────────────

    def get_stats(self) -> dict:
        statuses = {}
        for s in RuleStatus:
            count = self._db.execute(
                "SELECT COUNT(*) FROM rules WHERE status = ?", (s.value,),
            ).fetchone()[0]
            statuses[s.value] = count

        total = sum(statuses.values())
        return {
            "total": total,
            "by_status": statuses,
            "avg_success_rate": self._db.execute(
                "SELECT AVG(success_rate) FROM rules WHERE fire_count > 0"
            ).fetchone()[0] or 0.0,
        }

    # ── 内部 ────────────────────────────────────────

    def _row_to_rule(self, row) -> Rule:
        try:
            tags = _json.loads(row["tags"]) if row["tags"] else []
        except (_json.JSONDecodeError, TypeError):
            tags = []

        return Rule(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            trigger_condition=row["trigger_condition"],
            action=row["action"],
            scope=ScopeLevel(row["scope"]),
            agent_id=row["agent_id"],
            confidence=row["confidence"],
            success_rate=row["success_rate"],
            fire_count=row["fire_count"],
            last_fired_at=row["last_fired_at"],
            cooldown_seconds=row["cooldown_seconds"],
            status=RuleStatus(row["status"]),
            source_experience_id=row["source_experience_id"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            tags=tags,
        )

    def close(self) -> None:
        self._db.close()
