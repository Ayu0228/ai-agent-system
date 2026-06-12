"""任务记忆（L2 Warm）—— 5 状态认知状态机 + 幂等写入 + ACL 隔离。

对齐文档规格：
- 72h retention，10 max concurrent
- 5 状态：intent_recognition → planning → executing_tools → observing_results → done/failed
- 原子状态更新，利用 SQLite 行锁防并发
- 幂等写入：session_step_id UNIQUE，冲突返回已处理结果
"""

from __future__ import annotations

import json as _json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from src.shared.config import get_settings
from src.shared.models import TaskCognitiveState, TaskRecord, TaskStateTransition

logger = structlog.get_logger()

_VALID_TRANSITIONS: dict[str, set[str]] = {
    "intent_recognition": {"planning"},
    "planning": {"executing_tools", "failed"},
    "executing_tools": {"observing_results", "failed"},
    "observing_results": {"executing_tools", "done", "failed"},
    "done": set(),       # 终态
    "failed": set(),     # 终态
}


class TaskMemory:
    """L2 任务记忆管理器。操作 SQLite tasks 表。"""

    def __init__(self, db: sqlite3.Connection | None = None) -> None:
        self._settings = get_settings()
        if db is not None:
            self._db = db
        else:
            from src.memory.long_term import LongTermMemory
            ltm = LongTermMemory()
            self._db = ltm._db  # 复用同一个 SQLite 连接

    # ── Task CRUD ─────────────────────────────────────────

    def create_task(
        self,
        owner_agent_id: str,
        *,
        parent_task_id: str | None = None,
        task_id: str | None = None,
        session_step_id: str | None = None,
        acl: dict[str, str] | None = None,
    ) -> TaskRecord:
        """创建新任务。先检查并发上限，再写入幂等表。"""
        # 并发上限检查
        active_count = self._db.execute(
            "SELECT COUNT(*) FROM tasks WHERE owner_agent_id = ? AND status NOT IN ('done', 'failed')",
            (owner_agent_id,),
        ).fetchone()[0]
        if active_count >= self._settings.task_max_concurrent:
            raise RuntimeError(
                f"Agent {owner_agent_id} 已达最大并发任务数 {self._settings.task_max_concurrent}"
            )

        tid = task_id or f"task_{uuid.uuid4().hex[:12]}"
        sid = session_step_id or f"step_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()

        # 幂等检查
        existing = self._db.execute(
            "SELECT task_id, status FROM agent_idempotency WHERE session_step_id = ?",
            (sid,),
        ).fetchone()
        if existing:
            logger.info("task_idempotent_skip", session_step_id=sid, existing_task=existing["task_id"])
            return self.get_task(existing["task_id"])  # type: ignore[return-value]

        # 幂等表写入
        self._db.execute(
            """INSERT INTO agent_idempotency (session_step_id, user_id, goal_hash, status, task_id, created_at)
               VALUES (?, ?, ?, 'processing', ?, ?)""",
            (sid, owner_agent_id, "", tid, now),
        )

        # 任务写入
        self._db.execute(
            """INSERT INTO tasks (task_id, session_step_id, owner_agent_id, parent_task_id,
               status, subtasks, artifacts, acl, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'intent_recognition', '[]', '{}', ?, ?, ?)""",
            (tid, sid, owner_agent_id, parent_task_id,
             _json.dumps(acl or {}), now, now),
        )
        self._db.commit()

        logger.info("task_created", task_id=tid, owner=owner_agent_id, parent=parent_task_id)
        return TaskRecord(
            task_id=tid, session_step_id=sid, owner_agent_id=owner_agent_id,
            parent_task_id=parent_task_id, acl=acl or {}, created_at=now, updated_at=now,
        )

    def get_task(self, task_id: str) -> TaskRecord | None:
        """查询单个任务。"""
        row = self._db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_active_tasks(self, agent_id: str | None = None) -> list[TaskRecord]:
        """查询活跃任务。"""
        if agent_id:
            rows = self._db.execute(
                """SELECT * FROM tasks
                   WHERE owner_agent_id = ? AND status NOT IN ('done', 'failed')
                   ORDER BY updated_at DESC LIMIT 5""",
                (agent_id,),
            ).fetchall()
        else:
            rows = self._db.execute(
                """SELECT * FROM tasks WHERE status NOT IN ('done', 'failed')
                   ORDER BY updated_at DESC LIMIT 20""",
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_task_chain(self, task_id: str) -> list[TaskRecord]:
        """获取任务链（父任务 + 所有子任务）。"""
        task = self.get_task(task_id)
        if task is None:
            return []
        chain = [task]
        subtasks = self._db.execute(
            "SELECT * FROM tasks WHERE parent_task_id = ?", (task_id,)
        ).fetchall()
        for st in subtasks:
            chain.append(self._row_to_record(st))
            chain.extend(self.get_task_chain(st["task_id"])[1:])
        return chain

    # ── State Machine ─────────────────────────────────────

    def update_task_status(
        self,
        task_id: str,
        new_status: str,
        *,
        session_step_id: str | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> TaskRecord | None:
        """原子状态更新。利用 SQLite 行锁防并发。

        先校验状态转移合法性，再原子 UPDATE，第二条执行流影响行数为 0。
        """
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        current = task.status.value if isinstance(task.status, TaskCognitiveState) else task.status
        target = new_status.value if isinstance(new_status, TaskCognitiveState) else new_status

        # 校验状态转移
        valid_targets = _VALID_TRANSITIONS.get(current, set())
        if target not in valid_targets:
            raise ValueError(
                f"非法状态转移: {current} → {target}，允许的目标: {valid_targets}"
            )

        now = datetime.now(timezone.utc).isoformat()
        sid = session_step_id or f"step_{uuid.uuid4().hex[:16]}"

        # 原子更新：WHERE 带当前状态，利用行锁
        cursor = self._db.execute(
            """UPDATE tasks SET status = ?, updated_at = ?,
               artifacts = COALESCE(?, artifacts)
               WHERE task_id = ? AND status = ?""",
            (target, now, _json.dumps(artifacts) if artifacts else None, task_id, current),
        )
        if cursor.rowcount == 0:
            logger.warning("task_state_race", task_id=task_id, from_state=current, to=target)
            return self.get_task(task_id)  # 已被其他流更新

        # 状态机日志
        self._db.execute(
            """INSERT INTO task_state_log (task_id, from_state, to_state, session_step_id, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (task_id, current, target, sid, now),
        )

        # 幂等表更新
        self._db.execute(
            "UPDATE agent_idempotency SET status = ? WHERE session_step_id = ?",
            (target, sid),
        )

        self._db.commit()
        logger.info("task_state_changed", task_id=task_id, from_state=current, to=target)
        return self.get_task(task_id)

    def mark_done(self, task_id: str, *, artifacts: dict[str, Any] | None = None) -> TaskRecord | None:
        """标记任务完成。验证完成条件后写入。"""
        task = self.get_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        # 完成判定：①所有子任务 done/failed
        subtasks = self._db.execute(
            "SELECT status FROM tasks WHERE parent_task_id = ?", (task_id,)
        ).fetchall()
        pending_subs = [s["status"] for s in subtasks if s["status"] not in ("done", "failed")]
        if pending_subs:
            raise ValueError(f"任务 {task_id} 有未完成的子任务: {pending_subs}")

        return self.update_task_status(task_id, "done", artifacts=artifacts)

    def check_ttl_and_archive(self) -> list[str]:
        """检查 72h TTL 到期的 done 任务，返回待归档任务 ID 列表。"""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=self._settings.task_retention_hours)).isoformat()
        rows = self._db.execute(
            """SELECT task_id FROM tasks
               WHERE status = 'done' AND updated_at < ?""",
            (cutoff,),
        ).fetchall()
        archived = [r["task_id"] for r in rows]
        if archived:
            logger.info("task_ttl_archive", count=len(archived), task_ids=archived)
        return archived

    # ── ACL ──────────────────────────────────────────────

    def can_read(self, task_id: str, agent_id: str) -> bool:
        """检查 agent 是否有读权限。"""
        task = self.get_task(task_id)
        if task is None:
            return False
        if task.owner_agent_id == agent_id:
            return True
        acl = task.acl
        if agent_id in acl:
            return acl[agent_id] in ("rw", "r")
        # 非同任务 agent 只能读已完成的
        if task.status in (TaskCognitiveState.DONE, TaskCognitiveState.FAILED):
            return True
        return False

    def can_write(self, task_id: str, agent_id: str) -> bool:
        """检查 agent 是否有写权限。"""
        task = self.get_task(task_id)
        if task is None:
            return False
        if task.owner_agent_id == agent_id:
            return True
        acl = task.acl
        return agent_id in acl and acl[agent_id] == "rw"

    # ── States ────────────────────────────────────────────

    def get_state_log(self, task_id: str) -> list[TaskStateTransition]:
        """获取任务状态流转日志。"""
        rows = self._db.execute(
            "SELECT * FROM task_state_log WHERE task_id = ? ORDER BY timestamp",
            (task_id,),
        ).fetchall()
        return [TaskStateTransition(
            task_id=r["task_id"], from_state=r["from_state"],
            to_state=r["to_state"], session_step_id=r["session_step_id"],
            timestamp=r["timestamp"],
        ) for r in rows]

    def get_idempotency_stats(self) -> dict[str, int]:
        """幂等表统计。"""
        total = self._db.execute("SELECT COUNT(*) FROM agent_idempotency").fetchone()[0]
        conflicts = self._db.execute(
            "SELECT COUNT(*) FROM agent_idempotency WHERE status = 'duplicate'"
        ).fetchone()[0]
        return {"total_records": total, "conflicts": conflicts}

    # ── Helpers ───────────────────────────────────────────

    def _row_to_record(self, row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"],
            session_step_id=row["session_step_id"],
            owner_agent_id=row["owner_agent_id"],
            parent_task_id=row["parent_task_id"],
            status=TaskCognitiveState(row["status"]),
            subtasks=_json.loads(row["subtasks"]) if row["subtasks"] else [],
            artifacts=_json.loads(row["artifacts"]) if row["artifacts"] else {},
            acl=_json.loads(row["acl"]) if row["acl"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
