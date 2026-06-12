"""长期记忆存储：ChromaDB 向量检索 + SQLite 结构化存储。"""

from __future__ import annotations

import datetime
import json as _json
import os
import sqlite3
import threading
from pathlib import Path

import chromadb
import structlog

from src.shared.config import get_settings
from src.shared.models import MemoryEntry, MemoryQuery, MemorySearchResult

logger = structlog.get_logger()

# ChromaDB collection name — 可通过环境变量 CHROMADB_COLLECTION 覆盖
_CHROMA_COLLECTION = os.environ.get("CHROMADB_COLLECTION", "agent_memories")


class LongTermMemory:
    """两阶段检索：向量召回 → 重排序。

    ChromaDB 做语义检索，SQLite 做元数据管理和关键词搜索。
    """

    def __init__(self) -> None:
        settings = get_settings()
        chroma_path = settings.resolve_path(settings.chromadb_path)
        chroma_path.parent.mkdir(parents=True, exist_ok=True)

        self._chroma = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._chroma.get_or_create_collection(
            name=_CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        sqlite_path = settings.resolve_path(settings.sqlite_path)
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db_lock = threading.Lock()  # 多线程并发保护
        self._init_tables()

    # 当前数据库 schema 版本
    _SCHEMA_VERSION = 1

    def _init_tables(self) -> None:
        self._db.execute("PRAGMA journal_mode=WAL;")

        # Schema 版本管理
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS __schema_version__ (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )"""
        )
        current = self._db.execute(
            "SELECT MAX(version) FROM __schema_version__"
        ).fetchone()[0] or 0

        if current < self._SCHEMA_VERSION:
            self._run_migrations(current)
            self._db.execute(
                "INSERT INTO __schema_version__ (version, applied_at) VALUES (?, ?)",
                (self._SCHEMA_VERSION, datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )
            self._db.commit()

        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'fact',
            importance REAL NOT NULL DEFAULT 0.5,
            tags TEXT DEFAULT '[]',
            source_trace_id TEXT,
            created_at TEXT NOT NULL,
            last_accessed_at TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,
            expires_at TEXT,
            chroma_id TEXT UNIQUE
        );
        CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent_id);
        CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type);
        CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories(importance DESC);
        CREATE INDEX IF NOT EXISTS idx_mem_last_access ON memories(last_accessed_at);

        -- L2 任务记忆
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            session_step_id TEXT UNIQUE NOT NULL,
            owner_agent_id TEXT NOT NULL,
            parent_task_id TEXT,
            status TEXT NOT NULL DEFAULT 'intent_recognition',
            subtasks TEXT DEFAULT '[]',
            artifacts TEXT DEFAULT '{}',
            acl TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_owner ON tasks(owner_agent_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_task_id);

        -- 幂等记录
        CREATE TABLE IF NOT EXISTS agent_idempotency (
            session_step_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            goal_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'processing',
            final_result TEXT,
            task_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_idemp_goal ON agent_idempotency(goal_hash);

        -- L3 经验记忆
        CREATE TABLE IF NOT EXISTS experiences (
            experience_id TEXT PRIMARY KEY,
            owner_agent_id TEXT NOT NULL,
            scenario TEXT NOT NULL,
            approach TEXT NOT NULL,
            result TEXT NOT NULL,
            lesson TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            weight REAL NOT NULL DEFAULT 1.0,
            shareable INTEGER DEFAULT 0,
            usage_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.5,
            self_improving_track TEXT DEFAULT 'self-improving',
            autoscale_eligible INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            last_accessed_at TEXT NOT NULL,
            task_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_exp_owner ON experiences(owner_agent_id);
        CREATE INDEX IF NOT EXISTS idx_exp_shareable ON experiences(shareable);

        -- L3 知识图谱三元组
        CREATE TABLE IF NOT EXISTS knowledge_triples (
            id TEXT PRIMARY KEY,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            source TEXT DEFAULT '',
            confidence_weight REAL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            chroma_entity_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trip_subject ON knowledge_triples(subject);
        CREATE INDEX IF NOT EXISTS idx_trip_predicate ON knowledge_triples(predicate);

        -- 知识冲突队列
        CREATE TABLE IF NOT EXISTS conflict_queue (
            id TEXT PRIMARY KEY,
            triple_a_id TEXT NOT NULL,
            triple_b_id TEXT NOT NULL,
            conflict_type TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            resolved_by TEXT,
            resolved_at TEXT,
            created_at TEXT NOT NULL
        );

        -- L3 用户记忆
        CREATE TABLE IF NOT EXISTS user_facts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            entity TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            source_agent TEXT NOT NULL,
            write_decision TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,
            last_accessed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_uf_user ON user_facts(user_id);
        CREATE INDEX IF NOT EXISTS idx_uf_confidence ON user_facts(confidence);

        -- 认知状态机日志
        CREATE TABLE IF NOT EXISTS task_state_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            from_state TEXT NOT NULL,
            to_state TEXT NOT NULL,
            session_step_id TEXT NOT NULL,
            timestamp TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tsl_task ON task_state_log(task_id);

        -- 用户记忆写入提案（其他 agent 提交，main 审核）
        CREATE TABLE IF NOT EXISTS memory_proposals (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            entity TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            source_agent TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            reviewed_by TEXT,
            reviewed_at TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_mp_status ON memory_proposals(status);
        """)

    def _run_migrations(self, from_version: int) -> None:
        """执行增量 schema 迁移。按版本号顺序执行。"""
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()

        # v0 → v1: 添加 last_accessed_at 索引
        if from_version < 1:
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_last_access ON memories(last_accessed_at)"
            )
            logger.info("schema_migration", from_version=from_version, to=1)

    # ── Write ────────────────────────────────────────────

    def store(self, entry: MemoryEntry) -> str:
        """写入一条记忆到 ChromaDB + SQLite。"""
        # ChromaDB 向量存储（ChromaDB 内部有自己的锁）
        existing = self._collection.get(ids=[entry.id])
        if existing and existing["ids"]:
            self._collection.update(
                ids=[entry.id],
                documents=[entry.content],
                metadatas=[{
                    "agent_id": entry.agent_id,
                    "memory_type": entry.memory_type.value,
                    "importance": entry.importance,
                    "tags": ",".join(entry.tags),
                }],
            )
        else:
            self._collection.add(
                ids=[entry.id],
                documents=[entry.content],
                metadatas=[{
                    "agent_id": entry.agent_id,
                    "memory_type": entry.memory_type.value,
                    "importance": entry.importance,
                    "tags": ",".join(entry.tags),
                }],
            )

        # SQLite 元数据（加锁保护）
        with self._db_lock:
            self._db.execute(
                """INSERT OR REPLACE INTO memories
                   (id, agent_id, content, memory_type, importance, tags,
                    source_trace_id, created_at, last_accessed_at,
                    access_count, expires_at, chroma_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.id, entry.agent_id, entry.content,
                    entry.memory_type.value, entry.importance,
                    _json.dumps(entry.tags), entry.source_trace_id,
                    entry.created_at, entry.last_accessed_at,
                    entry.access_count, entry.expires_at, entry.id,
                ),
            )
            self._db.commit()
        logger.debug("memory_stored", id=entry.id, agent=entry.agent_id)
        return entry.id

    # ── Read ─────────────────────────────────────────────

    def search(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """语义搜索：ChromaDB 向量召回 → 结果组装。"""
        where: dict | None = None
        if query.agent_id:
            where = {"agent_id": query.agent_id}

        try:
            raw = self._collection.query(
                query_texts=[query.query_text],
                n_results=min(query.top_k * 3, 50),
                where=where,
            )
        except Exception as e:
            logger.error("chromadb_query_failed", error=str(e))
            return self._keyword_fallback(query)

        results: list[MemorySearchResult] = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return results

        with self._db_lock:
            for i, doc_id in enumerate(raw["ids"][0]):
                distance = raw.get("distances", [[1.0]])[0][i] if raw.get("distances") else 1.0
                metadata = raw.get("metadatas", [[{}]])[0][i] if raw.get("metadatas") else {}
                document = raw.get("documents", [[""]])[0][i] if raw.get("documents") else ""

                row = self._db.execute(
                    "SELECT * FROM memories WHERE id = ?", (doc_id,)
                ).fetchone()
                if row is None:
                    continue

                tags = _json.loads(row["tags"]) if row["tags"] else []
                from src.shared.models import MemoryType
                entry = MemoryEntry(
                    id=row["id"],
                    agent_id=row["agent_id"],
                    content=row["content"],
                    memory_type=MemoryType(row["memory_type"]),
                    tags=tags,
                    importance=row["importance"],
                    source_trace_id=row["source_trace_id"],
                    created_at=row["created_at"],
                    last_accessed_at=row["last_accessed_at"],
                    access_count=row["access_count"],
                    expires_at=row["expires_at"],
                )
                score = max(0.0, 1.0 - distance)
                results.append(MemorySearchResult(entry=entry, score=round(score, 4)))

        # 按分数降序 → top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.top_k]

    def _keyword_fallback(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """ChromaDB 不可用时降级为 SQLite LIKE 搜索。"""
        words = query.query_text.split()
        if not words:
            return []
        like_clauses = " OR ".join(["content LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words]
        if query.agent_id:
            like_clauses = f"({like_clauses}) AND agent_id = ?"
            params.append(query.agent_id)

        with self._db_lock:
            rows = self._db.execute(
                f"SELECT * FROM memories WHERE {like_clauses} ORDER BY importance DESC LIMIT ?",
                [*params, query.top_k],
            ).fetchall()

        results: list[MemorySearchResult] = []
        from src.shared.models import MemoryType
        for row in rows:
            tags = _json.loads(row["tags"]) if row["tags"] else []
            entry = MemoryEntry(
                id=row["id"], agent_id=row["agent_id"], content=row["content"],
                memory_type=MemoryType(row["memory_type"]), tags=tags,
                importance=row["importance"], source_trace_id=row["source_trace_id"],
                created_at=row["created_at"], last_accessed_at=row["last_accessed_at"],
                access_count=row["access_count"], expires_at=row["expires_at"],
            )
            results.append(MemorySearchResult(entry=entry, score=0.5))
        return results

    # ── Maintenance ──────────────────────────────────────

    def decay(self, *, threshold: float = 0.3) -> int:
        """遗忘策略：retain_score < threshold 的条目归档删除。

        用 SQL 先过滤掉高重要性/高频访问的记录，减少遍历量。
        """
        import datetime

        now = datetime.datetime.now(datetime.timezone.utc)
        with self._db_lock:
            # 只遍历中低重要性 + 低频访问的记录（has index on importance）
            rows = self._db.execute(
                """SELECT id, importance, memory_type, last_accessed_at, access_count
                   FROM memories
                   WHERE expires_at IS NULL
                     AND importance <= 0.9
                     AND access_count <= 10"""
            ).fetchall()

            removed = 0
            for row in rows:
                try:
                    last = datetime.datetime.fromisoformat(row["last_accessed_at"])
                    days = (now - last).days
                except (ValueError, TypeError):
                    days = 30

                freshness = 1.0 / (1.0 + days / 30.0)
                freq = min(row["access_count"] / 10.0, 1.0)

                if row["memory_type"] == "experience":
                    score = row["importance"] * 0.3 + freshness * 0.3 + freq * 0.4
                else:
                    score = row["importance"] * 0.6 + freshness * 0.2 + freq * 0.2

                if score < threshold:
                    self._db.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
                    try:
                        self._collection.delete(ids=[row["id"]])
                    except Exception:
                        pass
                    removed += 1

            self._db.commit()
        logger.info("memory_decay_complete", removed=removed)
        return removed

    def close(self) -> None:
        self._db.close()
