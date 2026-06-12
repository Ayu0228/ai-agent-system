"""核心逻辑单元测试：状态机、衰减公式、冲突检测、token 估算、触发词匹配。"""

import pytest


# ═══════════════════════════════════════════════════════════════
# 状态机转移测试
# ═══════════════════════════════════════════════════════════════

from src.memory.task_memory import _VALID_TRANSITIONS


class TestStateMachine:
    """任务认知状态机 _VALID_TRANSITIONS 的合法性验证。"""

    def test_valid_transition_intent_to_planning(self):
        assert "planning" in _VALID_TRANSITIONS["intent_recognition"]

    def test_valid_transition_planning_to_executing(self):
        assert "executing_tools" in _VALID_TRANSITIONS["planning"]

    def test_valid_transition_executing_to_observing(self):
        assert "observing_results" in _VALID_TRANSITIONS["executing_tools"]

    def test_valid_transition_observing_to_executing_loop(self):
        """observing_results 可以回退到 executing_tools（循环）。"""
        assert "executing_tools" in _VALID_TRANSITIONS["observing_results"]

    def test_valid_transition_observing_to_done(self):
        assert "done" in _VALID_TRANSITIONS["observing_results"]

    def test_done_is_terminal(self):
        assert _VALID_TRANSITIONS["done"] == set()

    def test_failed_is_terminal(self):
        assert _VALID_TRANSITIONS["failed"] == set()

    def test_illegal_transition_blocked(self):
        """非法跳转不应存在。"""
        assert "done" not in _VALID_TRANSITIONS["intent_recognition"]
        assert "planning" not in _VALID_TRANSITIONS["executing_tools"]

    def test_all_states_have_transitions(self):
        """所有 5 个状态都在转移表中。"""
        expected = {"intent_recognition", "planning", "executing_tools", "observing_results", "done", "failed"}
        assert set(_VALID_TRANSITIONS.keys()) == expected


# ═══════════════════════════════════════════════════════════════
# Token 估算测试
# ═══════════════════════════════════════════════════════════════

from src.memory.short_term import ContextCompressor


class TestTokenEstimation:
    """ContextCompressor.estimate_tokens 边界测试。"""

    def test_empty_text_returns_zero(self):
        assert ContextCompressor.estimate_tokens("") == 0

    def test_pure_english(self):
        """纯英文：约 4 字符/token。"""
        text = "Hello world this is a test"
        tokens = ContextCompressor.estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)

    def test_pure_chinese(self):
        """纯中文：约 1.5 字符/token。"""
        text = "这是一段中文测试文本用来验证分词估算"
        tokens = ContextCompressor.estimate_tokens(text)
        assert tokens > 0
        # 纯中文每 1.5 字符约 1 token
        assert tokens < len(text)

    def test_mixed_cjk_latin(self):
        text = "Hello 世界 test 测试"
        tokens = ContextCompressor.estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)

    def test_should_compress_under_threshold(self):
        compressor = ContextCompressor()
        # 少量消息不应触发压缩
        msgs = [{"role": "user", "content": "hi"}]
        assert compressor.should_compress(msgs) is False

    def test_compress_preserves_recent_turns(self):
        compressor = ContextCompressor(recent_turns=3)
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        result = compressor.compress(msgs)
        # 最近 3 轮应保留原文
        assert len(result) >= 3


# ═══════════════════════════════════════════════════════════════
# 冲突检测启发式打分测试
# ═══════════════════════════════════════════════════════════════

from src.memory.knowledge_memory import KnowledgeMemory


class TestConflictScoring:
    """KnowledgeMemory._heuristic_score 边界测试。"""

    def test_same_source_high_score(self):
        a = {"source": "doc1", "object": "hello world", "confidence_weight": 1.0}
        b = {"source": "doc1", "object": "hello world", "confidence_weight": 1.0}
        score = KnowledgeMemory._heuristic_score(a, b)
        assert score > 0.8  # source 相同 + object 完全重叠 + confidence 无差 → 应自动合并

    def test_different_source_low_score(self):
        a = {"source": "doc1", "object": "abc", "confidence_weight": 1.0}
        b = {"source": "doc2", "object": "xyz", "confidence_weight": 0.1}
        score = KnowledgeMemory._heuristic_score(a, b)
        assert score < 0.8  # 来源不同 + object 无重叠 + confidence 差距大 → 不自动合并

    def test_empty_objects(self):
        a = {"source": "doc1", "object": "", "confidence_weight": 1.0}
        b = {"source": "doc1", "object": "", "confidence_weight": 1.0}
        score = KnowledgeMemory._heuristic_score(a, b)
        # source 相同 + no object Jaccard → ~0.4
        assert 0.3 < score < 0.8

    def test_score_bounded(self):
        a = {"source": "s", "object": "abc", "confidence_weight": 1.0}
        b = {"source": "s", "object": "abc", "confidence_weight": 1.0}
        score = KnowledgeMemory._heuristic_score(a, b)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════
# 知识冲突 duplicate 记录测试
# ═══════════════════════════════════════════════════════════════

import sqlite3
from src.shared.models import ConflictStatus, KnowledgeTriple


class TestDuplicateConflictTracking:
    """验证 duplicate 冲突现在会写入 conflict_queue 做统计。"""

    @pytest.fixture
    def db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_triples (
            id TEXT PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL,
            object TEXT NOT NULL, source TEXT DEFAULT '',
            confidence_weight REAL DEFAULT 1.0, created_at TEXT NOT NULL,
            chroma_entity_id TEXT
        );
        CREATE TABLE IF NOT EXISTS conflict_queue (
            id TEXT PRIMARY KEY, triple_a_id TEXT NOT NULL, triple_b_id TEXT NOT NULL,
            conflict_type TEXT NOT NULL, status TEXT DEFAULT 'pending',
            resolved_by TEXT, resolved_at TEXT, created_at TEXT NOT NULL
        );
        """)
        conn.commit()
        yield conn
        conn.close()

    def test_duplicate_writes_conflict_record(self, db):
        """duplicate 类型应该在 conflict_queue 里写一条 auto_merged 记录。"""
        km = KnowledgeMemory(db=db)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        # 插入第一条
        t1 = KnowledgeTriple(
            id="t1", subject="Python", predicate="is_a", object="编程语言",
            source="wiki", confidence_weight=1.0, created_at=now,
        )
        km.add_triple(t1)

        # 插入完全相同的一条（只 id 不同）
        t2 = KnowledgeTriple(
            id="t2", subject="Python", predicate="is_a", object="编程语言",
            source="wiki", confidence_weight=0.5, created_at=now,
        )
        km.add_triple(t2)

        # assert: conflict_queue 里有一条 duplicate → auto_merged 记录
        conflicts = db.execute(
            "SELECT * FROM conflict_queue WHERE conflict_type = 'duplicate'"
        ).fetchall()
        assert len(conflicts) == 1
        assert conflicts[0]["status"] == ConflictStatus.AUTO_MERGED.value
        assert conflicts[0]["triple_a_id"] == "t1"
        assert conflicts[0]["triple_b_id"] == "t2"

        # assert: t1 的 confidence_weight 被 bump 了（1.0 + 0.5 = 1.5）
        row = db.execute("SELECT * FROM knowledge_triples WHERE id = 't1'").fetchone()
        assert row["confidence_weight"] == 1.5


# ═══════════════════════════════════════════════════════════════
# 用户记忆触发词匹配测试
# ═══════════════════════════════════════════════════════════════

from src.memory.user_memory import L1_TRIGGER_PATTERNS, L1_ANYWHERE_PATTERNS


class TestL1TriggerPatterns:
    """验证 L1 触发词改为句首匹配后的行为。"""

    def test_prefix_patterns_all_lowercase(self):
        """所有 prefix 触发词应为小写（content 被 .lower() 处理）。"""
        for pattern, position in L1_TRIGGER_PATTERNS:
            assert pattern == pattern.lower(), f"'{pattern}' should be lowercase"
            if position == "prefix":
                assert len(pattern) >= 2, f"'{pattern}' too short for prefix match"

    def test_sensitive_keywords_blocked(self):
        """敏感词应被拒绝。"""
        from src.memory.user_memory import UserMemory
        from src.shared.models import UserFact

        um = UserMemory.__new__(UserMemory)  # skip init
        fact = UserFact(
            user_id="u1", entity="密码", predicate="是", object="123456",
            source_agent="test", confidence=0.9,
        )
        result = um._evaluate_fact(fact)
        assert result == "rejected"

    def test_prefix_match_approved(self):
        """句首匹配应批准。"""
        from src.memory.user_memory import UserMemory
        from src.shared.models import UserFact

        um = UserMemory.__new__(UserMemory)
        fact = UserFact(
            user_id="u1", entity="我喜欢", predicate="吃", object="火锅",
            source_agent="main", confidence=0.8,
        )
        # content = "我喜欢 吃 火锅" starts with "我喜欢"
        result = um._evaluate_fact(fact)
        assert result == "approved"


# ═══════════════════════════════════════════════════════════════
# 经验权重衰减测试
# ═══════════════════════════════════════════════════════════════

from src.memory.experience_memory import ExperienceMemory
from src.shared.models import ExperienceCard
from datetime import datetime, timedelta, timezone


class TestExperienceDecay:
    """ExperienceMemory._calc_weight 衰减计算测试。"""

    def test_recent_experience_boosted(self):
        em = ExperienceMemory.__new__(ExperienceMemory)
        from src.shared.config import get_settings
        em._settings = get_settings()
        card = ExperienceCard(
            owner_agent_id="a1",
            scenario="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            success_rate=0.8,
            weight=1.0,
        )
        weight = em._calc_weight(card)
        # 今天的经验应有 recency boost（1.5× success bonus 1.3 = 1.95）
        assert weight > 1.0

    def test_old_experience_decayed(self):
        em = ExperienceMemory.__new__(ExperienceMemory)
        from src.shared.config import get_settings
        em._settings = get_settings()
        card = ExperienceCard(
            owner_agent_id="a1",
            scenario="test",
            created_at=(datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),
            success_rate=0.5,
            weight=1.0,
        )
        weight = em._calc_weight(card)
        # 90 天 + 低成功率 → 权重应显著衰减
        assert weight < 0.5

    def test_weight_bounded(self):
        em = ExperienceMemory.__new__(ExperienceMemory)
        from src.shared.config import get_settings
        em._settings = get_settings()
        card = ExperienceCard(
            owner_agent_id="a1",
            scenario="test",
            created_at=datetime.now(timezone.utc).isoformat(),
            success_rate=1.0,
            weight=2.0,
        )
        weight = em._calc_weight(card)
        assert weight <= 2.0  # max cap
