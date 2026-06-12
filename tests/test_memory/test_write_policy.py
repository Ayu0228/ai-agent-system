"""WritePolicy 单元测试。"""

from src.shared.models import MemoryEntry, MemoryType, MemorySearchResult, WriteDecision


class TestWritePolicy:
    """测试写入决策策略。"""

    def test_write_novel_important(self):
        from src.memory.write_policy import WritePolicy

        policy = WritePolicy()
        entry = MemoryEntry(
            agent_id="researcher",
            content="new insight about web scraping",
            importance=0.8,
        )
        result = policy.evaluate(entry, [])
        assert result.decision == WriteDecision.WRITE
        assert result.importance == 0.8

    def test_skip_low_importance(self):
        from src.memory.write_policy import WritePolicy

        policy = WritePolicy(min_importance=0.4)
        entry = MemoryEntry(
            agent_id="researcher",
            content="random thought",
            importance=0.2,
        )
        result = policy.evaluate(entry, [])
        assert result.decision == WriteDecision.SKIP

    def test_merge_duplicate(self):
        from src.memory.write_policy import WritePolicy

        policy = WritePolicy(dedup_threshold=0.9)
        entry = MemoryEntry(
            agent_id="researcher",
            content="similar content",
            importance=0.7,
        )
        existing = MemoryEntry(
            agent_id="researcher",
            content="very similar content here",
            importance=0.8,
        )
        result = policy.evaluate(
            entry,
            [MemorySearchResult(entry=existing, score=0.95)],
        )
        assert result.decision == WriteDecision.MERGE

    def test_write_below_dedup_threshold(self):
        from src.memory.write_policy import WritePolicy

        policy = WritePolicy(dedup_threshold=0.9)
        entry = MemoryEntry(
            agent_id="researcher",
            content="completely different topic",
            importance=0.7,
        )
        existing = MemoryEntry(
            agent_id="researcher",
            content="some other memory",
        )
        result = policy.evaluate(
            entry,
            [MemorySearchResult(entry=existing, score=0.5)],
        )
        assert result.decision == WriteDecision.WRITE

    def test_custom_thresholds(self):
        from src.memory.write_policy import WritePolicy

        policy = WritePolicy(min_importance=0.6, dedup_threshold=0.7)
        entry = MemoryEntry(
            agent_id="researcher",
            content="mediocre content",
            importance=0.5,
        )
        result = policy.evaluate(entry, [])
        assert result.decision == WriteDecision.SKIP
