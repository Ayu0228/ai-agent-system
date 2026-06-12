"""Feedback collector, analyzer, and data flywheel tests."""

import time
import json
import tempfile
from pathlib import Path

import pytest

from src.feedback.collector import (
    FeedbackCollector, FeedbackEntry, FeedbackType,
)
from src.feedback.analyzer import (
    FeedbackAnalyzer, AnalysisResult, ImprovementTicket, TicketStatus,
)
from src.feedback.flywheel import (
    DataFlywheel, FlywheelEntry, FlywheelStage,
)


class TestFeedbackEntry:
    """Test FeedbackEntry dataclass."""

    def test_defaults(self):
        entry = FeedbackEntry()
        assert entry.type == FeedbackType.OTHER
        assert entry.rating == 0
        assert entry.severity == "medium"
        assert entry.resolved is False

    def test_rating_entry(self):
        entry = FeedbackEntry(
            type=FeedbackType.RATING,
            agent_id="agent-1",
            rating=5,
            comment="Great!",
        )
        assert entry.type == FeedbackType.RATING
        assert entry.rating == 5

    def test_hallucination_entry(self):
        entry = FeedbackEntry(
            type=FeedbackType.HALLUCINATION,
            agent_id="agent-1",
            comment="Fake citation",
            expected_output="With source links",
            actual_output="Made up info",
            severity="high",
        )
        assert entry.severity == "high"


class TestFeedbackCollector:
    """Test FeedbackCollector."""

    @pytest.fixture
    def collector(self):
        return FeedbackCollector()

    def test_submit_returns_id(self, collector):
        entry = FeedbackEntry(type=FeedbackType.RATING, agent_id="a1", rating=4)
        eid = collector.submit(entry)
        assert len(eid) == 12
        assert collector.entry_count == 1

    def test_submit_rating_shortcut(self, collector):
        eid = collector.submit_rating("agent-1", 4, comment="nice")
        assert collector.entry_count == 1
        entry = collector.get_by_agent("agent-1")[0]
        assert entry.rating == 4

    def test_submit_correction_shortcut(self, collector):
        eid = collector.submit_correction(
            "agent-1", expected="correct answer",
            actual="wrong answer", comment="fix"
        )
        entry = collector.get_by_agent("agent-1")[0]
        assert entry.type == FeedbackType.CORRECTION
        assert entry.severity == "high"

    def test_ring_buffer(self, collector):
        collector._max_entries = 5
        for i in range(10):
            collector.submit(FeedbackEntry(agent_id="a1", comment=str(i)))
        assert collector.entry_count == 5
        assert collector._entries[0].comment == "5"  # first 5 evicted

    def test_get_by_agent(self, collector):
        collector.submit(FeedbackEntry(agent_id="a1", comment="x"))
        collector.submit(FeedbackEntry(agent_id="a1", comment="y"))
        collector.submit(FeedbackEntry(agent_id="a2", comment="z"))
        assert len(collector.get_by_agent("a1")) == 2
        assert len(collector.get_by_agent("a2")) == 1

    def test_get_by_type(self, collector):
        collector.submit(FeedbackEntry(type=FeedbackType.HALLUCINATION))
        collector.submit(FeedbackEntry(type=FeedbackType.HALLUCINATION))
        collector.submit(FeedbackEntry(type=FeedbackType.LATENCY))
        assert len(collector.get_by_type(FeedbackType.HALLUCINATION)) == 2

    def test_get_unresolved(self, collector):
        collector.submit(FeedbackEntry(severity="high", resolved=False))
        collector.submit(FeedbackEntry(severity="medium", resolved=False))
        collector.submit(FeedbackEntry(severity="low", resolved=False))
        # get_unresolved with severity="high" gets high + critical
        high = collector.get_unresolved(severity="high")
        assert len(high) == 1

    def test_get_unresolved_critical(self, collector):
        collector.submit(FeedbackEntry(severity="critical", resolved=False))
        collector.submit(FeedbackEntry(severity="high", resolved=True))
        critical = collector.get_unresolved(severity="critical")
        assert len(critical) == 1

    def test_resolve(self, collector):
        entry = FeedbackEntry(comment="bug")
        eid = collector.submit(entry)
        assert collector.resolve(eid, resolution="fixed") is True
        assert entry.resolved is True
        assert entry.resolution == "fixed"

    def test_resolve_nonexistent(self, collector):
        assert collector.resolve("nope") is False

    def test_get_agent_rating(self, collector):
        collector.submit(FeedbackEntry(agent_id="a1", type=FeedbackType.RATING, rating=4))
        collector.submit(FeedbackEntry(agent_id="a1", type=FeedbackType.RATING, rating=2))
        info = collector.get_agent_rating("a1")
        assert info["avg_rating"] == 3.0
        assert info["count"] == 2
        assert info["distribution"]["4"] == 1
        assert info["distribution"]["2"] == 1

    def test_get_agent_rating_no_ratings(self, collector):
        info = collector.get_agent_rating("unknown")
        assert info["avg_rating"] == 0
        assert info["count"] == 0

    def test_get_stats(self, collector):
        collector.submit(FeedbackEntry(agent_id="a1", type=FeedbackType.BUG_REPORT, severity="high"))
        collector.submit(FeedbackEntry(agent_id="a2", type=FeedbackType.RATING, rating=5))
        stats = collector.get_stats()
        assert stats["total"] == 2
        assert stats["resolved"] == 0
        assert stats["unresolved"] == 2

    def test_get_stats_filtered(self, collector):
        collector.submit(FeedbackEntry(agent_id="a1"))
        collector.submit(FeedbackEntry(agent_id="a2"))
        stats = collector.get_stats(agent_id="a1")
        assert stats["total"] == 1


class TestFeedbackAnalyzer:
    """Test FeedbackAnalyzer."""

    @pytest.fixture
    def collector_and_analyzer(self):
        collector = FeedbackCollector()
        analyzer = FeedbackAnalyzer(collector)
        return collector, analyzer

    def test_analyze_empty(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        result = analyzer.analyze(agent_id="a1")
        assert result.total_feedback == 0

    def test_analyze_no_collector(self):
        analyzer = FeedbackAnalyzer()
        result = analyzer.analyze()
        assert result.total_feedback == 0

    def test_analyze_with_ratings(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        for r in [4, 4, 4]:
            collector.submit(FeedbackEntry(
                agent_id="a1", type=FeedbackType.RATING, rating=r,
            ))
        result = analyzer.analyze(agent_id="a1", period_days=30)
        assert result.avg_rating == 4.0

    def test_rating_trend_improving(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        # First half: low ratings
        old = time.time() - 6 * 86400  # 6 days ago
        recent = time.time() - 1 * 86400  # 1 day ago
        collector.submit(FeedbackEntry(
            agent_id="a1", type=FeedbackType.RATING, rating=2,
            created_at=old,
        ))
        collector.submit(FeedbackEntry(
            agent_id="a1", type=FeedbackType.RATING, rating=5,
            created_at=recent,
        ))
        result = analyzer.analyze(agent_id="a1", period_days=7)
        assert result.rating_trend == "improving"

    def test_rating_trend_declining(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        old = time.time() - 6 * 86400
        recent = time.time() - 1 * 86400
        collector.submit(FeedbackEntry(
            agent_id="a1", type=FeedbackType.RATING, rating=5,
            created_at=old,
        ))
        collector.submit(FeedbackEntry(
            agent_id="a1", type=FeedbackType.RATING, rating=2,
            created_at=recent,
        ))
        result = analyzer.analyze(agent_id="a1", period_days=7)
        assert result.rating_trend == "declining"

    def test_alert_on_declining_low_rating(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        old = time.time() - 6 * 86400
        recent = time.time() - 1 * 86400
        collector.submit(FeedbackEntry(
            agent_id="a1", type=FeedbackType.RATING, rating=4,
            created_at=old,
        ))
        collector.submit(FeedbackEntry(
            agent_id="a1", type=FeedbackType.RATING, rating=1,
            created_at=recent,
        ))
        result = analyzer.analyze(agent_id="a1", period_days=7)
        assert result.alert is True

    def test_alert_on_hallucination_surge(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        for _ in range(5):
            collector.submit(FeedbackEntry(
                agent_id="a1", type=FeedbackType.HALLUCINATION,
                severity="high",
            ))
        result = analyzer.analyze(agent_id="a1", period_days=30)
        assert result.alert is True

    def test_alert_on_high_severity(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        for _ in range(3):
            collector.submit(FeedbackEntry(severity="critical"))
        result = analyzer.analyze(period_days=30)
        assert result.alert is True

    def test_no_alert_on_normal_feedback(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        collector.submit(FeedbackEntry(
            type=FeedbackType.RATING, rating=4, severity="low",
        ))
        result = analyzer.analyze(period_days=30)
        assert result.alert is False

    def test_recommendations_generated_on_alert(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        for _ in range(6):
            collector.submit(FeedbackEntry(
                type=FeedbackType.HALLUCINATION, severity="high",
            ))
        result = analyzer.analyze(period_days=30)
        assert len(result.recommendations) > 0

    # ── Tickets ───────────────────────────────────────

    def test_generate_tickets(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        for _ in range(3):
            collector.submit(FeedbackEntry(
                agent_id="a1", type=FeedbackType.TOOL_ERROR,
                severity="high", comment="tool X failed",
            ))
        analysis = analyzer.analyze(agent_id="a1", period_days=30)
        tickets = analyzer.generate_tickets(analysis)
        assert len(tickets) >= 1
        assert tickets[0].agent_id == "a1"
        assert tickets[0].suggested_action == "tool_fix"

    def test_generate_tickets_skips_low_count(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        collector.submit(FeedbackEntry(type=FeedbackType.BUG_REPORT))
        analysis = analyzer.analyze(period_days=30)
        tickets = analyzer.generate_tickets(analysis)
        assert len(tickets) == 0  # only 1 report, threshold is 2

    def test_get_open_tickets(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        collector.submit(FeedbackEntry(
            agent_id="a1", type=FeedbackType.HALLUCINATION,
            severity="high",
        ))
        collector.submit(FeedbackEntry(
            agent_id="a1", type=FeedbackType.HALLUCINATION,
            severity="high",
        ))
        analysis = analyzer.analyze(agent_id="a1", period_days=30)
        analyzer.generate_tickets(analysis)
        tickets = analyzer.get_open_tickets()
        assert len(tickets) >= 1

    def test_resolve_ticket(self, collector_and_analyzer):
        collector, analyzer = collector_and_analyzer
        for _ in range(3):
            collector.submit(FeedbackEntry(
                agent_id="a1", type=FeedbackType.LATENCY,
                severity="high", comment="slow",
            ))
        analysis = analyzer.analyze(agent_id="a1", period_days=30)
        tickets = analyzer.generate_tickets(analysis)
        if tickets:
            assert analyzer.resolve_ticket(tickets[0].id) is True
            assert tickets[0].status == TicketStatus.RESOLVED

    def test_resolve_ticket_nonexistent(self, collector_and_analyzer):
        _, analyzer = collector_and_analyzer
        assert analyzer.resolve_ticket("nope") is False

    def test_set_collector(self):
        analyzer = FeedbackAnalyzer()
        collector = FeedbackCollector()
        analyzer.set_collector(collector)
        assert analyzer._collector is collector


class TestDataFlywheel:
    """Test DataFlywheel."""

    @pytest.fixture
    def fw(self):
        return DataFlywheel()

    def test_collect_failure(self, fw):
        entry = fw.collect_failure(
            agent_id="researcher",
            input_message="What is ML?",
            actual_output="ML is magic",
            error_type="hallucination",
        )
        assert entry.stage == FlywheelStage.COLLECT
        assert entry.agent_id == "researcher"
        assert entry.error_type == "hallucination"

    def test_collect_failure_with_metadata(self, fw):
        entry = fw.collect_failure(
            agent_id="a1", input_message="q", actual_output="a",
            error_type="wrong_answer", trace_id="t1", severity="critical",
            extra_field="value",
        )
        assert entry.metadata.get("extra_field") == "value"

    def test_annotate(self, fw):
        entry = fw.collect_failure(
            agent_id="a1", input_message="q", actual_output="a",
            error_type="wrong_answer",
        )
        assert fw.annotate(entry.id, "expected answer", annotator="reviewer-1")
        assert entry.expected_output == "expected answer"
        assert entry.stage == FlywheelStage.REVIEW

    def test_annotate_nonexistent(self, fw):
        assert fw.annotate("nope", "answer") is False

    def test_annotate_wrong_stage(self, fw):
        entry = fw.collect_failure(
            agent_id="a1", input_message="q", actual_output="a",
            error_type="x",
        )
        fw.annotate(entry.id, "expected")
        # Already in REVIEW, can't annotate again
        assert fw.annotate(entry.id, "another") is False

    def test_promote(self, fw):
        entry = fw.collect_failure(
            agent_id="a1", input_message="q", actual_output="a",
            error_type="hallucination",
        )
        fw.annotate(entry.id, "expected answer", "reviewer-1")
        promoted = fw.promote(entry.id, reviewer="lead-1")
        assert promoted is not None
        assert promoted.stage == FlywheelStage.PROMOTE
        assert promoted.golden_case_id == f"FW-{entry.id}"

    def test_promote_without_annotation(self, fw):
        entry = fw.collect_failure(
            agent_id="a1", input_message="q", actual_output="a",
            error_type="x",
        )
        # Not yet annotated = no expected_output
        assert fw.promote(entry.id) is None

    def test_promote_nonexistent(self, fw):
        assert fw.promote("nope") is None

    def test_reject(self, fw):
        entry = fw.collect_failure(
            agent_id="a1", input_message="q", actual_output="a",
            error_type="x",
        )
        fw.annotate(entry.id, "expected")
        assert fw.reject(entry.id, "not useful") is True
        assert entry.stage == FlywheelStage.COLLECT
        assert "REJECTED" in entry.notes

    def test_reject_nonexistent(self, fw):
        assert fw.reject("nope") is False

    # ── Export ────────────────────────────────────────

    def test_export_golden_cases(self, fw):
        entry = fw.collect_failure(
            agent_id="a1", input_message="q", actual_output="a",
            error_type="hallucination",
        )
        fw.annotate(entry.id, "expected")
        fw.promote(entry.id, "reviewer")
        cases = fw.export_golden_cases()
        assert len(cases) == 1
        assert cases[0]["agent_id"] == "a1"
        assert cases[0]["id"] == f"FW-{entry.id}"

    def test_export_empty(self, fw):
        assert fw.export_golden_cases() == []

    def test_export_jsonl(self, fw):
        entry = fw.collect_failure(
            agent_id="a1", input_message="q", actual_output="a",
            error_type="hallucination",
        )
        fw.annotate(entry.id, "expected")
        fw.promote(entry.id, "reviewer")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            fpath = f.name
        try:
            count = fw.export_jsonl(fpath)
            assert count == 1
            content = Path(fpath).read_text()
            assert "flywheel" in content
        finally:
            Path(fpath).unlink(missing_ok=True)

    # ── Query ─────────────────────────────────────────

    def test_get_by_stage(self, fw):
        e1 = fw.collect_failure(agent_id="a1", input_message="q1",
                                actual_output="a1", error_type="x")
        e2 = fw.collect_failure(agent_id="a1", input_message="q2",
                                actual_output="a2", error_type="y")
        fw.annotate(e1.id, "expected")
        assert len(fw.get_by_stage(FlywheelStage.COLLECT)) >= 1

    def test_get_pending_review(self, fw):
        entry = fw.collect_failure(agent_id="a1", input_message="q",
                                   actual_output="a", error_type="x")
        fw.annotate(entry.id, "expected")
        assert len(fw.get_pending_review()) == 1

    def test_get_stats(self, fw):
        entry = fw.collect_failure(agent_id="a1", input_message="q",
                                   actual_output="a", error_type="x")
        fw.annotate(entry.id, "expected")
        fw.promote(entry.id, "reviewer")
        stats = fw.get_stats()
        assert stats["total"] == 1
        assert stats["promoted"] == 1

    # ── Persistence ───────────────────────────────────

    def test_save_entry(self, fw):
        with tempfile.TemporaryDirectory() as d:
            fw2 = DataFlywheel(storage_dir=d)
            entry = fw2.collect_failure(
                agent_id="a1", input_message="q", actual_output="a",
                error_type="hallucination",
            )
            fw2.annotate(entry.id, "expected")
            fw2.promote(entry.id, "reviewer")
            # Check file exists
            files = list(Path(d).glob("*.json"))
            assert len(files) >= 1

    def test_to_golden_case(self, fw):
        entry = fw.collect_failure(
            agent_id="researcher",
            input_message="What is Python?",
            actual_output="Python is a snake",
            error_type="hallucination",
            notes="Should be about programming",
        )
        fw.annotate(entry.id, "Python is a programming language")
        case = entry.to_golden_case()
        assert case["agent_id"] == "researcher"
        assert "assertion_type" in case
        assert "input" in case


class TestFlywheelEntry:
    """Test FlywheelEntry dataclass."""

    def test_defaults(self):
        entry = FlywheelEntry()
        assert entry.stage == FlywheelStage.COLLECT
        assert entry.severity == "medium"
        assert entry.error_type == ""

    def test_extract_keywords(self):
        keywords = FlywheelEntry._extract_keywords("Python is a programming language")
        assert len(keywords) > 0
        assert isinstance(keywords[0], str)


class TestFeedbackType:
    """Test FeedbackType enum."""

    def test_all_types(self):
        types = list(FeedbackType)
        assert FeedbackType.RATING in types
        assert FeedbackType.HALLUCINATION in types
        assert FeedbackType.TOOL_ERROR in types
        assert len(types) == 8  # 7 + OTHER


class TestFlywheelStage:
    """Test FlywheelStage enum."""

    def test_all_stages(self):
        stages = list(FlywheelStage)
        assert FlywheelStage.COLLECT in stages
        assert FlywheelStage.PROMOTE in stages
        assert len(stages) == 5
