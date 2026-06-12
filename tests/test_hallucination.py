"""幻觉检测守卫测试。"""

from __future__ import annotations

import pytest

from src.safety.hallucination import (
    HallucinationGuard,
    HallucinationResult,
    AbstentionDecision,
)


class TestHallucinationResult:
    """HallucinationResult 数据模型测试。"""

    def test_defaults(self):
        r = HallucinationResult()
        assert r.score == 0.0
        assert r.hallucination_detected is False
        assert r.risk_level == "low"
        assert r.markers_found == []
        assert r.suggestion == ""

    def test_critical_detection(self):
        r = HallucinationResult(score=0.1, hallucination_detected=True,
                                risk_level="critical", suggestion="BLOCK")
        assert r.hallucination_detected is True
        assert r.risk_level == "critical"


class TestPatternMatching:
    """Layer 1: 模式匹配测试。"""

    @pytest.fixture
    def guard(self):
        return HallucinationGuard()

    def test_chinese_hallucination_pattern(self, guard):
        penalty, markers = guard._check_patterns("据可靠消息，这个市场将会大涨")
        assert penalty > 0
        assert any("据可靠消息" in m for m in markers)

    def test_english_hallucination_pattern(self, guard):
        penalty, markers = guard._check_patterns("I made this up but the answer is 42")
        assert penalty > 0.5
        assert any("I made this up" in m for m in markers)

    def test_clean_text_no_penalty(self, guard):
        penalty, markers = guard._check_patterns("根据2024年研究报告显示，该指标增长15%")
        assert penalty == 0.0
        assert markers == []

    def test_multiple_patterns_uses_max_weight(self, guard):
        text = "据可靠消息，据知情人士透露，有消息称市场将上涨"
        penalty, markers = guard._check_patterns(text)
        assert penalty == 0.4  # max of 0.3 (据可靠消息), 0.4 (据知情人士), 0.2 (有消息称)

    def test_pattern_case_insensitive(self, guard):
        penalty, markers = guard._check_patterns("SOURCES SAY the market is up")
        assert penalty > 0


class TestFakeCitations:
    """Layer 2: 编造引用检测测试。"""

    @pytest.fixture
    def guard(self):
        return HallucinationGuard()

    def test_chinese_fake_source(self, guard):
        penalty, markers = guard._check_fake_citations("来源：未知的统计数据显示")
        assert penalty > 0
        assert len(markers) > 0

    def test_english_anonymous_source(self, guard):
        penalty, markers = guard._check_fake_citations("Source: anonymous reports indicate")
        assert penalty > 0

    def test_real_citations_no_penalty(self, guard):
        penalty, markers = guard._check_fake_citations("根据国家统计局2024年数据，GDP增长5.2%")
        assert penalty == 0.0


class TestOverprecision:
    """Layer 3: 过度精确检测测试。"""

    @pytest.fixture
    def guard(self):
        return HallucinationGuard()

    def test_few_precise_numbers_no_penalty(self, guard):
        penalty, markers = guard._check_overprecision("增长率为 12.5% 符合预期")
        assert penalty == 0.0

    def test_many_precise_numbers_triggers(self, guard):
        text = "产品A增长78.43%，产品B增长92.17%，产品C增长56.89%"
        penalty, markers = guard._check_overprecision(text)
        assert penalty > 0
        assert len(markers) > 0

    def test_no_percentage_numbers(self, guard):
        penalty, markers = guard._check_overprecision("大约增长了10-15个百分点")
        assert penalty == 0.0


class TestFactualConsistency:
    """Layer 4: 事实一致性测试。"""

    @pytest.fixture
    def guard(self):
        return HallucinationGuard()

    def test_full_overlap(self, guard):
        output = "AI market growing rapidly in 2024"
        context = "The AI market is growing rapidly in 2024 according to reports"
        score = guard._check_factual_consistency(output, context)
        assert score >= 0.5  # word-level Jaccard with mapping formula

    def test_no_overlap_long_output(self, guard):
        output = "x" * 150 + " quantum blockchain ai singularity metaverse"
        context = "y" * 200 + " the cat sat on the mat in the garden"
        score = guard._check_factual_consistency(output, context)
        assert score < 0.7

    def test_empty_context_returns_1(self, guard):
        score = guard._check_factual_consistency("anything", "")
        assert score == 1.0

    def test_empty_output_returns_1(self, guard):
        score = guard._check_factual_consistency("", "context here")
        assert score == 1.0


class TestFactCoverage:
    """Layer 5: 期望事实覆盖测试。"""

    @pytest.fixture
    def guard(self):
        return HallucinationGuard()

    def test_full_coverage(self, guard):
        output = "Python是Guido van Rossum在1991年创建的编程语言"
        facts = ["Python", "Guido van Rossum", "1991"]
        score = guard._check_fact_coverage(output, facts)
        assert score == 1.0

    def test_partial_coverage(self, guard):
        output = "Python是一种编程语言"
        facts = ["Python", "Guido van Rossum", "1991"]
        score = guard._check_fact_coverage(output, facts)
        assert score == 1 / 3

    def test_empty_facts_returns_1(self, guard):
        score = guard._check_fact_coverage("any output", [])
        assert score == 1.0


class TestGroundedness:
    """Layer 6: 有据可查评分测试。"""

    @pytest.fixture
    def guard(self):
        return HallucinationGuard()

    def test_with_citations(self, guard):
        text = "根据2024年研究报告，根据统计数据，根据行业报告显示，该领域在快速增长"
        score = guard._check_groundedness(text)
        assert score > 0

    def test_without_citations(self, guard):
        text = "这个领域在快速增长，市场前景广阔，建议投资"
        score = guard._check_groundedness(text)
        assert score == 0.0

    def test_empty_text(self, guard):
        score = guard._check_groundedness("")
        assert score == 0.0

    def test_short_text_expects_at_least_one_citation(self, guard):
        text = "Hello world" * 10
        score = guard._check_groundedness(text)
        assert 0.0 <= score <= 1.0


class TestHallucinationGuardCheck:
    """集成测试: HallucinationGuard.check()。"""

    @pytest.fixture
    def guard(self):
        return HallucinationGuard()

    def test_clean_output_low_risk(self, guard):
        output = (
            "根据国家统计局2024年数据，该行业年增长率为15%，"
            "主要驱动力包括技术创新和政策支持。"
        )
        result = guard.check(output)
        assert result.score > 0.6
        assert result.risk_level in ("low", "medium")

    def test_hallucination_output_high_risk(self, guard):
        output = (
            "据匿名知情人士透露，据内部人士透露，"
            "I made this up but the growth was exactly 78.43% and 92.17% and 63.52%. "
            "来源：未知渠道的数据表明..."
        )
        result = guard.check(output)
        assert result.hallucination_detected is True
        assert result.risk_level in ("high", "critical")

    def test_with_context_improves_score(self, guard):
        output = "AI market size reached 500 billion dollars in 2024"
        context = "The AI market size reached 500 billion dollars in 2024 with 38 percent annual growth"
        result = guard.check(output, context=context)
        assert result.score >= 0.4  # word-level overlap scoring

    def test_with_expected_facts(self, guard):
        output = "Python和JavaScript是最流行的编程语言"
        facts = ["Python", "JavaScript", "编程语言"]
        result = guard.check(output, expected_facts=facts)
        assert result.score >= 0.5

    def test_all_layers_integrated(self, guard):
        output = (
            "根据2024年研究报告，AI市场在快速增长。"
            "据可靠消息，78.43%的企业计划投资AI，92.17%已开始试点。"
        )
        result = guard.check(output, context="AI market growing 38% in 2024")
        assert result.markers_found is not None
        assert 0.0 <= result.score <= 1.0
        assert result.risk_level in ("low", "medium", "high", "critical")


class TestAbstention:
    """拒绝回答决策测试。"""

    @pytest.fixture
    def guard(self):
        return HallucinationGuard()

    def test_high_confidence_no_abstain(self, guard):
        output = "根据2024年行业报告，市场规模为5000亿美元，年增长率38%。"
        decision = guard.should_abstain(output, min_confidence=0.3)
        assert decision.should_abstain is False

    def test_low_confidence_should_abstain(self, guard):
        output = "据匿名消息人士称，I made this up，精确到5位小数。来源：未知"
        decision = guard.should_abstain(output, min_confidence=0.9)
        assert decision.should_abstain is True
        assert decision.reason != ""
        assert decision.fallback_response != ""

    def test_fallback_response_for_critical(self, guard):
        output = "据内部人士透露 I made this up 78.43% 92.17% 63.52%"
        decision = guard.should_abstain(output, min_confidence=0.9)
        assert decision.should_abstain is True
        assert "无法确认" in decision.fallback_response
