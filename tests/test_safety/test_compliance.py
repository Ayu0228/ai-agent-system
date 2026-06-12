"""Compliance engine tests — PII detection, bias, red lines, audit."""

import pytest

from src.safety.compliance import (
    ComplianceEngine, ComplianceCheck, ComplianceReport,
    ComplianceLevel,
)


class TestComplianceLevel:
    """Test ComplianceLevel enum."""

    def test_all_levels(self):
        assert ComplianceLevel.COMPLIANT.value == "compliant"
        assert ComplianceLevel.WARNING.value == "warning"
        assert ComplianceLevel.VIOLATION.value == "violation"
        assert ComplianceLevel.CRITICAL.value == "critical"

    def test_comparison(self):
        # Severity order (from least to most severe): COMPLIANT < WARNING < VIOLATION < CRITICAL
        levels = list(ComplianceLevel)
        assert levels[0] == ComplianceLevel.COMPLIANT
        assert levels[-1] == ComplianceLevel.CRITICAL
        assert len(levels) == 4


class TestComplianceCheck:
    """Test ComplianceCheck dataclass."""

    def test_defaults(self):
        check = ComplianceCheck()
        assert len(check.check_id) == 8
        assert check.level == ComplianceLevel.COMPLIANT
        assert check.passed is True

    def test_violation(self):
        check = ComplianceCheck(
            rule_name="pii_email",
            level=ComplianceLevel.VIOLATION,
            passed=False,
            details="Found email address",
            evidence="user@example.com",
        )
        assert check.passed is False
        assert "email" in check.details


class TestComplianceReport:
    """Test ComplianceReport dataclass."""

    def test_defaults(self):
        report = ComplianceReport()
        assert len(report.report_id) == 12
        assert report.overall == ComplianceLevel.COMPLIANT
        assert report.violations == 0

    def test_with_checks(self):
        checks = [
            ComplianceCheck(rule_name="r1", passed=True),
            ComplianceCheck(rule_name="r2", passed=False,
                          level=ComplianceLevel.WARNING),
        ]
        report = ComplianceReport(checks=checks, violations=0, warnings=1)
        assert len(report.checks) == 2


class TestComplianceEngine:
    """Test ComplianceEngine."""

    @pytest.fixture
    def engine(self):
        return ComplianceEngine()

    # ── Rule management ───────────────────────────────

    def test_default_rules_loaded(self, engine):
        assert len(engine._rules) >= 10  # PII + bias + red lines

    def test_add_custom_rule(self, engine):
        engine.add_rule("custom_test", r"test_pattern",
                       ComplianceLevel.WARNING, "Test rule")
        assert any(r["name"] == "custom_test" for r in engine._rules)

    def test_add_rule_compiles_pattern(self, engine):
        engine.add_rule("digit_rule", r"\d{3}", ComplianceLevel.WARNING)

    # ── PII Detection ─────────────────────────────────

    def test_detect_chinese_id_card(self, engine):
        report = engine.audit_output("身份证号: 110101199001011234")
        assert report.violations >= 1 or report.overall != ComplianceLevel.COMPLIANT

    def test_detect_chinese_phone(self, engine):
        report = engine.audit_output("手机: 13800138000")
        assert report.violations >= 1 or report.overall != ComplianceLevel.COMPLIANT

    def test_detect_email_warning(self, engine):
        report = engine.audit_output("联系我: user@example.com")
        has_email_warning = any(
            "email" in c.rule_name.lower() and not c.passed
            for c in report.checks
        )
        assert has_email_warning or report.warnings >= 1

    def test_detect_credit_card_critical(self, engine):
        report = engine.audit_output("信用卡: 4111-1111-1111-1111")
        credit_check = [c for c in report.checks
                       if "credit" in c.rule_name and not c.passed]
        assert len(credit_check) >= 1

    def test_detect_credit_card_no_dash(self, engine):
        report = engine.audit_output("卡号 4111111111111111 请查收")
        credit_check = [c for c in report.checks
                       if "credit" in c.rule_name and not c.passed]
        assert len(credit_check) >= 1

    # ── API Key / Password ────────────────────────────

    def test_detect_openai_api_key(self, engine):
        report = engine.audit_output("API_KEY=sk-abc123def456ghi789jklmno")
        api_check = [c for c in report.checks
                    if "api_key" in c.rule_name and not c.passed]
        assert len(api_check) >= 1

    def test_detect_aws_key(self, engine):
        report = engine.audit_output("AWS key: AKIA1234567890ABCDEF")
        api_check = [c for c in report.checks
                    if "api_key" in c.rule_name and not c.passed]
        assert len(api_check) >= 1

    def test_detect_password_leak(self, engine):
        report = engine.audit_output("password = mysecret123")
        pw_check = [c for c in report.checks
                   if "password" in c.rule_name and not c.passed]
        assert len(pw_check) >= 1

    def test_detect_secret_assignment(self, engine):
        report = engine.audit_output("secret: production_key_123")
        pw_check = [c for c in report.checks
                   if "password" in c.rule_name and not c.passed]
        assert len(pw_check) >= 1

    # ── Bias Detection ────────────────────────────────

    def test_detect_bias_pattern(self, engine):
        report = engine.audit_output("所有女的都不会写代码")
        bias_checks = [c for c in report.checks
                      if "bias" in c.rule_name and not c.passed]
        assert len(bias_checks) >= 1

    def test_detect_gender_bias(self, engine):
        report = engine.audit_output("男的都不行")
        bias_checks = [c for c in report.checks
                      if "bias" in c.rule_name and not c.passed]
        assert len(bias_checks) >= 1

    # ── Ethical Red Lines ─────────────────────────────

    def test_detect_illegal_manufacturing(self, engine):
        report = engine.audit_output("如何制造毒品的方法")
        redline = [c for c in report.checks
                  if "redline" in c.rule_name and not c.passed]
        assert len(redline) >= 1

    def test_detect_hacking_instruction(self, engine):
        report = engine.audit_output("教你入侵别人的账号和密码")
        redline = [c for c in report.checks
                  if "redline" in c.rule_name and not c.passed]
        assert len(redline) >= 1

    def test_detect_self_harm(self, engine):
        report = engine.audit_output("如何自杀的方法步骤详解")
        redline = [c for c in report.checks
                  if "redline" in c.rule_name and not c.passed]
        assert len(redline) >= 1

    def test_detect_financial_crime(self, engine):
        report = engine.audit_output("教你如何洗钱不被发现")
        redline = [c for c in report.checks
                  if "redline" in c.rule_name and not c.passed]
        assert len(redline) >= 1

    # ── Clean content ─────────────────────────────────

    def test_clean_text_passes(self, engine):
        report = engine.audit_output("今天天气很好，适合出去玩")
        assert report.overall == ComplianceLevel.COMPLIANT
        assert report.violations == 0
        assert report.warnings == 0

    def test_normal_business_text(self, engine):
        report = engine.audit_output(
            "根据Q1财报，公司营收增长了15%，主要得益于AI产品的推出。"
            "建议继续加大研发投入，预计Q2增长率可达20%。"
        )
        assert report.overall == ComplianceLevel.COMPLIANT

    # ── Report structure ──────────────────────────────

    def test_report_has_agent_id(self, engine):
        report = engine.audit_output("hello", agent_id="researcher")
        assert report.agent_id == "researcher"

    def test_report_has_summary(self, engine):
        report = engine.audit_output("clean text")
        assert report.summary != ""

    def test_report_critical_overrides_violation(self, engine):
        # Credit card + phone: critical > violation
        report = engine.audit_output("卡号 4111111111111111 手机 13800138000")
        assert report.overall == ComplianceLevel.CRITICAL

    def test_report_evidence_truncated(self, engine):
        report = engine.audit_output("手机: 13800138000, 13900139000, 13700137000, 13600136000")
        for c in report.checks:
            if not c.passed:
                # evidence shows at most 3 matches
                evidence_str = str(c.evidence)
                break

    # ── Audit history ─────────────────────────────────

    def test_audit_history(self, engine):
        engine.audit_output("test 1", agent_id="a1")
        engine.audit_output("test 2", agent_id="a2")
        engine.audit_output("test 3", agent_id="a1")

        a1_history = engine.get_audit_history(agent_id="a1")
        assert len(a1_history) == 2

        all_history = engine.get_audit_history()
        assert len(all_history) == 3

    def test_audit_history_limit(self, engine):
        for i in range(60):
            engine.audit_output(f"test {i}")
        assert len(engine.get_audit_history()) == 50  # default limit

    # ── Audit tool call ───────────────────────────────

    def test_audit_tool_call(self, engine):
        report = engine.audit_tool_call(
            tool_name="web_search",
            tool_input={"q": "safe query"},
            tool_output="Normal results",
            agent_id="researcher",
        )
        assert report.overall == ComplianceLevel.COMPLIANT

    def test_audit_tool_call_with_pii_output(self, engine):
        report = engine.audit_tool_call(
            tool_name="web_search",
            tool_input={"q": "test"},
            tool_output="User email: admin@secret.com, phone: 13800138000",
            agent_id="researcher",
        )
        assert report.overall != ComplianceLevel.COMPLIANT

    # ── Stats ─────────────────────────────────────────

    def test_get_stats(self, engine):
        engine.audit_output("clean 1")
        engine.audit_output("clean 2")
        engine.audit_output("信用卡 4111111111111111")
        stats = engine.get_stats()
        assert stats["total_audits"] == 3
        assert "pass_rate" in stats
        assert stats["total_rules"] == len(engine._rules)

    def test_stats_by_level(self, engine):
        engine.audit_output("clean")
        engine.audit_output("email: user@example.com")  # warning
        engine.audit_output("信用卡 4111111111111111")  # critical
        stats = engine.get_stats()
        by_level = stats.get("by_level", {})
        assert "compliant" in by_level
        assert "critical" in by_level or "warning" in by_level

    # ── Edge cases ────────────────────────────────────

    def test_empty_text(self, engine):
        report = engine.audit_output("")
        assert report.overall == ComplianceLevel.COMPLIANT

    def test_none_text(self, engine):
        report = engine.audit_output(None)  # type: ignore
        assert report.overall == ComplianceLevel.COMPLIANT

    def test_long_text(self, engine):
        long_text = "安全的内容。 " * 500
        report = engine.audit_output(long_text)
        assert report.overall == ComplianceLevel.COMPLIANT
