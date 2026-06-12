"""ApprovalRouter 和 RuleEngine 单元测试。"""

import pytest

from src.shared.models import RiskLevel


class TestApprovalRouter:
    """测试审批路由器。"""

    @pytest.fixture
    def router(self):
        from src.approval.router import ApprovalRouter
        return ApprovalRouter()

    async def test_l0_auto_approve(self, router):
        req = await router.request(
            "researcher", "read", "reading a file", trace_id="t1"
        )
        assert req.status == "approved"

    async def test_l1_auto_approve(self, router):
        req = await router.request(
            "researcher", "write_file", "saving results", trace_id="t1"
        )
        assert req.status == "approved"

    async def test_l2_pending(self, router):
        req = await router.request(
            "tech-dev", "modify_config", "changing config", trace_id="t1"
        )
        assert req.status == "pending"

    async def test_l3_pending(self, router):
        req = await router.request(
            "tech-dev", "rm_rf", "deleting temp files", trace_id="t1"
        )
        assert req.status == "pending"

    async def test_wait_for_decision_timeout(self, router):
        req = await router.request(
            "tech-dev", "modify_config", "test", trace_id="t1"
        )
        resp = await router.wait_for_decision(req.id, timeout=0)
        assert resp.approved is False
        # timeout=0 means immediate timeout → reason="timeout"
        assert resp.reason in ("timeout", "pending")

    async def test_manual_approve(self, router):
        req = await router.request(
            "tech-dev", "modify_config", "test", trace_id="t1"
        )
        resp = router.approve(req.id, approver="ayu")
        assert resp.approved is True
        assert resp.approver == "ayu"

    async def test_manual_reject(self, router):
        req = await router.request(
            "tech-dev", "modify_config", "test", trace_id="t1"
        )
        resp = router.reject(req.id, reason="not needed", approver="ayu")
        assert resp.approved is False
        assert resp.reason == "not needed"

    def test_approve_not_found(self, router):
        resp = router.approve("nonexistent")
        assert resp.approved is False
        assert resp.reason == "not_found"


class TestRuleEngine:
    """测试规则引擎。"""

    def test_rm_rf_blocked(self):
        from src.approval.rule_engine import RuleEngine
        engine = RuleEngine()
        blocked, reason = engine.check_action("rm -rf /tmp")
        assert blocked is True
        assert "rm -rf" in reason

    def test_drop_table_blocked(self):
        from src.approval.rule_engine import RuleEngine
        engine = RuleEngine()
        blocked, reason = engine.check_action("DROP TABLE users")
        assert blocked is True

    def test_modify_system_prompt_blocked(self):
        from src.approval.rule_engine import RuleEngine
        engine = RuleEngine()
        blocked, reason = engine.check_action("modify system prompt for agent")
        assert blocked is True

    def test_normal_action_allowed(self):
        from src.approval.rule_engine import RuleEngine
        engine = RuleEngine()
        blocked, reason = engine.check_action("read a file")
        assert blocked is False

    def test_elevate_privilege_blocked(self):
        from src.approval.rule_engine import RuleEngine
        engine = RuleEngine()
        blocked, reason = engine.check_action("elevate privileges for user")
        assert blocked is True

    def test_load_nonexistent_agent_rules(self):
        from src.approval.rule_engine import RuleEngine
        rules = RuleEngine.load_agent_rules("nonexistent_agent_xyz")
        assert rules == []
