"""HookManager tests — 9 hook points, legacy API compat, decorator registration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.shared.models import HookDecision, RiskLevel


class TestHookManagerLegacy:
    """Test legacy API compatibility."""

    @pytest.fixture
    def hooks(self):
        from src.safety.hooks import HookManager

        with patch("src.safety.hooks.AuditLogger") as mock_audit, \
             patch("src.safety.hooks.CostTracker") as mock_cost:
            mgr = HookManager()
            mgr._audit = mock_audit.return_value
            mgr._cost = mock_cost.return_value
            yield mgr

    async def test_read_operation_allowed(self, hooks):
        hooks._cost.check_budget.return_value = {
            "tokens_used": 100, "tokens_remaining": 49900, "is_exceeded": False,
        }
        result = await hooks.run_pre_tool_use(
            "researcher", "read", {}, trace_id="t1"
        )
        assert result == HookDecision.ALLOW

    async def test_l3_operation_escalated(self, hooks):
        hooks._cost.check_budget.return_value = {
            "tokens_used": 100, "tokens_remaining": 49900, "is_exceeded": False,
        }
        result = await hooks.run_pre_tool_use(
            "tech-dev", "rm_rf", {"path": "/tmp/test"}, trace_id="t1"
        )
        assert result == HookDecision.ESCALATE

    async def test_budget_exceeded_blocks_l2(self, hooks):
        hooks._cost.check_budget.return_value = {
            "tokens_used": 50000, "tokens_remaining": 0, "is_exceeded": True,
        }
        result = await hooks.run_pre_tool_use(
            "researcher", "call_paid_api", {}, trace_id="t1"
        )
        assert result == HookDecision.BLOCK

    async def test_pii_in_params_blocked(self, hooks):
        hooks._cost.check_budget.return_value = {
            "tokens_used": 100, "tokens_remaining": 49900, "is_exceeded": False,
        }
        result = await hooks.run_pre_tool_use(
            "researcher", "write_file",
            {"content": "phone: 13800138000"}, trace_id="t1"
        )
        assert result == HookDecision.BLOCK

    async def test_post_tool_use_prompt_leak_blocked(self, hooks):
        result = await hooks.run_post_tool_use(
            "researcher", "read",
            "the system prompt is: you are a helpful...", trace_id="t1"
        )
        assert result == HookDecision.BLOCK

    async def test_post_tool_use_pii_blocked(self, hooks):
        result = await hooks.run_post_tool_use(
            "researcher", "read",
            "here is the key: sk-abc123def456ghi789jklmno", trace_id="t2"
        )
        assert result == HookDecision.BLOCK

    async def test_post_tool_use_normal_result(self, hooks):
        result = await hooks.run_post_tool_use(
            "researcher", "read",
            "normal search results here", trace_id="t1"
        )
        assert result == HookDecision.ALLOW

    async def test_on_stop_records_cost(self, hooks):
        await hooks.run_on_stop(
            "researcher", "completed",
            {"tokens": 500}, trace_id="t1"
        )
        hooks._cost.record.assert_called_once_with("researcher", 500)

    def test_risk_mapping(self):
        from src.safety.hooks import HookManager
        assert HookManager._get_risk("read") == RiskLevel.L0
        assert HookManager._get_risk("write_file") == RiskLevel.L1
        assert HookManager._get_risk("modify_config") == RiskLevel.L2
        assert HookManager._get_risk("rm_rf") == RiskLevel.L3

    def test_unknown_tool_defaults_to_l1(self):
        from src.safety.hooks import HookManager
        assert HookManager._get_risk("unknown_custom_tool") == RiskLevel.L1


class TestHookManagerExtended:
    """Test new HookPoint/HookContext/HookResult system."""

    @pytest.fixture
    def hooks(self):
        from src.safety.hooks import HookManager

        with patch("src.safety.hooks.AuditLogger"), \
             patch("src.safety.hooks.CostTracker"):
            return HookManager()

    # ── Registration ──────────────────────────────────

    def test_decorator_registration(self, hooks):
        from src.safety.hooks import HookPoint, HookAction, HookResult

        @hooks.on(HookPoint.PRE_TOOL_USE)
        def my_handler(ctx):
            return HookResult(action=HookAction.ALLOW)

        assert len(hooks._handlers[HookPoint.PRE_TOOL_USE]) == 1

    def test_convenience_registration_methods(self, hooks):
        from src.safety.hooks import HookPoint, HookAction, HookResult

        hooks.pre_tool_use(lambda ctx: HookResult(action=HookAction.ALLOW))
        hooks.post_tool_use(lambda ctx: HookResult(action=HookAction.ALLOW))
        hooks.on_error(lambda ctx: HookResult(action=HookAction.NOTIFY))
        hooks.on_approval(lambda ctx: HookResult(action=HookAction.ALLOW))
        hooks.pre_llm_call(lambda ctx: HookResult(action=HookAction.ALLOW))
        hooks.post_llm_call(lambda ctx: HookResult(action=HookAction.ALLOW))

        assert len(hooks._handlers[HookPoint.PRE_TOOL_USE]) == 1
        assert len(hooks._handlers[HookPoint.POST_TOOL_USE]) == 1
        assert len(hooks._handlers[HookPoint.ON_ERROR]) == 1
        assert len(hooks._handlers[HookPoint.ON_APPROVAL]) == 1
        assert len(hooks._handlers[HookPoint.PRE_LLM_CALL]) == 1
        assert len(hooks._handlers[HookPoint.POST_LLM_CALL]) == 1

    # ── Handler execution ─────────────────────────────

    async def test_handler_blocks_when_returning_block(self, hooks):
        from src.safety.hooks import HookPoint, HookAction, HookResult

        hooks.pre_tool_use(lambda ctx: HookResult(action=HookAction.BLOCK, block_reason="test"))

        result = await hooks.run_pre_tool_use("agent", "read", {}, trace_id="t1")
        assert result == HookDecision.BLOCK

    async def test_handler_pause_returns_escalate(self, hooks):
        from src.safety.hooks import HookPoint, HookAction, HookResult

        hooks.pre_tool_use(lambda ctx: HookResult(action=HookAction.PAUSE, pause_reason="need human"))

        result = await hooks.run_pre_tool_use("agent", "search", {}, trace_id="t1")
        assert result == HookDecision.ESCALATE

    async def test_multiple_handlers_first_block_wins(self, hooks):
        from src.safety.hooks import HookPoint, HookAction, HookResult

        calls = []
        hooks.pre_tool_use(lambda ctx: calls.append("first") or HookResult(action=HookAction.BLOCK))
        hooks.pre_tool_use(lambda ctx: calls.append("second") or HookResult(action=HookAction.ALLOW))

        result = await hooks.run_pre_tool_use("agent", "read", {}, trace_id="t1")
        assert result == HookDecision.BLOCK
        assert calls == ["first"]  # second never runs

    async def test_handler_accepts_hook_context(self, hooks):
        from src.safety.hooks import HookPoint, HookAction, HookResult, HookContext

        received = []

        @hooks.on(HookPoint.PRE_TOOL_USE)
        def check_ctx(ctx):
            assert isinstance(ctx, HookContext)
            received.append((ctx.agent_id, ctx.tool_name))
            return HookResult(action=HookAction.ALLOW)

        await hooks.run_pre_tool_use("researcher", "search", {"q": "test"}, trace_id="xyz")
        assert received == [("researcher", "search")]

    # ── Extended hook points ──────────────────────────

    async def test_run_on_error(self, hooks):
        from src.safety.hooks import HookAction, HookResult

        hooks.on_error(lambda ctx: HookResult(action=HookAction.NOTIFY, notification_msg="error!"))

        result = await hooks.run_on_error("agent", ValueError("test error"), trace_id="t1")
        assert result.action == HookAction.NOTIFY

    async def test_run_on_approval(self, hooks):
        from src.safety.hooks import HookAction

        result = await hooks.run_on_approval("agent", "delete_file", trace_id="t1")
        assert result.action == HookAction.ALLOW  # default when no handler blocks

    async def test_run_pre_llm_call(self, hooks):
        from src.safety.hooks import HookAction

        result = await hooks.run_pre_llm_call("agent", "gpt-4", "hello", trace_id="t1")
        assert result.action == HookAction.ALLOW

    async def test_run_post_llm_call(self, hooks):
        from src.safety.hooks import HookAction

        result = await hooks.run_post_llm_call("agent", "gpt-4", "response", trace_id="t1")
        assert result.action == HookAction.ALLOW

    async def test_run_on_step_start_end(self, hooks):
        from src.safety.hooks import HookAction

        r1 = await hooks.run_on_step_start("agent", "s1", "search step", trace_id="t1")
        r2 = await hooks.run_on_step_end("agent", "s1", "search step", "results", trace_id="t1")
        assert r1.action == HookAction.ALLOW
        assert r2.action == HookAction.ALLOW

    async def test_handler_can_return_dict(self, hooks):
        from src.safety.hooks import HookPoint, HookAction

        hooks.pre_tool_use(lambda ctx: {"action": HookAction.BLOCK, "block_reason": "dict test"})

        result = await hooks.run_pre_tool_use("agent", "read", {}, trace_id="t1")
        assert result == HookDecision.BLOCK

    # ── Stats ─────────────────────────────────────────

    async def test_get_stats(self, hooks):
        stats = hooks.get_stats()
        assert "handlers_registered" in stats
        assert "total_executions" in stats
        assert "blocks" in stats
        assert "allows" in stats

    async def test_clear_log(self, hooks):
        from src.safety.hooks import HookAction

        await hooks.run_pre_llm_call("agent", "m", "p", trace_id="t1")
        assert hooks._allow_count >= 0
        hooks.clear_log()
        assert hooks._allow_count == 0
        assert len(hooks._execution_log) == 0


class TestHookDataStructures:
    """Test HookPoint, HookAction, HookContext, HookResult."""

    def test_hook_point_enum(self):
        from src.safety.hooks import HookPoint
        assert HookPoint.PRE_TOOL_USE.value == "pre_tool_use"
        assert HookPoint.ON_ERROR.value == "on_error"
        assert HookPoint.PRE_LLM_CALL.value == "pre_llm_call"
        assert HookPoint.POST_LLM_CALL.value == "post_llm_call"
        assert len(list(HookPoint)) == 9

    def test_hook_action_enum(self):
        from src.safety.hooks import HookAction
        assert HookAction.ALLOW.value == "allow"
        assert HookAction.BLOCK.value == "block"
        assert HookAction.TRANSFORM.value == "transform"
        assert HookAction.PAUSE.value == "pause"
        assert HookAction.NOTIFY.value == "notify"
        assert len(list(HookAction)) == 5

    def test_hook_context_defaults(self):
        from src.safety.hooks import HookContext
        ctx = HookContext()
        assert ctx.agent_id == ""
        assert ctx.tool_name == ""
        assert ctx.error is None

    def test_hook_context_with_data(self):
        from src.safety.hooks import HookContext
        ctx = HookContext(
            agent_id="agent-1",
            tool_name="search",
            tool_params={"q": "test"},
            trace_id="trace-1",
        )
        assert ctx.agent_id == "agent-1"
        assert ctx.tool_params == {"q": "test"}

    def test_hook_result_defaults(self):
        from src.safety.hooks import HookResult, HookAction
        r = HookResult()
        assert r.action == HookAction.ALLOW
        assert r.blocked is False

    def test_hook_result_block(self):
        from src.safety.hooks import HookResult, HookAction
        r = HookResult(action=HookAction.BLOCK, blocked=True, block_reason="unsafe")
        assert r.action == HookAction.BLOCK
        assert r.block_reason == "unsafe"

    def test_operation_risk_keys(self):
        from src.safety.hooks import OPERATION_RISK
        assert "read" in OPERATION_RISK
        assert "rm_rf" in OPERATION_RISK
        assert "modify_system_prompt" in OPERATION_RISK
