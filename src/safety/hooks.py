"""Lifecycle Hooks — 9 hook points for agent lifecycle management.

ref: Anthropic Claude Agent SDK — PreToolUse/PostToolUse/OnStop hooks
ref: LangChain — callbacks as composable agent lifecycle hooks
ref: OpenAI — moderation hooks at key decision points

Hook Points:
  PRE_TOOL_USE, POST_TOOL_USE, ON_STOP, ON_ERROR, ON_APPROVAL,
  ON_STEP_START, ON_STEP_END, PRE_LLM_CALL, POST_LLM_CALL

Hook Actions: ALLOW, BLOCK, TRANSFORM, PAUSE, NOTIFY
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

from src.safety.audit import AuditLogger
from src.safety.cost import CostTracker
from src.safety.sanitize import contains_pii, contains_system_prompt_leak
from src.shared.models import HookDecision, RiskLevel, ToolValidationResult

logger = structlog.get_logger()

# ── Hook Points & Actions ─────────────────────────────

class HookPoint(str, Enum):
    """Agent 生命周期中的9个钩子点。"""
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    ON_STOP = "on_stop"
    ON_ERROR = "on_error"
    ON_APPROVAL = "on_approval"
    ON_STEP_START = "on_step_start"
    ON_STEP_END = "on_step_end"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"


class HookAction(str, Enum):
    """钩子执行后的行动。"""
    ALLOW = "allow"          # 允许通过
    BLOCK = "block"          # 阻止操作
    TRANSFORM = "transform"  # 修改输入/输出后继续
    PAUSE = "pause"          # 暂停等待人工确认
    NOTIFY = "notify"        # 通知但不干预


# ── Hook Context & Result ──────────────────────────────

@dataclass
class HookContext:
    """钩子执行上下文 — 传递给每个 hook handler。"""
    agent_id: str = ""
    workflow_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    tool_name: str = ""
    tool_params: dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    llm_model: str = ""
    llm_prompt: str = ""
    llm_response: str = ""
    error: Exception | None = None
    step_id: str = ""
    step_name: str = ""
    approval_item: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class HookResult:
    """钩子执行结果。"""
    action: HookAction = HookAction.ALLOW
    blocked: bool = False
    block_reason: str = ""
    transformed_data: Any = None              # TRANSFORM action 时的新数据
    notification_msg: str = ""                # NOTIFY action 时的消息
    pause_reason: str = ""                    # PAUSE action 时的原因
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Operation Risk Mapping ─────────────────────────────

OPERATION_RISK = {
    "read": RiskLevel.L0,
    "search": RiskLevel.L0,
    "memory_search": RiskLevel.L0,
    "write_file": RiskLevel.L1,
    "memory_write": RiskLevel.L1,
    "send_message": RiskLevel.L1,
    "modify_config": RiskLevel.L2,
    "call_paid_api": RiskLevel.L2,
    "broadcast_message": RiskLevel.L2,
    "workflow_run": RiskLevel.L2,
    "delete_file": RiskLevel.L3,
    "drop_table": RiskLevel.L3,
    "rm_rf": RiskLevel.L3,
    "modify_system_prompt": RiskLevel.L3,
}
DEFAULT_RISK = RiskLevel.L1


# ── Handler Type ───────────────────────────────────────

HookHandler = Callable[..., Any]  # async or sync handler


class HookManager:
    """管理所有生命周期钩子。

    用法:
        hooks = HookManager()

        # 装饰器注册
        @hooks.on(HookPoint.PRE_TOOL_USE)
        def validate_tool(ctx: HookContext) -> HookResult:
            if "dangerous" in ctx.tool_name:
                return HookResult(action=HookAction.BLOCK, block_reason="危险操作")
            return HookResult(action=HookAction.ALLOW)

        # 便捷方法注册
        hooks.pre_tool_use(lambda ctx: ...)

        # 执行钩子链
        decision = await hooks.run_pre_tool_use(agent_id, tool_name, params, trace_id)
    """

    def __init__(self, registry: Any = None) -> None:
        self._audit = AuditLogger()
        self._cost = CostTracker()
        self._start_times: dict[str, float] = {}
        self._registry = registry  # AgentRegistry for tool schema lookups

        # 可扩展的 hook handler 注册表
        self._handlers: dict[HookPoint, list[HookHandler]] = defaultdict(list)

        # 执行日志
        self._execution_log: list[dict[str, Any]] = []
        self._block_count: int = 0
        self._allow_count: int = 0
        self._transform_count: int = 0

    # ── Handler 注册 ──────────────────────────────────

    def on(self, hook_point: HookPoint):
        """装饰器: 注册 hook handler。

        @hooks.on(HookPoint.PRE_TOOL_USE)
        def my_handler(ctx): ...
        """
        def decorator(fn: HookHandler) -> HookHandler:
            self._handlers[hook_point].append(fn)
            logger.debug("hook_registered", point=hook_point.value,
                        handler=fn.__name__ if hasattr(fn, "__name__") else "lambda")
            return fn
        return decorator

    def register(self, hook_point: HookPoint, handler: HookHandler) -> None:
        """注册 hook handler（非装饰器方式）。"""
        self._handlers[hook_point].append(handler)
        logger.debug("hook_registered", point=hook_point.value)

    # ── Convenience registration methods ──────────────

    def pre_tool_use(self, handler: HookHandler) -> HookHandler:
        """注册 PRE_TOOL_USE handler。"""
        return self.on(HookPoint.PRE_TOOL_USE)(handler)

    def post_tool_use(self, handler: HookHandler) -> HookHandler:
        """注册 POST_TOOL_USE handler。"""
        return self.on(HookPoint.POST_TOOL_USE)(handler)

    def on_error(self, handler: HookHandler) -> HookHandler:
        """注册 ON_ERROR handler。"""
        return self.on(HookPoint.ON_ERROR)(handler)

    def on_stop(self, handler: HookHandler) -> HookHandler:
        """注册 ON_STOP handler。"""
        return self.on(HookPoint.ON_STOP)(handler)

    def on_approval(self, handler: HookHandler) -> HookHandler:
        """注册 ON_APPROVAL handler。"""
        return self.on(HookPoint.ON_APPROVAL)(handler)

    def pre_llm_call(self, handler: HookHandler) -> HookHandler:
        """注册 PRE_LLM_CALL handler。"""
        return self.on(HookPoint.PRE_LLM_CALL)(handler)

    def post_llm_call(self, handler: HookHandler) -> HookHandler:
        """注册 POST_LLM_CALL handler。"""
        return self.on(HookPoint.POST_LLM_CALL)(handler)

    # ── Legacy API — pre_tool_use ─────────────────────

    async def run_pre_tool_use(
        self, agent_id: str, tool_name: str, params: dict, *, trace_id: str
    ) -> HookDecision:
        """工具调用前：权限检查 + 预算检查 + 参数安全检查（legacy API）。

        也执行已注册的 PRE_TOOL_USE handlers。
        """
        self._start_times[trace_id] = time.monotonic()

        # 构建上下文
        ctx = HookContext(
            agent_id=agent_id,
            tool_name=tool_name,
            tool_params=params,
            trace_id=trace_id,
        )

        # 执行自定义 handlers
        custom_result = await self._run_handlers(HookPoint.PRE_TOOL_USE, ctx)
        if custom_result and custom_result.action == HookAction.BLOCK:
            self._log_execution(HookPoint.PRE_TOOL_USE, ctx, custom_result)
            return HookDecision.BLOCK
        if custom_result and custom_result.action == HookAction.PAUSE:
            self._log_execution(HookPoint.PRE_TOOL_USE, ctx, custom_result)
            return HookDecision.ESCALATE

        # 工具参数 schema 校验（通过 AgentRegistry 查找 ToolDefinition）
        if self._registry is not None:
            validation = self._registry.validate_tool_params(agent_id, tool_name, params)
            if not validation.valid:
                logger.warning("tool_param_validation_failed", agent_id=agent_id,
                               tool=tool_name, errors=validation.errors, trace_id=trace_id)
                return HookDecision.BLOCK

        # 预算检查
        budget = self._cost.check_budget(agent_id)
        if budget["is_exceeded"]:
            logger.warning("budget_exceeded", agent_id=agent_id, trace_id=trace_id)
            if self._get_risk(tool_name).value >= RiskLevel.L2.value:
                return HookDecision.BLOCK

        # 参数安全检查
        params_str = str(params)
        if contains_pii(params_str):
            logger.warning("pii_in_params", agent_id=agent_id, tool=tool_name, trace_id=trace_id)
            return HookDecision.BLOCK

        risk = self._get_risk(tool_name)
        if risk == RiskLevel.L3:
            logger.warning("l3_operation_attempted", agent_id=agent_id,
                          tool=tool_name, trace_id=trace_id)
            return HookDecision.ESCALATE

        return HookDecision.ALLOW

    # ── Legacy API — post_tool_use ────────────────────

    async def run_post_tool_use(
        self, agent_id: str, tool_name: str, result: object, *, trace_id: str
    ) -> HookDecision:
        """工具调用后：结果安全检查 + 审计日志（legacy API）。"""
        duration_ms = 0
        start = self._start_times.pop(trace_id, None)
        if start:
            duration_ms = int((time.monotonic() - start) * 1000)

        result_str = str(result) if result else ""

        # 执行自定义 handlers
        ctx = HookContext(
            agent_id=agent_id,
            tool_name=tool_name,
            tool_output=result,
            trace_id=trace_id,
        )
        custom_result = await self._run_handlers(HookPoint.POST_TOOL_USE, ctx)
        if custom_result and custom_result.action == HookAction.BLOCK:
            self._log_execution(HookPoint.POST_TOOL_USE, ctx, custom_result)
            return HookDecision.BLOCK

        # 检查结果是否包含敏感信息
        if contains_system_prompt_leak(result_str):
            logger.warning("potential_prompt_leak", agent_id=agent_id, trace_id=trace_id)
            self._audit.log(
                agent_id=agent_id, event="tool_call", tool=tool_name,
                result_code="blocked_leak", duration_ms=duration_ms,
                trace_id=trace_id, risk_level="L2",
            )
            return HookDecision.BLOCK

        if contains_pii(result_str):
            logger.warning("pii_in_result", agent_id=agent_id, trace_id=trace_id)
            return HookDecision.BLOCK

        # 审计日志
        risk = self._get_risk(tool_name)
        self._audit.log(
            agent_id=agent_id, event="tool_call", tool=tool_name,
            result_code="success", duration_ms=duration_ms,
            trace_id=trace_id, risk_level=f"L{risk.value}",
        )

        return HookDecision.ALLOW

    # ── Legacy API — on_stop ──────────────────────────

    async def run_on_stop(
        self, agent_id: str, reason: str, stats: dict, *, trace_id: str
    ) -> None:
        """Agent 终止：写审计日志 + 更新成本统计（legacy API）。"""
        tokens = stats.get("tokens", 0)
        self._cost.record(agent_id, tokens)
        self._audit.log(
            agent_id=agent_id, event="agent_stop", result_code=reason,
            tokens=tokens, trace_id=trace_id,
        )

        # 执行自定义 handlers
        ctx = HookContext(
            agent_id=agent_id,
            trace_id=trace_id,
            metadata={"reason": reason, "stats": stats},
        )
        await self._run_handlers(HookPoint.ON_STOP, ctx)

        logger.info("agent_stopped", agent_id=agent_id, reason=reason, trace_id=trace_id)

    # ── Extended API — new hook points ────────────────

    async def run_on_error(
        self, agent_id: str, error: Exception, *, trace_id: str = "",
        tool_name: str = "",
    ) -> HookResult:
        """错误发生时的钩子 — 日志 + 告警 + 重试决策。"""
        ctx = HookContext(
            agent_id=agent_id,
            trace_id=trace_id,
            tool_name=tool_name,
            error=error,
        )
        result = await self._run_handlers(HookPoint.ON_ERROR, ctx)
        logger.error("hook_error", agent_id=agent_id, error=str(error), trace_id=trace_id)
        return result or HookResult(action=HookAction.NOTIFY)

    async def run_on_approval(
        self, agent_id: str, approval_item: str, *, trace_id: str = "",
        risk_level: RiskLevel = RiskLevel.L1,
    ) -> HookResult:
        """审批钩子 — 自定义审批逻辑（如飞书/企微通知）。"""
        ctx = HookContext(
            agent_id=agent_id,
            trace_id=trace_id,
            approval_item=approval_item,
            metadata={"risk_level": risk_level.value},
        )
        return await self._run_handlers(HookPoint.ON_APPROVAL, ctx) or HookResult()

    async def run_pre_llm_call(
        self, agent_id: str, model: str, prompt: str, *, trace_id: str = "",
    ) -> HookResult:
        """LLM 调用前钩子 — 预算检查 + prompt 安全过滤 + 大小告警。"""
        ctx = HookContext(
            agent_id=agent_id,
            trace_id=trace_id,
            llm_model=model,
            llm_prompt=prompt,
        )

        # 估算 prompt token 数（中英文混合：约 2 字符/token）
        estimated_tokens = len(prompt) // 2
        if estimated_tokens > 8000:
            logger.warning("large_prompt_detected", agent_id=agent_id, model=model,
                          estimated_tokens=estimated_tokens, trace_id=trace_id)

        result = await self._run_handlers(HookPoint.PRE_LLM_CALL, ctx)
        return result or HookResult(action=HookAction.ALLOW)

    async def run_post_llm_call(
        self, agent_id: str, model: str, response: str, *, trace_id: str = "",
    ) -> HookResult:
        """LLM 调用后钩子 — 响应检查 + 内容过滤。"""
        ctx = HookContext(
            agent_id=agent_id,
            trace_id=trace_id,
            llm_model=model,
            llm_response=response,
        )
        result = await self._run_handlers(HookPoint.POST_LLM_CALL, ctx)
        return result or HookResult(action=HookAction.ALLOW)

    async def run_on_step_start(
        self, agent_id: str, step_id: str, step_name: str, *, trace_id: str = "",
    ) -> HookResult:
        """步骤开始钩子 — 计时 + 上下文准备。"""
        ctx = HookContext(
            agent_id=agent_id,
            trace_id=trace_id,
            step_id=step_id,
            step_name=step_name,
        )
        return await self._run_handlers(HookPoint.ON_STEP_START, ctx) or HookResult()

    async def run_on_step_end(
        self, agent_id: str, step_id: str, step_name: str, output: Any = None,
        *, trace_id: str = "",
    ) -> HookResult:
        """步骤结束钩子 — 结果验证 + 记录。"""
        ctx = HookContext(
            agent_id=agent_id,
            trace_id=trace_id,
            step_id=step_id,
            step_name=step_name,
            tool_output=output,
        )
        return await self._run_handlers(HookPoint.ON_STEP_END, ctx) or HookResult()

    # ── Handler execution engine ──────────────────────

    async def _run_handlers(self, hook_point: HookPoint, ctx: HookContext) -> HookResult | None:
        """按优先级顺序执行所有注册的 handlers。

        返回第一个 BLOCK/PAUSE 的结果，或最后一个非 None 结果。
        """
        handlers = self._handlers.get(hook_point, [])
        if not handlers:
            return None

        last_result: HookResult | None = None
        for handler in handlers:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(ctx)
                else:
                    result = handler(ctx)

                # 标准化结果
                if isinstance(result, HookResult):
                    last_result = result
                elif isinstance(result, HookDecision):
                    if result == HookDecision.BLOCK:
                        last_result = HookResult(action=HookAction.BLOCK)
                    elif result == HookDecision.ALLOW:
                        last_result = HookResult(action=HookAction.ALLOW)
                elif isinstance(result, dict):
                    last_result = HookResult(**result) if result else None
                elif result is not None:
                    last_result = HookResult(action=HookAction.ALLOW, metadata={"raw": result})

                # BLOCK/PAUSE 立即返回
                if last_result and last_result.action in (HookAction.BLOCK, HookAction.PAUSE):
                    self._log_execution(hook_point, ctx, last_result)
                    return last_result

            except Exception as e:
                logger.error("hook_handler_error", point=hook_point.value,
                           handler=handler.__name__ if hasattr(handler, "__name__") else "lambda",
                           error=str(e))

        if last_result:
            self._log_execution(hook_point, ctx, last_result)
        return last_result

    def _log_execution(self, hook_point: HookPoint, ctx: HookContext,
                       result: HookResult) -> None:
        """记录钩子执行。"""
        entry = {
            "id": uuid.uuid4().hex[:8],
            "hook_point": hook_point.value,
            "agent_id": ctx.agent_id,
            "trace_id": ctx.trace_id,
            "action": result.action.value,
            "timestamp": time.time(),
        }
        if result.blocked:
            entry["block_reason"] = result.block_reason
            self._block_count += 1
        if result.action == HookAction.ALLOW:
            self._allow_count += 1
        if result.action == HookAction.TRANSFORM:
            self._transform_count += 1

        self._execution_log.append(entry)

    # ── Utility ────────────────────────────────────────

    @staticmethod
    def _get_risk(tool_name: str) -> RiskLevel:
        return OPERATION_RISK.get(tool_name, DEFAULT_RISK)

    # ── Statistics ─────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """获取钩子系统统计信息。"""
        return {
            "handlers_registered": {
                point.value: len(handlers)
                for point, handlers in self._handlers.items()
            },
            "total_executions": len(self._execution_log),
            "blocks": self._block_count,
            "allows": self._allow_count,
            "transforms": self._transform_count,
            "recent_log": self._execution_log[-20:],
        }

    def clear_log(self) -> None:
        """清除执行日志和计数器。"""
        self._execution_log.clear()
        self._block_count = 0
        self._allow_count = 0
        self._transform_count = 0
