"""步骤执行器。支持 task / condition / parallel / loop / human 五种类型。

每步自动创建 OpenTelemetry GenAI 语义约定的追踪 Span。
所有 Agent 间消息传递使用固定 JSON 格式（AgentMessage）。
ref: OpenTelemetry GenAI Semantic Conventions
"""

from __future__ import annotations

import asyncio
import json as _json
from typing import Any

import structlog

from src.observability.tracer import SpanKind, get_tracer
from src.shared.models import AgentMessage, StepConfig, StepResult, StepType

logger = structlog.get_logger()

# Agent 间消息传递的固定 JSON schema 说明，追加到每个 prompt 末尾
_JSON_FORMAT_INSTRUCTION = """
===
输出格式要求：必须输出以下 JSON 格式，不要输出其他内容。

{
  "summary": "一句话总结你的结论",
  "confidence": "high",
  "data": { ... 你的结构化结果，按任务要求组织字段 ... },
  "error": null
}

字段说明：
- summary: 一句话总结
- confidence: 置信度。high=有可靠依据可放心用 / medium=有线索但不确定 / low=推测或无法查证
- data: 结构化结果
- error: null 表示正常，有错误时写原因并把 confidence 设为 low

防幻觉规则（必须遵守）：
1. 如果你不确定答案或信息无法查证 → confidence 设 "low"，在 data 里说明哪里不确定
2. 如果完全不知道 → error 字段写"无法获取可靠信息"，confidence 设 "low"，data 设 {}
3. 不要编造数据、来源、数字。宁可说不知道也不要瞎编
4. 所有事实性陈述必须在 data 中标注来源或依据

只输出 JSON，不要输出解释、不要输出 markdown 代码块标记。
"""


class StepExecutor:
    """执行单个工作流步骤。"""

    async def execute(
        self, step: StepConfig, context: dict, *, trace_id: str = "",
        budget_manager: Any = None,
    ) -> StepResult:
        """根据步骤类型分发。自动追踪。"""
        tracer = get_tracer()

        with tracer.start_span(
            kind=SpanKind.STEP,
            name=f"{step.type.value}:{step.id}",
            agent_id=step.agent or "",
            step_id=step.id,
            skill_name=step.type.value,
        ) as span:
            span.metadata["depends_on"] = step.depends_on
            span.metadata["has_retry"] = bool(step.config.get("retry"))

            match step.type:
                case StepType.TASK:
                    result = await self._execute_task(step, context, span.trace_id)
                case StepType.CONDITION:
                    result = self._execute_condition(step, context, span.trace_id)
                case StepType.PARALLEL:
                    result = await self._execute_parallel(step, context, span.trace_id)
                case StepType.LOOP:
                    result = await self._execute_loop(step, context, span.trace_id, budget_manager=budget_manager)
                case StepType.HUMAN:
                    result = StepResult(
                        step_id=step.id, status="success",
                        output="[awaiting_human_approval]", trace_id=span.trace_id,
                    )
                case _:
                    result = StepResult(
                        step_id=step.id, status="failed",
                        error=f"unknown_step_type: {step.type}", trace_id=span.trace_id,
                    )

            if result.status == "success":
                span.status = "ok"
            else:
                span.status = "error"
                span.error_message = result.error or ""

            return result

    # ── Task ─────────────────────────────────────────────

    async def _execute_task(
        self, step: StepConfig, context: dict, trace_id: str = ""
    ) -> StepResult:
        """执行任务步骤。调用 Agent，解析 JSON 格式输出。

        支持 RAG 检索增强和 verify 多模型校验（通过 step.config 配置）。
        """
        prompt_template = step.config.get("prompt", "")
        timeout = step.config.get("timeout", 300)
        retry_config = step.config.get("retry", {})
        rag_config = step.config.get("rag", {})
        verify_config = step.config.get("verify", {})

        # 替换模板变量（支持 {{key}} 和 $key 两种语法）
        prompt = prompt_template
        for key, value in context.items():
            if isinstance(value, dict):
                if "data" in value and "status" in value:
                    text_value = _json.dumps(value["data"], ensure_ascii=False)
                elif "output" in value:
                    inner = value["output"]
                    if isinstance(inner, dict) and "data" in inner:
                        text_value = _json.dumps(inner["data"], ensure_ascii=False)
                    else:
                        text_value = str(inner)
                else:
                    text_value = _json.dumps(value, ensure_ascii=False)
            else:
                text_value = str(value)
            prompt = prompt.replace(f"{{{{{key}}}}}", text_value)
            prompt = prompt.replace(f"${key}", text_value)

        # ── RAG 检索增强 ──
        if rag_config:
            rag_prompt = await self._apply_rag(prompt, rag_config)
            if rag_prompt:
                prompt = rag_prompt

        # 追加 JSON 输出格式要求
        prompt = prompt + _JSON_FORMAT_INSTRUCTION

        max_attempts = retry_config.get("max_attempts", 1)
        backoff = retry_config.get("backoff", "none")
        delays = [2**i if backoff == "exponential" else 1 for i in range(max_attempts)]

        last_error: str | None = None
        for attempt in range(max_attempts):
            try:
                raw_response = await asyncio.wait_for(
                    self._call_agent(step.agent or "main", prompt, trace_id),
                    timeout=timeout,
                )
                parsed = self._parse_agent_response(
                    step.agent or "unknown", raw_response, step.id
                )

                # ── 多模型校验 ──
                if verify_config and parsed["status"] == "success":
                    parsed = await self._apply_verify(
                        parsed, verify_config, prompt, timeout, trace_id
                    )

                return StepResult(
                    step_id=step.id, status=parsed["status"],
                    output=parsed,
                    retry_count=attempt, trace_id=trace_id,
                )
            except asyncio.TimeoutError:
                last_error = "timeout"
            except Exception as e:
                last_error = str(e)

            if attempt < max_attempts - 1:
                await asyncio.sleep(delays[attempt])

        return StepResult(
            step_id=step.id, status="failed", error=last_error,
            retry_count=max_attempts, trace_id=trace_id,
        )

    # ── Condition ────────────────────────────────────────

    def _execute_condition(
        self, step: StepConfig, context: dict, trace_id: str = ""
    ) -> StepResult:
        """条件路由：解析表达式，返回分支信息。"""
        expr = step.config.get("expression", "true")
        then_step = step.config.get("then", "")
        else_step = step.config.get("else", "")

        # 简化表达式解析：$variable comparison
        evaluated = self._eval_expr(expr, context)
        next_step = then_step if evaluated else else_step

        return StepResult(
            step_id=step.id, status="success",
            output={"condition_met": evaluated, "next_step": next_step},
            trace_id=trace_id,
        )

    # ── Parallel ─────────────────────────────────────────

    async def _execute_parallel(
        self, step: StepConfig, context: dict, trace_id: str = ""
    ) -> StepResult:
        """并行执行多个子步骤。"""
        branches = step.config.get("branches", [])
        merge = step.config.get("merge", "all")

        tasks = []
        for branch_id in branches:
            # 子步骤用简化的 task 执行
            tasks.append(asyncio.create_task(
                self._call_agent(step.agent or "main", str(branch_id), trace_id)
            ))

        if merge == "all":
            results = await asyncio.gather(*tasks, return_exceptions=True)
        elif merge == "first":
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            results = [t.result() for t in done]
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        return StepResult(
            step_id=step.id, status="success",
            output={"parallel_results": [
                str(r) if not isinstance(r, Exception) else str(r) for r in results
            ]},
            trace_id=trace_id,
        )

    # ── Loop ─────────────────────────────────────────────

    async def _execute_loop(
        self, step: StepConfig, context: dict, trace_id: str = "",
        budget_manager: Any = None,
    ) -> StepResult:
        """循环执行直到条件满足或达到最大迭代次数。"""
        condition_expr = step.config.get("condition", "false")
        max_iterations = step.config.get("max_iterations", 3)
        body_prompt = step.config.get("body_prompt", "")

        outputs: list[Any] = []
        for i in range(max_iterations):
            # 预算检查：每轮迭代前检查是否可继续
            if budget_manager is not None:
                estimated = len(body_prompt) // 2 + 500
                if not budget_manager.can_proceed(estimated, workflow_id="reflection"):
                    logger.warning("loop_budget_exhausted", step_id=step.id,
                                  iteration=i, trace_id=trace_id)
                    break  # 优雅降级：停止循环而非 crash

            if not self._eval_expr(condition_expr, context):
                break
            try:
                output = await self._call_agent(
                    step.agent or "main", body_prompt, f"{trace_id}_loop_{i}"
                )
                outputs.append(output)
                context["loop_output"] = output
                context["loop_iteration"] = i + 1
            except Exception as e:
                return StepResult(
                    step_id=step.id, status="failed", error=str(e),
                    trace_id=trace_id,
                )

        return StepResult(
            step_id=step.id, status="success",
            output={"loop_results": outputs, "iterations": len(outputs)},
            trace_id=trace_id,
        )

    # ── Helpers ──────────────────────────────────────────

    async def _call_agent(
        self, agent_id: str, prompt: str, trace_id: str = ""
    ) -> str:
        """调用 Agent 执行任务。通过 relay client（支持 MCP relay + LLM fallback）。"""
        from src.relay.client import get_relay_client
        from src.observability.metrics import get_metrics

        tracer = get_tracer()
        metrics = get_metrics()

        with tracer.start_span(
            kind=SpanKind.LLM,
            name=f"call_agent:{agent_id}",
            agent_id=agent_id,
        ) as span:
            relay = get_relay_client()
            result = await relay.call_agent(agent_id, prompt, trace_id=trace_id)

            # 粗略 token 估算（中文约 1.5 字符/token，英文约 4 字符/token）
            estimated_input = len(prompt) // 2
            estimated_output = len(result) // 2
            span.input_tokens = estimated_input
            span.output_tokens = estimated_output
            span.total_tokens = estimated_input + estimated_output
            span.model = getattr(relay, "_llm", None) and getattr(relay._llm, "model", "") or ""

            # 成本归因
            if span.model:
                span.cost_usd = metrics.calculate_cost(span.model, estimated_input, estimated_output)

            return result

    # ── RAG & Verify ───────────────────────────────────────

    @staticmethod
    async def _apply_rag(prompt: str, rag_config: dict) -> str | None:
        """RAG 检索增强：从知识库检索上下文，拼接到 Prompt 前。

        rag_config:
            enabled: true/false
            top_k: 3
            threshold: 0.4
            agent_id: "researcher"  # 可选，限定来源 Agent
        """
        if not rag_config.get("enabled", True):
            return None

        from src.rag.retriever import RAGRetriever

        top_k = rag_config.get("top_k", 3)
        threshold = rag_config.get("threshold", 0.4)
        agent_id = rag_config.get("agent_id")

        retriever = RAGRetriever(top_k=top_k, threshold=threshold)
        results = retriever.retrieve(prompt, agent_id=agent_id)
        if not results:
            return None

        context = retriever.format_context(results)
        # 把上下文放在 prompt 前面，JSON 格式说明放在最后
        return context + "\n\n" + prompt

    async def _apply_verify(
        self, parsed: dict, verify_config: dict,
        original_prompt: str, timeout: int, trace_id: str,
    ) -> dict:
        """多模型校验：把主 Agent 的结果发给校验 Agent 做事实核查。

        verify_config:
            agent: "tech-dev"      # 校验 Agent ID
            prompt: "..."          # 可选，自定义校验 prompt
        """
        verify_agent = verify_config.get("agent", "")
        if not verify_agent:
            return parsed

        verify_prompt = verify_config.get("prompt", "")
        if not verify_prompt:
            verify_prompt = (
                "请校验以下内容的准确性，指出事实错误、逻辑矛盾、或不确定的地方。\n\n"
                f"原始问题：\n{original_prompt[:2000]}\n\n"
                f"待校验回答：\n{_json.dumps(parsed, ensure_ascii=False)}"
            )

        try:
            raw = await asyncio.wait_for(
                self._call_agent(verify_agent, verify_prompt + _JSON_FORMAT_INSTRUCTION,
                                 f"{trace_id}_verify"),
                timeout=timeout,
            )
            verify_result = self._parse_agent_response(verify_agent, raw, "verify")
        except Exception as e:
            logger.warning("verify_agent_failed", agent=verify_agent, error=str(e))
            # 校验失败不阻断，保留原始结果但标低置信度
            parsed["confidence"] = "low"
            parsed["data"]["_verify_error"] = str(e)
            return parsed

        # 校验结果写回
        parsed["data"]["_verified_by"] = verify_agent
        parsed["data"]["_verify_feedback"] = verify_result.get("summary", "")
        # 如果校验 Agent 指出了具体问题，调整置信度
        if verify_result.get("status") == "error" or verify_result.get("confidence") == "low":
            parsed["confidence"] = "low"
        elif verify_result.get("confidence") == "high":
            parsed["confidence"] = "high"

        logger.debug(
            "verify_complete",
            agent=verify_agent,
            confidence=parsed["confidence"],
            trace_id=trace_id,
        )
        return parsed

    # ── Response Parsing ──────────────────────────────────

    @staticmethod
    def _parse_agent_response(agent_id: str, raw_response: str, step_id: str) -> dict:
        """解析 Agent 的 JSON 输出为标准化字典。

        正常情况 Agent 返回 JSON: {"summary": "...", "data": {...}, "error": null}
        解析失败时把原始文本包装为 success 输出，不阻断工作流。
        """
        text = raw_response.strip()
        # 去掉可能的 markdown 代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()

        try:
            parsed = _json.loads(text)
        except (_json.JSONDecodeError, ValueError):
            logger.warning(
                "agent_response_parse_failed",
                agent_id=agent_id,
                step_id=step_id,
                response_preview=raw_response[:200],
            )
            return {
                "status": "success",
                "summary": f"Agent {agent_id} response",
                "confidence": "low",
                "data": {"raw": raw_response},
                "error": None,
            }

        if not isinstance(parsed, dict):
            return {
                "status": "success",
                "summary": f"Agent {agent_id} response",
                "confidence": "low",
                "data": {"raw": raw_response},
                "error": None,
            }

        # 标准化 confidence 值
        raw_confidence = parsed.get("confidence", "medium")
        valid_confidences = {"high", "medium", "low"}
        confidence = raw_confidence if raw_confidence in valid_confidences else "medium"

        return {
            "status": parsed.get("status", "success") if not parsed.get("error") else "error",
            "summary": parsed.get("summary", ""),
            "confidence": confidence,
            "data": parsed.get("data", {}),
            "error": parsed.get("error"),
        }

    @staticmethod
    def _resolve_path(value: str, context: dict) -> str:
        """解析 $var.path1.path2 引用，返回已替换的字符串。"""
        import re as _re

        def _replacer(match: _re.Match) -> str:
            brace_form = match.group(0).startswith("${")
            inner = match.group(1) or match.group(2)
            parts = inner.split(".")
            current: Any = context
            for p in parts:
                if isinstance(current, dict):
                    current = current.get(p)
                else:
                    return match.group(0)  # 无法导航，保留原文
                if current is None:
                    return match.group(0)
            if isinstance(current, (str, int, float, bool)):
                return str(current)
            if isinstance(current, dict):
                return _json.dumps(current, ensure_ascii=False)
            return match.group(0)

        # 匹配 $var.path1.path2 或 ${var.path1.path2}
        resolved = _re.sub(
            r'\$\{(\w+(?:\.\w+)*)\}|\$(\w+(?:\.\w+)*)',
            _replacer,
            value,
        )
        return resolved

    @staticmethod
    def _eval_expr(expr: str, context: dict) -> bool:
        """简化条件表达式评估。支持 $var 和 $var.path 引用，以及基本比较。"""
        resolved = StepExecutor._resolve_path(expr, context)

        # 安全评估（仅支持简单比较）
        try:
            if ">=" in resolved:
                left, right = resolved.split(">=", 1)
                return float(left.strip().strip("'").strip('"')) >= float(right.strip().strip("'").strip('"'))
            if "<=" in resolved:
                left, right = resolved.split("<=", 1)
                return float(left.strip().strip("'").strip('"')) <= float(right.strip().strip("'").strip('"'))
            if ">" in resolved:
                left, right = resolved.split(">", 1)
                return float(left.strip().strip("'").strip('"')) > float(right.strip().strip("'").strip('"'))
            if "<" in resolved:
                left, right = resolved.split("<", 1)
                return float(left.strip().strip("'").strip('"')) < float(right.strip().strip("'").strip('"'))
            if "==" in resolved:
                left, right = resolved.split("==", 1)
                return left.strip().strip("'").strip('"') == right.strip().strip("'").strip('"')
            # 默认：truthy/falsy 检查
            return bool(resolved.strip() and resolved.strip().lower() not in ("false", "0", "none", "null", ""))
        except (ValueError, TypeError):
            return False
