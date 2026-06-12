"""工作流编排引擎核心。不依赖 Agent 框架，直接 asyncio 实现。

每步执行自动通过 OpenTelemetry GenAI semconv 追踪。
ref: OpenTelemetry GenAI Semantic Conventions
ref: Grafana Cloud + OpenLIT — zero-code AI observability
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import structlog
import yaml

from src.observability.tracer import SpanKind, get_tracer
from src.observability.audit import get_audit_logger
from src.shared.models import StepResult, WorkflowDefinition, WorkflowResult
from src.workflow.dependency import DependencyResolver
from src.workflow.steps import StepExecutor

logger = structlog.get_logger()


class WorkflowEngine:
    """编排引擎：加载 YAML → 拓扑排序 → 执行 steps → 返回结果。"""

    def __init__(self, step_executor: StepExecutor | None = None) -> None:
        self._executor = step_executor or StepExecutor()
        self._resolver = DependencyResolver()

    # ── Load ─────────────────────────────────────────────

    @staticmethod
    def load(yaml_path: str | Path) -> WorkflowDefinition:
        """从 YAML 文件加载工作流定义。支持相对路径（相对于项目根）。"""
        from src.shared.config import get_project_root

        path = Path(yaml_path)
        if not path.is_absolute():
            path = get_project_root() / path
        if not path.exists():
            from src.shared.errors import WorkflowParseError
            raise WorkflowParseError(f"工作流文件不存在: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        wf_data = raw.get("workflow", raw)
        from src.shared.models import StepConfig, StepType

        steps = []
        for s in wf_data.get("steps", []):
            step_type = StepType(s.get("type", "task"))
            steps.append(StepConfig(
                id=s["id"],
                name=s.get("name", s["id"]),
                agent=s.get("agent"),
                type=step_type,
                depends_on=s.get("depends_on", []),
                input=s.get("input", {}),
                config=s.get("config", {}),
            ))

        return WorkflowDefinition(
            name=wf_data["name"],
            version=wf_data.get("version", "1.0"),
            description=wf_data.get("description", ""),
            trigger=wf_data.get("trigger", {}),
            steps=steps,
        )

    # ── Execute ──────────────────────────────────────────

    async def run(
        self, workflow: WorkflowDefinition, params: dict, *, trace_id: str = ""
    ) -> WorkflowResult:
        """执行工作流。自动创建追踪 Trace。"""
        tracer = get_tracer()
        audit = get_audit_logger()

        trace_id = trace_id or f"wf-{workflow.name}-{uuid.uuid4().hex[:8]}"
        session_id = f"session-{workflow.name}-{uuid.uuid4().hex[:8]}"

        with tracer.start_trace(session_id=session_id):
            with tracer.start_span(
                kind=SpanKind.AGENT,
                name=f"workflow:{workflow.name}",
                skill_name=workflow.name,
            ) as root_span:
                root_span.metadata["workflow_name"] = workflow.name
                root_span.metadata["workflow_version"] = workflow.version
                root_span.metadata["params"] = {k: str(v)[:100] for k, v in params.items()}

                result = await self._run_steps(workflow, params, trace_id, root_span)

                # 审计日志
                trace = tracer.get_current_trace()
                if trace:
                    audit.on_trace(trace)

                return result

    async def _run_steps(
        self,
        workflow: WorkflowDefinition,
        params: dict,
        trace_id: str,
        root_span: "Span",
    ) -> WorkflowResult:
        result = WorkflowResult(workflow_name=workflow.name, trace_id=trace_id)

        try:
            sorted_steps = self._resolver.topological_sort(workflow.steps)
        except Exception as e:
            result.status = "failed"
            root_span.status = "error"
            root_span.error_message = str(e)
            logger.error("workflow_topology_error", error=str(e), trace_id=trace_id)
            return result

        context: dict = {**params}
        total_start = time.monotonic()

        for step in sorted_steps:
            step_start = time.monotonic()

            try:
                # 解析输入
                resolved_input = self._resolver.resolve_inputs(step, context)
                # 执行步骤
                step_result = await self._executor.execute(
                    step, {**context, **resolved_input}, trace_id=trace_id
                )
                step_result.duration_ms = int((time.monotonic() - step_start) * 1000)
                result.steps.append(step_result)
                result.total_tokens += step_result.tokens_used

                if step_result.status == "failed":
                    result.status = "partial_success"
                    logger.warning(
                        "step_failed", step=step.id, error=step_result.error, trace_id=trace_id
                    )
                    break

                # 将输出写入上下文（存为字典，供 $step.output 等引用）
                context[step.id] = {
                    "output": step_result.output,
                    "status": step_result.status,
                    "tokens_used": step_result.tokens_used,
                    "duration_ms": step_result.duration_ms,
                }

            except Exception as e:
                result.steps.append(StepResult(
                    step_id=step.id, status="failed", error=str(e),
                    trace_id=trace_id,
                ))
                result.status = "failed"
                logger.error("step_exception", step=step.id, error=str(e), trace_id=trace_id)
                break

        result.total_duration_ms = int((time.monotonic() - total_start) * 1000)

        if result.status not in ("failed", "partial_success", "cancelled"):
            result.status = "completed"
            result.final_output = context.get(sorted_steps[-1].id, {}).get("output") if sorted_steps else None

        logger.info(
            "workflow_completed",
            name=workflow.name,
            status=result.status,
            steps=len(result.steps),
            duration_ms=result.total_duration_ms,
            trace_id=trace_id,
        )
        return result
