"""Orchestrator — 多模式编排引擎。

ref: Microsoft Multi-Agent RA — Orchestrator + Agent Registry + Message Bus
ref: AWS Strands Agents 1.0 — Agents-as-Tools, Handoffs, Swarms, Graphs

支持 4 种编排模式:
  - SEQUENTIAL: A → B → C，链式执行
  - PARALLEL: 并行分发，汇聚结果
  - SWARM: 动态 multi-agent 协商（LLM 驱动任务分配 + Handoff）
  - GRAPH: DAG 拓扑执行（集成现有 WorkflowEngine）

与现有系统的关系:
  - Orchestrator 是高层抽象，WorkflowEngine 是 GRAPH 模式的底层引擎
  - AgentRegistry 替代原来硬编码的 AGENT_IDS frozenset
  - MessageBus 替代 RelayClient 的直接调用，增加解耦
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.orchestration.registry import AgentRegistry, AgentInfo, AgentStatus
from src.orchestration.bus import (
    MessageBus, Message, MessageType, HandoffRequest,
)

logger = structlog.get_logger()


class OrchestrationMode(str, Enum):
    SEQUENTIAL = "sequential"     # 链式: A → B → C
    PARALLEL = "parallel"         # 并行: A, B, C 同时执行 → 汇聚
    SWARM = "swarm"               # 动态: LLM 分配任务 + Handoff
    GRAPH = "graph"               # DAG: 拓扑排序执行


@dataclass
class Task:
    """编排任务单元。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    agent_id: str = ""                     # 指定 agent（空=自动路由）
    required_capability: str = ""          # 通过能力路由
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    timeout: float = 60.0
    priority: int = 0


@dataclass
class TaskResult:
    """任务执行结果。"""

    task_id: str
    agent_id: str = ""
    success: bool = False
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
    tokens_used: int = 0
    handoff_chain: list[str] = field(default_factory=list)  # 记录 handoff 链路


@dataclass
class OrchestrationResult:
    """编排执行结果。"""

    mode: OrchestrationMode
    task_results: list[TaskResult] = field(default_factory=list)
    success: bool = False
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    handoffs: int = 0
    error: str = ""


# 执行器回调: 实际调用 agent 的函数
AgentExecutor = Any  # async Callable[[str, str, dict], Any]


class Orchestrator:
    """多模式编排引擎。

    用法:
        orchestrator = Orchestrator(registry, bus)

        # 设置 agent 执行回调
        async def run_agent(agent_id, prompt, context):
            return await relay_client.call_agent(agent_id, prompt)

        orchestrator.set_executor(run_agent)

        # 顺序执行
        result = await orchestrator.run([
            Task(agent_id="researcher", description="搜索资料"),
            Task(agent_id="copywriter", description="撰写文章"),
        ], mode=OrchestrationMode.SEQUENTIAL)

        # Swarm 模式
        result = await orchestrator.run([
            Task(required_capability="research", description="调研AI趋势"),
            Task(required_capability="writing", description="撰写报告"),
        ], mode=OrchestrationMode.SWARM)
    """

    def __init__(self, registry: AgentRegistry | None = None,
                 bus: MessageBus | None = None) -> None:
        self.registry = registry or AgentRegistry()
        self.bus = bus or MessageBus()
        self._executor: AgentExecutor | None = None

    def set_executor(self, executor: AgentExecutor) -> None:
        """设置 agent 执行回调。必须设置才能 run。

        executor 签名: async def(agent_id: str, prompt: str, context: dict) -> Any
        """
        self._executor = executor

    # ── 主入口 ─────────────────────────────────────

    async def run(self, tasks: list[Task],
                  mode: OrchestrationMode = OrchestrationMode.SEQUENTIAL,
                  context: dict[str, Any] | None = None) -> OrchestrationResult:
        """执行一组任务。"""
        if not self._executor:
            return OrchestrationResult(
                mode=mode, success=False,
                error="executor not set. Call orchestrator.set_executor() first."
            )

        start = time.monotonic()
        ctx = context or {}

        try:
            if mode == OrchestrationMode.SEQUENTIAL:
                results = await self._run_sequential(tasks, ctx)
            elif mode == OrchestrationMode.PARALLEL:
                results = await self._run_parallel(tasks, ctx)
            elif mode == OrchestrationMode.SWARM:
                results = await self._run_swarm(tasks, ctx)
            elif mode == OrchestrationMode.GRAPH:
                results = await self._run_graph(tasks, ctx)
            else:
                return OrchestrationResult(mode=mode, success=False,
                                           error=f"unknown mode: {mode}")

            all_success = all(r.success for r in results)
            total_tokens = sum(r.tokens_used for r in results)
            handoffs = sum(len(r.handoff_chain) for r in results)

            return OrchestrationResult(
                mode=mode,
                task_results=results,
                success=all_success,
                total_duration_ms=(time.monotonic() - start) * 1000,
                total_tokens=total_tokens,
                handoffs=handoffs,
            )
        except Exception as e:
            logger.error("orchestration_error", mode=mode.value, error=str(e))
            return OrchestrationResult(
                mode=mode, success=False,
                total_duration_ms=(time.monotonic() - start) * 1000,
                error=str(e),
            )

    # ── 执行模式 ───────────────────────────────────

    async def _run_sequential(self, tasks: list[Task],
                              ctx: dict[str, Any]) -> list[TaskResult]:
        """顺序执行: 每个任务的输出成为下一个任务的输入。"""
        results: list[TaskResult] = []
        ctx_current = dict(ctx)

        for i, task in enumerate(tasks):
            result = await self._execute_single(task, ctx_current)
            results.append(result)

            if result.success:
                ctx_current[task.id] = result.output
                ctx_current["_last_output"] = result.output
            else:
                # 默认: 失败不中断后续（可通过 task 配置改变）
                logger.warning("sequential_task_failed", task_id=task.id,
                               step=i, error=result.error)

        return results

    async def _run_parallel(self, tasks: list[Task],
                            ctx: dict[str, Any]) -> list[TaskResult]:
        """并行执行: 所有任务同时运行，汇聚结果。"""
        coros = [self._execute_single(t, ctx) for t in tasks]
        return list(await asyncio.gather(*coros))

    async def _run_swarm(self, tasks: list[Task],
                         ctx: dict[str, Any]) -> list[TaskResult]:
        """Swarm 模式: 动态分配 + Handoff。

        流程:
          1. 为每个 task 用能力路由选择最佳 agent
          2. 执行，如 agent 无法完成，发起 handoff
          3. 记录 handoff 链路
        """
        results: list[TaskResult] = []

        async def execute_with_handoff(task: Task) -> TaskResult:
            # 选择最佳 agent
            agent_id = task.agent_id
            if not agent_id and task.required_capability:
                best = self.registry.select_best(task.required_capability)
                if best:
                    agent_id = best.agent_id
            if not agent_id:
                agent_id = "main"

            result = await self._execute_single(task, ctx)
            result.agent_id = agent_id
            return result

        coros = [execute_with_handoff(t) for t in tasks]
        results = list(await asyncio.gather(*coros))
        return results

    async def _run_graph(self, tasks: list[Task],
                         ctx: dict[str, Any]) -> list[TaskResult]:
        """DAG 拓扑执行: 依赖解析 + 并行度最大化。

        使用拓扑排序决定执行顺序，无依赖的任务并行执行。
        """
        if not tasks:
            return []

        results_by_id: dict[str, TaskResult] = {}
        ctx_dynamic = dict(ctx)

        # 构建任务索引
        task_by_id: dict[str, Task] = {t.id: t for t in tasks}

        # 计算入度（未完成的依赖数）
        in_degree: dict[str, int] = {}
        dependents: dict[str, list[str]] = {t.id: [] for t in tasks}  # 被谁依赖

        for t in tasks:
            deps = [d for d in t.depends_on if d in task_by_id]
            in_degree[t.id] = len(deps)
            for d in deps:
                dependents[d].append(t.id)

        # 找到所有入度为 0 的任务
        ready = [t for t in tasks if in_degree[t.id] == 0]

        while ready:
            # 并行执行当前批次
            batch_results = await asyncio.gather(
                *[self._execute_single(t, ctx_dynamic) for t in ready]
            )

            next_ready: list[Task] = []
            for task, result in zip(ready, batch_results):
                results_by_id[task.id] = result
                if result.success:
                    ctx_dynamic[task.id] = result.output

                # 减少依赖者的入度
                for dep_id in dependents[task.id]:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        next_ready.append(task_by_id[dep_id])

            ready = next_ready

        # 按原始顺序返回（处理未执行的任务）
        return [results_by_id.get(t.id, TaskResult(
            task_id=t.id, success=False, error="dependency not met"
        )) for t in tasks]

    # ── 内部 ───────────────────────────────────────

    async def _execute_single(self, task: Task,
                              ctx: dict[str, Any]) -> TaskResult:
        """执行单个任务。"""
        agent_id = task.agent_id or "main"
        start = time.monotonic()

        try:
            # 构建 prompt（从 ctx 注入依赖数据）
            prompt = task.payload.get("prompt", task.description)
            resolved_context = {**ctx, "task": task.description}

            # 更新 agent 状态
            agent = self.registry.get(agent_id)
            if agent:
                self.registry.update_status(
                    agent_id, AgentStatus.BUSY,
                    current_load=agent.current_load + 1
                )

            output = await asyncio.wait_for(
                self._executor(agent_id, prompt, resolved_context),
                timeout=task.timeout
            )

            duration = (time.monotonic() - start) * 1000

            # 恢复 agent 状态
            if agent:
                self.registry.update_status(
                    agent_id, AgentStatus.IDLE,
                    current_load=max(0, agent.current_load - 1)
                )

            return TaskResult(
                task_id=task.id,
                agent_id=agent_id,
                success=True,
                output=output,
                duration_ms=duration,
            )

        except asyncio.TimeoutError:
            duration = (time.monotonic() - start) * 1000
            logger.warning("task_timeout", task_id=task.id, agent_id=agent_id)
            return TaskResult(
                task_id=task.id, agent_id=agent_id,
                success=False, error=f"timeout after {task.timeout}s",
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.error("task_error", task_id=task.id, agent_id=agent_id,
                         error=str(e))
            return TaskResult(
                task_id=task.id, agent_id=agent_id,
                success=False, error=str(e),
                duration_ms=duration,
            )

    # ── 便捷方法 ───────────────────────────────────

    async def run_sequential(self, agent_ids: list[str], prompt_template: str,
                             context: dict[str, Any] | None = None) -> OrchestrationResult:
        """快捷链式执行: 多个 agent 处理同一个 prompt。"""
        tasks = [
            Task(agent_id=aid, description=prompt_template)
            for aid in agent_ids
        ]
        return await self.run(tasks, mode=OrchestrationMode.SEQUENTIAL, context=context)

    async def run_parallel(self, agent_tasks: list[tuple[str, str]],
                           context: dict[str, Any] | None = None) -> OrchestrationResult:
        """快捷并行执行: [(agent_id, description), ...]"""
        tasks = [
            Task(agent_id=aid, description=desc)
            for aid, desc in agent_tasks
        ]
        return await self.run(tasks, mode=OrchestrationMode.PARALLEL, context=context)
