"""多 Agent 协同大纲系统 —— 中心化任务状态表 + 原子边界划分 + 强制收敛。

对齐文档规格：
- 唯一真相源：中心化"大纲"维护任务状态
- 原子级边界划分：分发前标记每个子任务的独占数据域
- 强制收敛：子 agent 返回前必须自我压缩
- 扁平优于嵌套：只有顶层 Orchestrator 能派生 agent
- 物理限流：最大并发 20
"""

from __future__ import annotations

import asyncio
import json as _json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.shared.config import get_settings
from src.shared.models import TaskCognitiveState

logger = structlog.get_logger()


class TaskOrchestrator:
    """多 Agent 协同调度器（大纲系统）。"""

    # 域锁默认 TTL（秒），超时自动释放，防止死锁
    DOMAIN_LOCK_TTL_SECONDS = 600

    def __init__(self, task_memory, *, max_concurrent: int | None = None) -> None:
        self._settings = get_settings()
        self._task_memory = task_memory  # TaskMemory instance
        self._max_concurrent = max_concurrent or self._settings.orchestrator_max_concurrent
        self._active_dispatches: dict[str, dict[str, Any]] = {}
        self._domain_locks: dict[str, tuple[str, float]] = {}  # domain → (agent_id, locked_at)

    # ── Dispatch ──────────────────────────────────────────

    async def dispatch(
        self,
        parent_task_id: str,
        sub_agent_id: str,
        instruction: str,
        *,
        data_domain: str | None = None,
    ) -> str:
        """派发子任务到 agent。原子级边界划分 + 物理限流。"""
        # 物理限流
        active = len(self._active_dispatches)
        if active >= self._max_concurrent:
            raise RuntimeError(f"已达最大并发 {self._max_concurrent}，当前活跃 {active}")

        domain_locked = False
        sub_task_id = ""
        try:
            # 数据域锁检查（原子边界划分）+ TTL 超时自动释放
            if data_domain:
                now_ts = datetime.now(timezone.utc).timestamp()
                if data_domain in self._domain_locks:
                    existing_agent, locked_at = self._domain_locks[data_domain]
                    if now_ts - locked_at > self.DOMAIN_LOCK_TTL_SECONDS:
                        # 锁已超时，自动释放
                        logger.warning("domain_lock_expired", domain=data_domain, agent=existing_agent)
                        self._domain_locks.pop(data_domain, None)
                    elif existing_agent != sub_agent_id:
                        raise RuntimeError(
                            f"数据域 '{data_domain}' 已被 {existing_agent} 锁定，{sub_agent_id} 不能并行操作"
                        )
                self._domain_locks[data_domain] = (sub_agent_id, now_ts)
                domain_locked = True

            # 创建子任务
            sub_task_id = f"sub_{uuid.uuid4().hex[:12]}"
            session_step_id = f"step_{uuid.uuid4().hex[:16]}"
            task = self._task_memory.create_task(
                owner_agent_id=sub_agent_id,
                parent_task_id=parent_task_id,
                task_id=sub_task_id,
                session_step_id=session_step_id,
                acl={sub_agent_id: "rw", "main": "r"},
            )

            self._active_dispatches[sub_task_id] = {
                "agent_id": sub_agent_id,
                "domain": data_domain,
                "dispatched_at": datetime.now(timezone.utc).isoformat(),
                "instruction": instruction,
            }

            logger.info("subtask_dispatched", task_id=sub_task_id, agent=sub_agent_id, domain=data_domain)
            return sub_task_id

        except Exception:
            # 失败时清理锁和追踪
            if domain_locked and data_domain:
                self._domain_locks.pop(data_domain, None)
            if sub_task_id and sub_task_id in self._active_dispatches:
                self._active_dispatches.pop(sub_task_id, None)
            logger.error("subtask_dispatch_failed", agent=sub_agent_id, domain=data_domain, exc_info=True)
            raise

    # ── Convergence ────────────────────────────────────────

    @staticmethod
    def converge(result: str, *, max_tokens: int = 2000) -> str:
        """强制收敛：子 agent 返回前必须自我压缩。

        过滤噪音，只保留精炼结论与关键事实。
        """
        if len(result) // 4 <= max_tokens:
            return result

        # 简单截断 + 元信息标注
        truncated = result[: max_tokens * 4]
        summary = (
            "[已压缩] 以下为精炼结论与关键事实（原始内容过长已截断）：\n"
            + truncated
            + f"\n\n[原始长度: {len(result)} 字符，已截断至 {max_tokens * 4} 字符]"
        )
        return summary

    @staticmethod
    def get_convergence_prompt() -> str:
        """返回收敛指令（注入到子 agent prompt 中）。"""
        return (
            "返回前必须自我压缩：只带回精炼结论与关键事实，不要返回全文原文。"
            "格式：①核心发现（1-3 句）②关键数据（如有）③建议下一步。"
        )

    # ── Status ────────────────────────────────────────────

    def get_outline(self, parent_task_id: str) -> dict[str, Any]:
        """获取任务大纲状态。"""
        chain = self._task_memory.get_task_chain(parent_task_id)
        return {
            "parent_task_id": parent_task_id,
            "subtasks": [
                {
                    "task_id": t.task_id,
                    "agent_id": t.owner_agent_id,
                    "status": t.status.value if isinstance(t.status, TaskCognitiveState) else t.status,
                    "artifacts": t.artifacts,
                }
                for t in chain[1:]  # 跳过 parent 自身
            ],
            "domain_locks": {k: v[0] for k, v in self._domain_locks.items()},
            "active_dispatches": len(self._active_dispatches),
            "max_concurrent": self._max_concurrent,
        }

    async def complete_subtask(self, sub_task_id: str, *, result: str | None = None) -> None:
        """子任务完成，释放资源。"""
        dispatch = self._active_dispatches.pop(sub_task_id, None)
        if dispatch and dispatch.get("domain"):
            self._domain_locks.pop(dispatch["domain"], None)

        if result:
            self._task_memory.mark_done(sub_task_id, artifacts={"result": result})
        else:
            self._task_memory.update_task_status(sub_task_id, "done")

        logger.info("subtask_completed", task_id=sub_task_id)

    # ── Forbidden: Nested spawn ───────────────────────────

    @staticmethod
    def is_orchestrator(agent_id: str) -> bool:
        """只有顶层 Orchestrator（main）可以派生 agent。"""
        return agent_id == "main"
