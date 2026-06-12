"""Agent 调用客户端 — 连接工作流引擎与 OpenClaw Agent 实例。

三级调用策略（按优先级）：
1. OpenClaw CLI — ``openclaw agent --agent <id> --json``，调用真实 Agent
2. cc-connect relay — 跨 bot 消息中继
3. LLM fallback — 直接调 LLM 模拟 Agent 行为
"""

from __future__ import annotations

import asyncio
import json as _json
import shutil
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.shared.config import get_settings
from src.shared.llm import get_llm
from src.shared.marker import apply_marker

logger = structlog.get_logger()

# OpenClaw CLI 路径（自动检测）
_OPENCLAW_BIN = shutil.which("openclaw") or "openclaw"


@dataclass
class PromptLoadResult:
    """加载 agent prompt 的结果，含 token 估算。"""
    content: str = ""
    estimated_tokens: int = 0
    source_path: str = ""


class RelayClient:
    """Agent 调用客户端。支持 OpenClaw CLI / cc-connect relay / LLM fallback。"""

    def __init__(self, *, use_relay: bool = False) -> None:
        self._use_relay = use_relay
        self._llm = get_llm()
        self._settings = get_settings()
        self._active_tasks: dict[str, dict[str, Any]] = {}

    # ── Public API ────────────────────────────────────────

    async def call_agent(
        self,
        agent_id: str,
        prompt: str,
        *,
        trace_id: str = "",
        timeout: int | None = None,
    ) -> str:
        """调用 Agent 执行任务。

        当 ``use_relay=True`` 时：OpenClaw CLI → cc-connect → LLM fallback。
        当 ``use_relay=False`` 时：直接 LLM fallback（安全默认）。
        """
        logger.debug("relay_call", agent_id=agent_id, trace_id=trace_id)

        timeout = timeout or self._settings.agent_task_timeout

        if self._use_relay:
            # 第一优先：OpenClaw CLI（真实 Agent）
            try:
                result = await asyncio.wait_for(
                    self._openclaw_send(agent_id, prompt, trace_id),
                    timeout=timeout,
                )
                return apply_marker(result)
            except asyncio.TimeoutError:
                logger.warning("openclaw_timeout", agent_id=agent_id, trace_id=trace_id)
            except Exception as e:
                logger.warning("openclaw_failed", agent_id=agent_id, error=str(e))

            # 第二优先：cc-connect relay
            try:
                result = await asyncio.wait_for(
                    self._relay_send(agent_id, prompt, trace_id),
                    timeout=timeout,
                )
                return apply_marker(result)
            except asyncio.TimeoutError:
                logger.warning("relay_timeout", agent_id=agent_id, trace_id=trace_id)
                raise
            except Exception as e:
                logger.warning("relay_failed_fallback_to_llm", agent_id=agent_id, error=str(e))
                result = await self._llm_fallback(agent_id, prompt, trace_id)
                return apply_marker(result)

        result = await self._llm_fallback(agent_id, prompt, trace_id)
        return apply_marker(result)

    async def call_agent_with_tools(
        self,
        agent_id: str,
        prompt: str,
        tools: list[str] | None = None,
        *,
        trace_id: str = "",
        timeout: int | None = None,
    ) -> str:
        """调用 Agent 执行任务，指定可用工具。"""
        tool_hint = f"\n可用工具: {', '.join(tools)}" if tools else ""
        full_prompt = f"{prompt}{tool_hint}"
        return await self.call_agent(agent_id, full_prompt, trace_id=trace_id, timeout=timeout)

    async def broadcast(
        self,
        agent_ids: list[str],
        prompt: str,
        *,
        trace_id: str = "",
    ) -> dict[str, str]:
        """向多个 Agent 广播相同任务，并行执行。"""
        tasks = {
            aid: asyncio.create_task(
                self.call_agent(aid, prompt, trace_id=f"{trace_id}_{aid}")
            )
            for aid in agent_ids
        }
        results = {}
        for aid, task in tasks.items():
            try:
                results[aid] = await task
            except Exception as e:
                results[aid] = f"error: {e}"
        return results

    # ── Internal: OpenClaw CLI ────────────────────────────

    async def _openclaw_send(self, agent_id: str, prompt: str, trace_id: str) -> str:
        """通过 OpenClaw CLI 调用 Agent。``openclaw agent --agent <id> --json``。"""
        agent_timeout = self._settings.agent_task_timeout

        proc = await asyncio.create_subprocess_exec(
            _OPENCLAW_BIN, "agent",
            "--agent", agent_id,
            "--message", prompt,
            "--json",
            "--timeout", str(agent_timeout),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "unknown error"
            raise RuntimeError(f"OpenClaw agent {agent_id} failed (exit {proc.returncode}): {error_msg[:200]}")

        try:
            data = _json.loads(stdout)
        except _json.JSONDecodeError as e:
            raise RuntimeError(f"OpenClaw agent {agent_id} returned invalid JSON: {e}") from e

        if data.get("status") != "ok":
            raise RuntimeError(f"OpenClaw agent {agent_id} status={data.get('status')}: {data.get('summary', '')}")

        payloads = data.get("result", {}).get("payloads", [])
        if not payloads:
            raise RuntimeError(f"OpenClaw agent {agent_id} returned no payloads")

        text = payloads[0].get("text", "")
        logger.info(
            "openclaw_agent_completed",
            agent_id=agent_id,
            trace_id=trace_id,
            text_len=len(text),
            duration_ms=data.get("result", {}).get("meta", {}).get("durationMs", 0),
        )
        return text

    # ── Internal: cc-connect relay ──────────────────────────

    async def _relay_send(self, agent_id: str, prompt: str, trace_id: str) -> str:
        """通过 cc-connect relay 发送任务到目标 Agent。"""
        # cc-connect relay send --to <agent_id> "<prompt>"
        proc = await asyncio.create_subprocess_exec(
            "cc-connect", "relay", "send",
            "--to", agent_id,
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            error_msg = stderr.decode() if stderr else "unknown relay error"
            raise RuntimeError(f"Relay to {agent_id} failed: {error_msg}")
        return stdout.decode().strip()

    async def _llm_fallback(self, agent_id: str, prompt: str, trace_id: str) -> str:
        """直接使用 LLM 模拟 Agent 行为。加载 Agent 的 system prompt。"""
        result = self._load_agent_prompt(agent_id)

        if result.estimated_tokens > 8000:
            logger.warning("large_system_prompt", agent_id=agent_id,
                          estimated_tokens=result.estimated_tokens,
                          source=result.source_path, trace_id=trace_id)

        messages = []
        if result.content:
            messages.append({"role": "system", "content": result.content})
        messages.append({"role": "user", "content": prompt})

        response = await self._llm.chat(
            messages,
            trace_id=f"{trace_id}_fallback",
        )
        return response

    @staticmethod
    def _load_agent_prompt(agent_id: str) -> PromptLoadResult:
        """加载 Agent 的 AGENTS.md 内容作为 system prompt，含 token 估算。"""
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[2]
        md_path = project_root / "config" / "agents" / f"{agent_id}.md"
        if md_path.exists():
            content = md_path.read_text(encoding="utf-8")
            estimated = len(content) // 2  # 中英文混合约 2 字符/token
            return PromptLoadResult(content=content, estimated_tokens=estimated,
                                    source_path=str(md_path))
        return PromptLoadResult()

    # ── Status ────────────────────────────────────────────

    def get_active_tasks(self) -> dict[str, dict[str, Any]]:
        return dict(self._active_tasks)

    @staticmethod
    def is_openclaw_available() -> bool:
        """检查 openclaw CLI 是否安装。"""
        return shutil.which("openclaw") is not None

    @property
    def relay_available(self) -> bool:
        """OpenClaw CLI 可用且 relay 模式已启用。"""
        return self._use_relay and self.is_openclaw_available()


# 模块级单例
_relay_client: RelayClient | None = None


def get_relay_client(*, use_relay: bool = False) -> RelayClient:
    """获取 RelayClient 单例。

    Args:
        use_relay: 是否启用 OpenClaw Agent 调用。默认 False（仅 LLM fallback）。
    """
    global _relay_client
    if _relay_client is None:
        _relay_client = RelayClient(use_relay=use_relay)
    elif use_relay and not _relay_client._use_relay:
        # 如果之前用 False 创建，但现在是 True，重新创建
        _relay_client = RelayClient(use_relay=True)
    return _relay_client
