"""Streaming Engine — SSE/WebSocket 实时流式输出。

ref: OpenAI streaming — SSE format with token-level deltas
ref: Anthropic — streaming events: message_start, content_block_start/delta/stop, message_delta
ref: LangChain — streaming callbacks for real-time visibility

事件类型:
  - token: 逐 token 输出
  - tool_call: 工具调用开始/结果
  - step: 执行步骤开始/完成
  - thinking: 推理过程（CoT visible thinking）
  - status: 状态更新（进度、估计剩余时间）
  - error: 错误事件
  - done: 流结束
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable

import structlog

logger = structlog.get_logger()


class StreamEventType(str, Enum):
    TOKEN = "token"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    THINKING = "thinking"
    STATUS = "status"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """流事件。"""
    type: StreamEventType
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    agent_id: str = ""

    def to_sse(self) -> str:
        """转为 SSE 格式。"""
        payload = {
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
        }, ensure_ascii=False)


class StreamingEngine:
    """流式输出引擎。

    用法:
        engine = StreamingEngine()

        async for event in engine.stream_response(agent_id="researcher", prompt="..."):
            if event.type == StreamEventType.TOKEN:
                print(event.data["text"], end="", flush=True)
            elif event.type == StreamEventType.DONE:
                print()

        # 通过回调注册
        @engine.on(StreamEventType.TOKEN)
        def on_token(event): ...

        # 转为 SSE 供 HTTP 使用
        async for sse_chunk in engine.stream_sse(agent_id="researcher", prompt="..."):
            yield sse_chunk
    """

    def __init__(self) -> None:
        self._listeners: dict[StreamEventType, list[Callable[[StreamEvent], Any]]] = \
            defaultdict(list)
        self._active_streams: dict[str, asyncio.Event] = {}

    # ── 回调注册 ───────────────────────────────────

    def on(self, event_type: StreamEventType):
        """装饰器: 注册事件监听器。"""
        def decorator(fn):
            self._listeners[event_type].append(fn)
            return fn
        return decorator

    def emit(self, event: StreamEvent) -> None:
        """同步触发事件（非流式场景）。"""
        for listener in self._listeners.get(event.type, []):
            try:
                listener(event)
            except Exception as e:
                logger.error("stream_listener_error",
                            event_type=event.type.value, error=str(e))

    # ── 流式生成 ──────────────────────────────────

    async def stream_response(self, agent_id: str, prompt: str,
                              session_id: str = "",
                              simulate: bool = False,
                              response_text: str = "") -> AsyncIterator[StreamEvent]:
        """流式输出 agent 响应。

        生产环境: 接入真实 LLM streaming API。
        当前实现: token 级模拟 + 事件钩子（可替换为实际 API）。
        """
        sid = session_id or uuid.uuid4().hex[:12]
        cancel = asyncio.Event()
        self._active_streams[sid] = cancel

        try:
            # 1. 状态: 开始
            yield self._emit_sync(StreamEvent(
                type=StreamEventType.STATUS,
                data={"status": "started", "agent_id": agent_id, "prompt_length": len(prompt)},
                session_id=sid, agent_id=agent_id,
            ))

            # 2. 思考阶段
            thinking = f"分析用户问题: {prompt[:100]}..."
            yield self._emit_sync(StreamEvent(
                type=StreamEventType.THINKING,
                data={"text": thinking},
                session_id=sid, agent_id=agent_id,
            ))

            # 对异步取消敏感
            if cancel.is_set():
                return

            # 3. Token 流式输出
            text = response_text or f"这是对 '{prompt[:30]}...' 的模拟流式响应。"
            words = text.split()
            for i, word in enumerate(words):
                if cancel.is_set():
                    yield self._emit_sync(StreamEvent(
                        type=StreamEventType.DONE,
                        data={"status": "cancelled", "total_tokens": i},
                        session_id=sid, agent_id=agent_id,
                    ))
                    return
                token_text = word + (" " if i < len(words) - 1 else "")
                yield self._emit_sync(StreamEvent(
                    type=StreamEventType.TOKEN,
                    data={"text": token_text, "index": i, "total": len(words)},
                    session_id=sid, agent_id=agent_id,
                ))
                await asyncio.sleep(0.02)  # 模拟流延迟

            # 4. 完成
            yield self._emit_sync(StreamEvent(
                type=StreamEventType.DONE,
                data={"status": "completed", "total_tokens": len(words),
                      "elapsed_ms": 0},
                session_id=sid, agent_id=agent_id,
            ))

        finally:
            self._active_streams.pop(sid, None)

    async def stream_sse(self, agent_id: str, prompt: str,
                         session_id: str = "",
                         **kwargs: Any) -> AsyncIterator[str]:
        """流式输出 SSE 格式 — 适合 HTTP 响应。"""
        async for event in self.stream_response(agent_id, prompt, session_id, **kwargs):
            yield event.to_sse()

    async def stream_tool_call(self, agent_id: str, tool_name: str,
                               tool_input: dict[str, Any],
                               session_id: str = "") -> AsyncIterator[StreamEvent]:
        """流式输出工具调用过程。"""
        sid = session_id or uuid.uuid4().hex[:12]

        # Tool call start
        yield self._emit_sync(StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            data={"tool_name": tool_name, "input": tool_input},
            session_id=sid, agent_id=agent_id,
        ))

        await asyncio.sleep(0.1)  # 模拟工具执行

        # Tool call end
        yield self._emit_sync(StreamEvent(
            type=StreamEventType.TOOL_CALL_END,
            data={"tool_name": tool_name, "status": "success",
                  "output_preview": "...", "elapsed_ms": 150},
            session_id=sid, agent_id=agent_id,
        ))

    # ── 控制 ───────────────────────────────────────

    def cancel(self, session_id: str) -> bool:
        """取消流式输出。"""
        cancel = self._active_streams.get(session_id)
        if cancel:
            cancel.set()
            return True
        return False

    # ── 内部 ───────────────────────────────────────

    def _emit_sync(self, event: StreamEvent) -> StreamEvent:
        """发出事件并通知同步监听器。"""
        self.emit(event)
        return event

    @property
    def active_streams(self) -> int:
        return len(self._active_streams)
