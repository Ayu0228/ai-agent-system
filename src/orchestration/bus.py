"""Message Bus — 发布/订阅事件总线，跨 Agent 可靠消息传递。

ref: Microsoft Multi-Agent RA — Message Bus pattern for inter-agent communication
ref: AWS Strands Agents 1.0 — Handoffs primitive

支持:
  - 点对点消息（命令、查询）
  - 发布/订阅（事件广播）
  - Handoff 协议（任务转移 + 上下文传递）
  - 消息去重 + 超时重试
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger()


class MessageType(str, Enum):
    COMMAND = "command"           # 点对点指令
    QUERY = "query"               # 请求-响应
    EVENT = "event"               # 广播事件
    HANDOFF = "handoff"           # 任务转移


class MessageStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Message:
    """总线消息。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: MessageType = MessageType.EVENT
    sender: str = ""              # agent_id
    recipient: str = ""           # agent_id (空=广播)
    topic: str = ""               # 消息主题
    payload: dict[str, Any] = field(default_factory=dict)
    status: MessageStatus = MessageStatus.PENDING
    created_at: float = field(default_factory=time.time)
    delivered_at: float = 0.0
    timeout: float = 30.0         # 超时秒数
    retry_count: int = 0
    max_retries: int = 3
    correlation_id: str = ""      # 关联消息链
    context: dict[str, Any] = field(default_factory=dict)

    def ack(self) -> None:
        self.status = MessageStatus.DELIVERED
        self.delivered_at = time.time()

    def fail(self) -> None:
        self.status = MessageStatus.FAILED


@dataclass
class HandoffRequest:
    """任务转移请求 — AWS Strands Agents 1.0 Handoff 语义。

    将当前任务连同上下文转移给另一个 agent。
    目标 agent 可以接受或拒绝。
    """

    from_agent: str
    to_agent: str
    task: str                              # 任务描述
    context: dict[str, Any] = field(default_factory=dict)
    priority: int = 0                      # 0=normal, 1=high, 2=urgent
    handoff_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    result_channel: str = ""               # 结果回调 topic
    accepted: bool = False
    accepted_at: float = 0.0


# 消息处理器签名
MessageHandler = Callable[[Message], Awaitable[None]]
HandoffHandler = Callable[[HandoffRequest], Awaitable[bool]]


class MessageBus:
    """发布/订阅消息总线。

    用法:
        bus = MessageBus()

        @bus.subscribe("research.complete")
        async def on_research_complete(msg: Message):
            ...

        await bus.publish(Message(topic="research.complete", payload={...}))
        reply = await bus.request(Message(recipient="analyst", payload={...}))
        accepted = await bus.handoff(HandoffRequest(from_agent="main", to_agent="researcher", task="..."))
    """

    def __init__(self) -> None:
        # topic -> [handler, ...]
        self._subscriptions: dict[str, list[MessageHandler]] = defaultdict(list)
        # agent_id -> handler (direct messaging)
        self._agent_handlers: dict[str, MessageHandler] = {}
        # handoff handlers
        self._handoff_handlers: dict[str, HandoffHandler] = {}
        # message_id -> Message (for tracking)
        self._pending: dict[str, Message] = {}
        # handoff_id -> HandoffRequest
        self._pending_handoffs: dict[str, HandoffRequest] = {}
        # agent_id -> [Message, ...] (mailbox for offline agents)
        self._mailboxes: dict[str, list[Message]] = defaultdict(list)

        # message_id -> asyncio.Event (for request-response)
        self._response_events: dict[str, Any] = {}

    # ── 订阅 ───────────────────────────────────────

    def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """订阅某个 topic。"""
        self._subscriptions[topic].append(handler)
        logger.debug("bus_subscribed", topic=topic)

    def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        subs = self._subscriptions.get(topic, [])
        if handler in subs:
            subs.remove(handler)

    def register_agent(self, agent_id: str, handler: MessageHandler,
                       handoff: HandoffHandler | None = None) -> None:
        """注册 agent 的消息处理器。"""
        self._agent_handlers[agent_id] = handler
        if handoff:
            self._handoff_handlers[agent_id] = handoff

    # ── 发布 ───────────────────────────────────────

    async def publish(self, msg: Message) -> None:
        """发布消息到指定 topic（或直接发送给 recipient）。"""
        msg.ack()

        # 点对点
        if msg.recipient and msg.recipient in self._agent_handlers:
            handler = self._agent_handlers[msg.recipient]
            await handler(msg)
            return

        # 广播到 topic 订阅者
        subs = self._subscriptions.get(msg.topic, [])
        if not subs:
            # 无人订阅，放入 mailbox
            if msg.recipient:
                self._mailboxes[msg.recipient].append(msg)
            logger.debug("bus_no_subscribers", topic=msg.topic)
            return

        for handler in subs:
            try:
                await handler(msg)
            except Exception as e:
                logger.error("bus_handler_error", topic=msg.topic, error=str(e))

    async def request(self, msg: Message, timeout: float = 30.0) -> Message | None:
        """同步请求-响应模式。发送消息并等待回复。"""
        import asyncio

        msg.type = MessageType.QUERY
        msg.timeout = timeout
        self._pending[msg.id] = msg

        event = asyncio.Event()
        reply_holder: dict[str, Message | None] = {"reply": None}

        async def reply_handler(reply: Message) -> None:
            reply_holder["reply"] = reply
            event.set()

        # 临时注册响应处理器
        reply_topic = f"_reply.{msg.id}"
        self.subscribe(reply_topic, reply_handler)

        try:
            await self.publish(msg)
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return reply_holder["reply"]
        except asyncio.TimeoutError:
            msg.status = MessageStatus.TIMEOUT
            logger.warning("bus_request_timeout", msg_id=msg.id)
            return None
        finally:
            self.unsubscribe(reply_topic, reply_handler)
            self._pending.pop(msg.id, None)

    async def reply_to(self, original_msg: Message, payload: dict[str, Any]) -> None:
        """回复一个请求消息。"""
        reply = Message(
            topic=f"_reply.{original_msg.id}",
            payload=payload,
            correlation_id=original_msg.id,
            type=MessageType.EVENT,
        )
        await self.publish(reply)

    # ── Handoff ────────────────────────────────────

    async def handoff(self, req: HandoffRequest) -> bool:
        """发起任务转移。返回是否被接受。

        目标 agent 通过 handoff_handler 决定是否接受。
        """
        self._pending_handoffs[req.handoff_id] = req

        handler = self._handoff_handlers.get(req.to_agent)
        if not handler:
            logger.warning("handoff_no_handler", to=req.to_agent)
            return False

        try:
            accepted = await handler(req)
            req.accepted = accepted
            if accepted:
                req.accepted_at = time.time()
                logger.info("handoff_accepted", from_=req.from_agent,
                            to=req.to_agent, task=req.task[:50])
            else:
                logger.info("handoff_rejected", from_=req.from_agent,
                            to=req.to_agent, task=req.task[:50])
            return accepted
        except Exception as e:
            logger.error("handoff_error", error=str(e))
            return False

    # ── 邮箱 ───────────────────────────────────────

    def deliver_mailbox(self, agent_id: str) -> list[Message]:
        """获取并清空 mailbox。agent 上线时调用。"""
        msgs = self._mailboxes.pop(agent_id, [])
        return msgs

    def has_pending(self, agent_id: str) -> bool:
        return len(self._mailboxes.get(agent_id, [])) > 0

    # ── 统计 ───────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        return {
            "subscriptions": sum(len(s) for s in self._subscriptions.values()),
            "topics": len(self._subscriptions),
            "registered_agents": len(self._agent_handlers),
            "pending_messages": len(self._pending),
            "pending_handoffs": len(self._pending_handoffs),
            "mailbox_messages": sum(len(m) for m in self._mailboxes.values()),
        }
