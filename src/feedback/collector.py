"""Feedback Collector — 用户反馈收集与分类。

ref: LangChain data flywheel — production monitoring creates feedback loop
ref: Anthropic — human feedback at checkpoints

反馈类型:
  - RATING: 用户评分 (1-5)
  - CORRECTION: 用户纠正了错误回答
  - HALLUCINATION: 用户标记了幻觉
  - TOOL_ERROR: 工具调用失败
  - LATENCY: 响应太慢
  - FEATURE_REQUEST: 功能建议
  - BUG_REPORT: Bug 报告
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class FeedbackType(str, Enum):
    RATING = "rating"
    CORRECTION = "correction"
    HALLUCINATION = "hallucination"
    TOOL_ERROR = "tool_error"
    LATENCY = "latency"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    OTHER = "other"


@dataclass
class FeedbackEntry:
    """用户反馈条目。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: FeedbackType = FeedbackType.OTHER
    agent_id: str = ""
    workflow_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    user_id: str = ""
    rating: int = 0                          # 1-5 (仅 RATING 类型)
    comment: str = ""
    expected_output: str = ""                # 用户期望的正确输出
    actual_output: str = ""                  # agent 实际输出
    severity: str = "medium"                 # low / medium / high / critical
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackCollector:
    """反馈收集器。

    用法:
        collector = FeedbackCollector()
        collector.submit(FeedbackEntry(
            type=FeedbackType.HALLUCINATION,
            agent_id="researcher",
            comment="引用了不存在的论文",
            expected_output="应该标注来源",
        ))
        stats = collector.get_stats(agent_id="researcher")
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: list[FeedbackEntry] = []
        self._max_entries = max_entries
        # 实时聚合
        self._agent_scores: dict[str, list[int]] = defaultdict(list)
        self._type_counts: dict[str, int] = defaultdict(int)

    # ── 提交 ───────────────────────────────────────

    def submit(self, entry: FeedbackEntry) -> str:
        """提交反馈，返回 entry id。"""
        self._entries.append(entry)

        if entry.rating > 0:
            self._agent_scores[entry.agent_id].append(entry.rating)
        self._type_counts[entry.type.value] += 1

        # 环形缓冲
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        logger.info("feedback_submitted", id=entry.id, type=entry.type.value,
                    agent=entry.agent_id, severity=entry.severity)
        return entry.id

    def submit_rating(self, agent_id: str, rating: int,
                      session_id: str = "", comment: str = "") -> str:
        """快捷评分提交。"""
        return self.submit(FeedbackEntry(
            type=FeedbackType.RATING,
            agent_id=agent_id,
            session_id=session_id,
            rating=max(1, min(5, rating)),
            comment=comment,
        ))

    def submit_correction(self, agent_id: str, expected: str,
                          actual: str, comment: str = "") -> str:
        """快捷纠正提交。"""
        return self.submit(FeedbackEntry(
            type=FeedbackType.CORRECTION,
            agent_id=agent_id,
            expected_output=expected,
            actual_output=actual,
            comment=comment,
            severity="high",
        ))

    # ── 查询 ───────────────────────────────────────

    def get_by_agent(self, agent_id: str, limit: int = 50) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.agent_id == agent_id][-limit:]

    def get_by_type(self, fb_type: FeedbackType, limit: int = 50) -> list[FeedbackEntry]:
        return [e for e in self._entries if e.type == fb_type][-limit:]

    def get_unresolved(self, severity: str = "high") -> list[FeedbackEntry]:
        """获取未解决的高严重度反馈。"""
        sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        threshold = sev_order.get(severity, 0)
        return [e for e in self._entries
                if not e.resolved and sev_order.get(e.severity, 0) >= threshold]

    def resolve(self, entry_id: str, resolution: str = "") -> bool:
        for e in self._entries:
            if e.id == entry_id:
                e.resolved = True
                e.resolution = resolution
                return True
        return False

    # ── 统计 ───────────────────────────────────────

    def get_agent_rating(self, agent_id: str) -> dict[str, Any]:
        scores = self._agent_scores.get(agent_id, [])
        if not scores:
            return {"agent_id": agent_id, "avg_rating": 0, "count": 0}
        return {
            "agent_id": agent_id,
            "avg_rating": sum(scores) / len(scores),
            "count": len(scores),
            "distribution": {str(i): scores.count(i) for i in range(1, 6)},
        }

    def get_stats(self, agent_id: str = "") -> dict[str, Any]:
        entries = self._entries
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]

        severity_dist = defaultdict(int)
        type_dist = defaultdict(int)
        for e in entries:
            severity_dist[e.severity] += 1
            type_dist[e.type.value] += 1

        return {
            "total": len(entries),
            "resolved": sum(1 for e in entries if e.resolved),
            "unresolved": sum(1 for e in entries if not e.resolved),
            "by_severity": dict(severity_dist),
            "by_type": dict(type_dist),
            "agent_ratings": {aid: self.get_agent_rating(aid)["avg_rating"]
                              for aid in self._agent_scores},
        }

    @property
    def entry_count(self) -> int:
        return len(self._entries)
