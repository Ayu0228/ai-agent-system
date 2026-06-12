"""Feedback Analyzer — 反馈分析、趋势检测、改进工单生成。

ref: LangChain — analyzing production feedback for systematic improvements
ref: Anthropic — using evaluation results to drive prompt and tool improvements
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.feedback.collector import FeedbackEntry, FeedbackType

logger = structlog.get_logger()


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"


@dataclass
class AnalysisResult:
    """分析结果。"""
    agent_id: str
    period_days: int = 7
    total_feedback: int = 0
    avg_rating: float = 0.0
    rating_trend: str = "stable"             # improving / stable / declining
    top_issues: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    alert: bool = False
    alert_reason: str = ""


@dataclass
class ImprovementTicket:
    """改进工单 — 从反馈分析自动生成。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    agent_id: str = ""
    severity: str = "medium"
    source_feedback_ids: list[str] = field(default_factory=list)
    suggested_action: str = ""               # prompt_update / tool_fix / model_change / human_review
    status: TicketStatus = TicketStatus.OPEN
    created_at: float = field(default_factory=time.time)
    assigned_to: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackAnalyzer:
    """反馈分析器。

    用法:
        analyzer = FeedbackAnalyzer(collector)
        report = analyzer.analyze(agent_id="researcher", period_days=7)
        if report.alert:
            tickets = analyzer.generate_tickets(report)
    """

    def __init__(self, collector=None):  # FeedbackCollector | None
        self._collector = collector
        self._tickets: list[ImprovementTicket] = []

    def set_collector(self, collector) -> None:
        self._collector = collector

    # ── 分析 ───────────────────────────────────────

    def analyze(self, agent_id: str = "", period_days: int = 7) -> AnalysisResult:
        """分析指定时间窗口的反馈趋势。"""
        if not self._collector:
            return AnalysisResult(agent_id=agent_id, period_days=period_days)

        entries = self._collector._entries
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]

        cutoff = time.time() - period_days * 86400
        recent = [e for e in entries if e.created_at >= cutoff]

        if not recent:
            return AnalysisResult(agent_id=agent_id, period_days=period_days)

        result = AnalysisResult(
            agent_id=agent_id,
            period_days=period_days,
            total_feedback=len(recent),
        )

        # 平均评分
        ratings = [e.rating for e in recent if e.rating > 0]
        if ratings:
            result.avg_rating = sum(ratings) / len(ratings)

        # 趋势检测: 比较前后半段
        mid = cutoff + period_days * 86400 / 2
        first_half = [e.rating for e in recent if e.created_at < mid and e.rating > 0]
        second_half = [e.rating for e in recent if e.created_at >= mid and e.rating > 0]
        if first_half and second_half:
            avg1 = sum(first_half) / len(first_half)
            avg2 = sum(second_half) / len(second_half)
            if avg2 > avg1 + 0.5:
                result.rating_trend = "improving"
            elif avg2 < avg1 - 0.5:
                result.rating_trend = "declining"
            else:
                result.rating_trend = "stable"

        # 高频问题分析
        issue_map: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "type": "", "examples": []})
        for e in recent:
            key = f"{e.type.value}:{e.severity}"
            issue_map[key]["count"] += 1
            issue_map[key]["type"] = e.type.value
            if len(issue_map[key]["examples"]) < 3:
                issue_map[key]["examples"].append(e.comment[:100] if e.comment else "(no comment)")

        result.top_issues = sorted(
            issue_map.values(), key=lambda x: x["count"], reverse=True
        )[:5]

        # 告警条件
        hallucination_count = sum(1 for e in recent if e.type == FeedbackType.HALLUCINATION)
        high_sev_count = sum(1 for e in recent if e.severity in ("high", "critical"))

        if result.rating_trend == "declining" and result.avg_rating < 3.0:
            result.alert = True
            result.alert_reason = f"评分下降趋势，当前均分 {result.avg_rating:.1f}"
        elif hallucination_count >= 5:
            result.alert = True
            result.alert_reason = f"幻觉报告激增: {hallucination_count} 次"
        elif high_sev_count >= 3:
            result.alert = True
            result.alert_reason = f"高严重度问题: {high_sev_count} 次"

        # 推荐
        if result.alert:
            result.recommendations = self._generate_recommendations(recent, result)

        logger.info("feedback_analysis", agent=agent_id, alert=result.alert,
                    avg_rating=f"{result.avg_rating:.1f}",
                    trend=result.rating_trend)
        return result

    def _generate_recommendations(self, entries: list[FeedbackEntry],
                                  result: AnalysisResult) -> list[str]:
        """基于反馈模式生成改进建议。"""
        recs: list[str] = []

        type_counts = defaultdict(int)
        for e in entries:
            type_counts[e.type.value] += 1

        if type_counts.get("hallucination", 0) > type_counts.get("correction", 0):
            recs.append("建议加强幻觉检测阈值或更新 Prompt 中的引用要求")
        if type_counts.get("tool_error", 0) > 2:
            recs.append("工具调用错误频繁，建议检查工具 ACI 文档和错误处理")
        if type_counts.get("latency", 0) > 3:
            recs.append("延迟投诉较多，建议评估模型降级或缓存策略")
        if result.rating_trend == "declining":
            recs.append("评分持续下降，建议进行 Prompt 回归测试")
        if not recs:
            recs.append("进行人工审核以确定改进方向")

        return recs

    # ── 工单生成 ───────────────────────────────────

    def generate_tickets(self, analysis: AnalysisResult) -> list[ImprovementTicket]:
        """根据分析结果自动生成改进工单。"""
        tickets: list[ImprovementTicket] = []

        for issue in analysis.top_issues:
            issue_type = issue.get("type", "other")
            count = issue.get("count", 0)
            if count < 2:
                continue

            action = "human_review"
            if issue_type == "hallucination":
                action = "prompt_update"
            elif issue_type == "tool_error":
                action = "tool_fix"
            elif issue_type == "latency":
                action = "model_change"

            ticket = ImprovementTicket(
                title=f"[{analysis.agent_id or 'all'}] {issue_type} — {count} reports",
                description=f"过去 {analysis.period_days} 天内收到 {count} 次 {issue_type} 类型反馈。\n"
                            f"示例: {', '.join(issue.get('examples', []))}",
                agent_id=analysis.agent_id,
                severity="high" if count >= 5 else "medium",
                suggested_action=action,
            )
            tickets.append(ticket)
            self._tickets.append(ticket)

        logger.info("tickets_generated", count=len(tickets))
        return tickets

    def get_open_tickets(self, agent_id: str = "") -> list[ImprovementTicket]:
        tickets = self._tickets
        if agent_id:
            tickets = [t for t in tickets if t.agent_id == agent_id]
        return [t for t in tickets if t.status == TicketStatus.OPEN]

    def resolve_ticket(self, ticket_id: str) -> bool:
        for t in self._tickets:
            if t.id == ticket_id:
                t.status = TicketStatus.RESOLVED
                return True
        return False
