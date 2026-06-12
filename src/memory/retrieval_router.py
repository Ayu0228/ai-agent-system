"""检索路由器 —— 按文档优先级路由记忆查询。

对齐文档规格：
- 优先级权重：User(1.0) > Context(0.9) > Task(0.7) > Experience(0.6) > Knowledge(0.5)
- Token 预算分配：上下文 ≤30% + 系统 prompt 10% + 生成 20%
- 渐进式披露：元数据常驻 → 正文触发加载 → 捆绑资源按需调用
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.shared.models import (
    MemoryLayer,
    MemoryType,
    RetrievalBudget,
    RetrievalPriority,
)

logger = structlog.get_logger()

# 检索优先级配置表
RETRIEVAL_PRIORITIES: list[RetrievalPriority] = [
    RetrievalPriority(memory_type=MemoryType.USER_FACT, weight=1.0,
                      layer=MemoryLayer.L3_COLD, description="用户偏好最优先"),
    RetrievalPriority(memory_type=MemoryType.TASK_CONTEXT, weight=0.9,
                      layer=MemoryLayer.L1_HOT, description="当前对话必须知道"),
    RetrievalPriority(memory_type=MemoryType.TASK_STATE, weight=0.7,
                      layer=MemoryLayer.L2_WARM, description="近期相关任务有帮助"),
    RetrievalPriority(memory_type=MemoryType.EXPERIENCE, weight=0.6,
                      layer=MemoryLayer.L3_COLD, description="套路有用但别僵化"),
    RetrievalPriority(memory_type=MemoryType.KNOWLEDGE, weight=0.5,
                      layer=MemoryLayer.L3_COLD, description="事实基础，容易泛化"),
]


@dataclass
class RouteResult:
    """路由结果：告诉调用方查哪些记忆、顺序和预算。"""
    targets: list[tuple[MemoryType, float, MemoryLayer, list[str]]] = field(default_factory=list)
    token_budget: int = 0
    query_intent: str = ""


class RetrievalRouter:
    """检索路由器。根据查询意图决定记忆类型优先级和 token 预算。

    Phase 2 计划：将关键词计数升级为轻量分类器（BM25 + 语义向量），提升意图识别准确率。
    """

    def __init__(self, *, total_tokens: int = 128_000) -> None:
        self._budget = RetrievalBudget(total_tokens=total_tokens)

    # ── Route ─────────────────────────────────────────────

    def route(self, query: str, *, agent_id: str = "", has_active_task: bool = False) -> RouteResult:
        """分析查询意图，返回路由计划。

        路由逻辑：
        - 查询中含用户相关词（偏好/习惯/我）→ 用户记忆第一优先
        - 查询中涉及当前任务 → 任务记忆提前
        - 查询含"怎么做""经验""上次"→ 经验记忆提前
        - 默认按权重顺序

        使用 TF-IDF 风格的加权匹配替代简单关键词计数。
        """
        query_lower = query.lower()

        # 加权关键词：每个词有不同权重（高频通用词低权重，强信号词高权重）
        intent_keywords: dict[str, list[tuple[str, float]]] = {
            "user_profile": [
                ("我是", 1.0), ("我的", 0.9), ("偏好", 0.9), ("习惯", 0.8),
                ("喜欢", 0.6), ("讨厌", 0.6), ("profile", 0.5),
            ],
            "current_task": [
                ("任务", 1.0), ("进度", 0.9), ("正在", 0.8), ("状态", 0.7),
                ("task", 0.8), ("status", 0.5),
            ],
            "experience": [
                ("上次", 0.9), ("之前", 0.8), ("经验", 0.8), ("教训", 0.7),
                ("曾经", 0.6), ("experience", 0.6), ("lesson", 0.5),
            ],
            "knowledge": [
                ("什么是", 0.9), ("定义", 0.8), ("概念", 0.7), ("规则", 0.6),
                ("流程", 0.6), ("how to", 0.5), ("what is", 0.5),
            ],
        }

        intents: dict[str, float] = {}
        for intent, weighted_kw in intent_keywords.items():
            score = 0.0
            max_possible = sum(w for _, w in weighted_kw)
            for kw, weight in weighted_kw:
                if kw in query_lower:
                    score += weight
            normalized = score / max_possible if max_possible > 0 else 0.0
            if normalized > 0:
                intents[intent] = normalized

        # 确定主导意图（提高阈值到 0.15 减少误判）
        if "user_profile" in intents and intents["user_profile"] > 0.15:
            primary = "user_profile"
        elif "current_task" in intents and intents["current_task"] > 0.15:
            primary = "current_task"
        elif "experience" in intents and intents["experience"] > 0.15:
            primary = "experience"
        else:
            primary = "general"

        return self._build_route(primary, has_active_task=has_active_task)

    def _build_route(self, intent: str, *, has_active_task: bool) -> RouteResult:
        """根据意图构建检索计划。"""
        targets: list[tuple[MemoryType, float, MemoryLayer, list[str]]] = []

        if intent == "user_profile":
            targets = [
                (MemoryType.USER_FACT, 1.0, MemoryLayer.L3_COLD,
                 ["查询 user_facts 表 → 用户画像加载到 system prompt"]),
                (MemoryType.TASK_CONTEXT, 0.9, MemoryLayer.L1_HOT,
                 ["直接读取上下文"]),
            ]
            if has_active_task:
                targets.append((MemoryType.TASK_STATE, 0.7, MemoryLayer.L2_WARM,
                                ["查询 SQLite tasks 表活跃任务"]))
            targets.append((MemoryType.KNOWLEDGE, 0.5, MemoryLayer.L3_COLD,
                            ["ChromaDB 宽召回 → Rerank 精排"]))
        elif intent == "experience":
            targets = [
                (MemoryType.EXPERIENCE, 0.6, MemoryLayer.L3_COLD,
                 ["向量检索经验 → 按 weight×recency_boost → Top3"]),
                (MemoryType.TASK_CONTEXT, 0.9, MemoryLayer.L1_HOT,
                 ["直接读取上下文"]),
                (MemoryType.KNOWLEDGE, 0.5, MemoryLayer.L3_COLD,
                 ["ChromaDB 查询补充 → 相似度 ≥ baseline → Top3"]),
            ]
        elif intent == "current_task":
            targets = [
                (MemoryType.TASK_STATE, 0.7, MemoryLayer.L2_WARM,
                 ["查询 SQLite tasks 表 → 按 updated_at 倒序 → 取最近 5 个"]),
                (MemoryType.TASK_CONTEXT, 0.9, MemoryLayer.L1_HOT,
                 ["直接读取上下文"]),
                (MemoryType.EXPERIENCE, 0.6, MemoryLayer.L3_COLD,
                 ["向量检索经验 → 按 weight×recency_boost 排序 → Top3"]),
            ]
        else:  # general — 默认权重顺序
            targets = [
                (MemoryType.USER_FACT, 1.0, MemoryLayer.L3_COLD,
                 ["用户画像优先：每次对话前先读"]),
                (MemoryType.TASK_CONTEXT, 0.9, MemoryLayer.L1_HOT,
                 ["直接读取内存中的对话历史 → 超过 20 轮读摘要"]),
            ]
            if has_active_task:
                targets.append((MemoryType.TASK_STATE, 0.7, MemoryLayer.L2_WARM,
                                ["查询活跃任务 → 过滤 status!='done' 或 72h 内"]))
            targets.extend([
                (MemoryType.EXPERIENCE, 0.6, MemoryLayer.L3_COLD,
                 ["向量检索经验 → 按 weight×recency_boost → Top3 → 只读 shareable 或自己的"]),
                (MemoryType.KNOWLEDGE, 0.5, MemoryLayer.L3_COLD,
                 ["ChromaDB 查询 → 相似度 ≥ baseline → Top5 → 图谱精确查询"]),
            ])

        # 计算 token 预算
        available = int(self._budget.total_tokens * self._budget.context_max)
        per_target = available // max(len(targets), 1)

        return RouteResult(
            targets=targets,
            token_budget=per_target,
            query_intent=intent,
        )

    # ── Budget ────────────────────────────────────────────

    def get_budget_allocation(self) -> dict[str, Any]:
        """获取当前 token 预算分配。"""
        return {
            "total": self._budget.total_tokens,
            "context_max_pct": self._budget.context_max,
            "context_max_tokens": int(self._budget.total_tokens * self._budget.context_max),
            "system_prompt_pct": self._budget.system_prompt,
            "system_prompt_tokens": int(self._budget.total_tokens * self._budget.system_prompt),
            "generation_pct": self._budget.generation,
            "generation_tokens": int(self._budget.total_tokens * self._budget.generation),
            "remaining": int(self._budget.total_tokens * (1 - self._budget.context_max - self._budget.system_prompt - self._budget.generation)),
        }

    @staticmethod
    def priorities() -> list[dict[str, Any]]:
        """返回优先级列表（供查询用）。"""
        return [
            {"type": p.memory_type.value, "weight": p.weight, "layer": p.layer.value, "desc": p.description}
            for p in RETRIEVAL_PRIORITIES
        ]
