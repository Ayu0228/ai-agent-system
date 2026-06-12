"""记忆写入决策策略。决定什么值得记住、什么应该遗忘。

对齐文档规格：
- WritePolicy 适用于通用记忆（知识/经验）
- 用户记忆的三层过滤由 UserMemory 内部处理
- 轻量预过滤 90% 无效文本 → 剩余 10% 送 LLM 终审
"""

from __future__ import annotations

import re

from src.shared.models import MemoryEntry, MemorySearchResult, WriteDecision, WriteEvaluation


class WritePolicy:
    """评估内容的新颖性、重要性、可靠性，决定是否写入记忆。"""

    def __init__(
        self,
        *,
        min_importance: float = 0.4,
        dedup_threshold: float = 0.9,
    ) -> None:
        self.min_importance = min_importance
        self.dedup_threshold = dedup_threshold

    def evaluate(
        self,
        entry: MemoryEntry,
        existing: list[MemorySearchResult],
    ) -> WriteEvaluation:
        """决定是否写入以及写入策略。"""
        # 轻量预过滤：纯噪音文本直接丢弃
        if self._is_noise(entry.content):
            return WriteEvaluation(
                decision=WriteDecision.SKIP,
                importance=0.0,
                reason="noise_filtered",
            )

        # 去重检查
        if existing and existing[0].score > self.dedup_threshold:
            return WriteEvaluation(
                decision=WriteDecision.MERGE,
                importance=entry.importance,
                reason=f"similar_memory_exists (score={existing[0].score:.2f})",
            )

        # 重要性评估（含新型性加分）
        importance = self._assess_importance(entry)
        entry.importance = importance

        if importance < self.min_importance:
            return WriteEvaluation(
                decision=WriteDecision.SKIP,
                importance=importance,
                reason=f"importance_below_threshold ({importance:.2f} < {self.min_importance})",
            )

        return WriteEvaluation(
            decision=WriteDecision.WRITE,
            importance=importance,
            reason="novel_and_important",
        )

    def should_persist(self, content: str, *, memory_type: str = "knowledge") -> bool:
        """LLM 判定是否为"持久知识"（配合 LLM 终审使用）。"""
        if self._is_noise(content):
            return False

        # 检查是否有实质性内容
        if len(content) < 20:
            return False

        # 临时性信息过滤
        temp_patterns = [
            r"目前正在", r"现在", r"刚才", r"刚刚",
            r"今天天气", r"当前时间", r"正在输入",
        ]
        for pat in temp_patterns:
            if re.search(pat, content):
                return False

        return True

    @staticmethod
    def _is_noise(content: str) -> bool:
        """轻量预过滤：识别纯噪音文本。"""
        if not content or len(content.strip()) < 5:
            return True

        # 纯标点/数字/空白
        cleaned = content.strip().replace(" ", "").replace("\n", "")
        if len(cleaned) < 3:
            return True

        # 纯 URL（超出合理范围的链接文本）
        if content.startswith("http") and len(content.split()) <= 2:
            return True

        # 纯单字重复
        if len(set(cleaned)) <= 2 and len(cleaned) > 5:
            return True

        return False

    @staticmethod
    def _assess_importance(entry: MemoryEntry) -> float:
        """评估记忆重要性（0-1）。"""
        score = entry.importance  # 外部传入的基础分

        # 内容长度加权（太短或太长都降分）
        length = len(entry.content)
        if length < 20:
            score *= 0.5
        elif length > 1500:
            score *= 0.8

        # 标签相关性
        priority_tags = {"preference", "goal", "constraint", "identity", "error", "fix", "pattern"}
        tag_match = len(set(entry.tags) & priority_tags)
        if tag_match > 0:
            score += tag_match * 0.1

        return min(score, 1.0)
