"""幻觉检测守卫 — 多层验证管线。

层次:
  1. 模式匹配 — 标记常见幻觉用语（成本极低）
  2. 事实一致性 — 输出 vs 检索到的上下文的矛盾检测
  3. 置信度校准 — 输出置信度与事实性的一致性检查
  4. Abstention — 低置信度时拒绝回答

ref: Luna — DeBERTa-large, 97% cost reduction vs GPT-3.5 evaluators
ref: Granite Guardian (IBM) — open source, all risk dimensions
ref: HalluGuard — 4B SRM, evidence-grounded justifications
ref: HALT-RAG — dual-NLI ensemble, calibrated abstention
ref: Token-Guard — decoding-time intervention
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# 常见幻觉标记模式
_HALLUCINATION_PATTERNS: list[tuple[str, float]] = [
    # 中文
    ("据可靠消息", 0.3),
    ("据内部人士透露", 0.5),
    ("据不愿透露姓名", 0.6),
    ("据知情人士", 0.4),
    ("据业内人士", 0.3),
    ("据悉尼", 0.1),
    ("据匿名", 0.6),
    ("消息人士称", 0.4),
    ("有消息称", 0.2),
    # 英文
    ("I made this up", 0.9),
    ("I don't know but I'll guess", 0.9),
    ("I'm not sure but", 0.5),
    ("let me imagine", 0.7),
    ("hypothetically speaking", 0.4),
    ("sources say", 0.2),
    ("it is believed that", 0.3),
    ("reports suggest", 0.2),
]

# 编造引用模式
_FAKE_CITATION_PATTERNS: list[str] = [
    r"\[\d+\].*\[引用.*不可用\]",
    r"\[来源:\s*(未知|不明|佚名|anonymous|unknown)\]",
    r"Source:\s*(unknown|anonymous|佚名)",
    r"来源：\s*(未知|不明|网络)",
]

# 过度确定性模式（数字过于精确 = 可能幻觉）
_OVERPRECISION_PATTERNS: list[str] = [
    r"\d{2,4}\.\d{2,3}%",       # "78.43%" — 通常不可能这么精确
    r"精确到\s*\d+位小数",
]

# 健康输出标记
_GROUNDED_MARKERS: list[str] = [
    "根据.*研究",
    "根据.*报告",
    "根据.*数据",
    "据.*统计",
    "Source:",
    "参考：",
    "引用：",
    "参见：",
]


@dataclass
class HallucinationResult:
    """幻觉检测结果。"""

    score: float = 0.0                # 0=完全幻觉, 1=完全可靠
    hallucination_detected: bool = False
    risk_level: str = "low"           # low / medium / high / critical
    markers_found: list[str] = field(default_factory=list)
    groundedness_score: float = 0.0   # 是否有据可查
    suggestion: str = ""              # 建议操作


@dataclass
class AbstentionDecision:
    """拒绝回答决策。"""

    should_abstain: bool = False
    confidence: float = 0.0
    reason: str = ""
    fallback_response: str = ""


class HallucinationGuard:
    """幻觉检测守卫。

    用法:
        guard = HallucinationGuard()
        result = guard.check(output_text, context=retrieved_docs)
        if result.hallucination_detected:
            logger.warning("hallucination_detected", risk=result.risk_level)
    """

    def __init__(self) -> None:
        self._pattern_cache: dict[str, float] = {}

    # ── 核心方法 ───────────────────────────────────

    def check(
        self,
        output: str,
        *,
        context: str = "",
        expected_facts: list[str] | None = None,
    ) -> HallucinationResult:
        """对输出进行多层幻觉检测。

        Args:
            output: Agent 输出文本
            context: 用于生成回答的上下文（RAG 检索结果等）
            expected_facts: 期望包含的事实列表
        """
        markers: list[str] = []
        total_penalty = 0.0

        # Layer 1: 模式匹配
        pattern_penalty, pattern_markers = self._check_patterns(output)
        total_penalty = max(total_penalty, pattern_penalty)
        markers.extend(pattern_markers)

        # Layer 2: 编造引用检测
        fake_cite_penalty, fake_markers = self._check_fake_citations(output)
        total_penalty = max(total_penalty, fake_cite_penalty)
        markers.extend(fake_markers)

        # Layer 3: 过度精确检测
        overprec_penalty, overprec_markers = self._check_overprecision(output)
        total_penalty = max(total_penalty, overprec_penalty)
        markers.extend(overprec_markers)

        # Layer 4: 事实一致性（如果提供了上下文）
        consistency = 1.0
        if context:
            consistency = self._check_factual_consistency(output, context)

        # Layer 5: 期望事实覆盖（如果提供了）
        fact_coverage = 1.0
        if expected_facts:
            fact_coverage = self._check_fact_coverage(output, expected_facts)

        # Layer 6: 有据可查评分
        groundedness = self._check_groundedness(output)

        # 综合评分
        base_score = 1.0 - total_penalty
        score = base_score * consistency * fact_coverage * (0.5 + 0.5 * groundedness)
        score = max(0.0, min(1.0, score))

        # 风险等级
        if score < 0.3:
            risk = "critical"
            suggestion = "BLOCK: 输出可能包含严重幻觉，建议阻断"
        elif score < 0.5:
            risk = "high"
            suggestion = "WARN: 输出可信度低，建议人工审核"
        elif score < 0.7:
            risk = "medium"
            suggestion = "REVIEW: 输出有可疑标记，标记为低置信度"
        else:
            risk = "low"
            suggestion = ""

        return HallucinationResult(
            score=score,
            hallucination_detected=score < 0.5,
            risk_level=risk,
            markers_found=markers,
            groundedness_score=groundedness,
            suggestion=suggestion,
        )

    # ── 各层次检测 ─────────────────────────────────

    def _check_patterns(self, text: str) -> tuple[float, list[str]]:
        """Layer 1: 模式匹配。"""
        penalty = 0.0
        markers: list[str] = []
        text_lower = text.lower()

        for pattern, weight in _HALLUCINATION_PATTERNS:
            if pattern.lower() in text_lower:
                penalty = max(penalty, weight)
                markers.append(f"pattern:{pattern}")

        return penalty, markers

    def _check_fake_citations(self, text: str) -> tuple[float, list[str]]:
        """Layer 2: 编造引用检测。"""
        markers: list[str] = []
        for pat in _FAKE_CITATION_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                markers.append(f"fake_citation:{pat[:30]}")
        penalty = min(0.9, len(markers) * 0.3)
        return penalty, markers

    def _check_overprecision(self, text: str) -> tuple[float, list[str]]:
        """Layer 3: 过度精确检测。"""
        markers: list[str] = []
        for pat in _OVERPRECISION_PATTERNS:
            matches = re.findall(pat, text)
            if len(matches) >= 3:  # 3+ 精确数字 = 可疑
                markers.append(f"overprecision:{len(matches)}_matches")
        penalty = min(0.4, len(markers) * 0.15)
        return penalty, markers

    def _check_factual_consistency(self, output: str, context: str) -> float:
        """Layer 4: 事实一致性 — 输出与上下文的词级 overlap。

        注意: 这是轻量近似，生产环境应替换为 NLI 模型。
        """
        if not context:
            return 1.0

        # 从输出提取核心实体/数字
        output_words = set(output.lower().split())
        context_words = set(context.lower().split())

        if not output_words:
            return 1.0

        # 简单 overlap 比例
        overlap = output_words & context_words
        union = output_words | context_words

        if len(union) == 0:
            return 0.5

        jaccard = len(overlap) / len(union)

        # 如果 overlap 太低，可能是编造
        if jaccard < 0.05 and len(output) > 100:
            return 0.4
        if jaccard < 0.10 and len(output) > 200:
            return 0.6

        return min(1.0, 0.5 + jaccard * 5)  # 映射到 0.5-1.0

    def _check_fact_coverage(
        self, output: str, expected_facts: list[str]
    ) -> float:
        """Layer 5: 期望事实覆盖率。"""
        if not expected_facts:
            return 1.0

        output_lower = output.lower()
        hits = sum(1 for fact in expected_facts if fact.lower() in output_lower)
        return hits / len(expected_facts)

    def _check_groundedness(self, text: str) -> float:
        """Layer 6: 有据可查评分。"""
        if not text:
            return 0.0

        text_lower = text.lower()
        markers_found = sum(
            1 for m in _GROUNDED_MARKERS if re.search(m, text_lower)
        )

        # 长文本应该有更多引用标记
        expected = max(1, len(text) / 500)  # 每 500 字符期望 1 个引用
        return min(1.0, markers_found / expected)

    # ── Abstention ─────────────────────────────────

    def should_abstain(
        self, output: str, *, min_confidence: float = 0.5
    ) -> AbstentionDecision:
        """决定是否应该拒绝回答。"""
        result = self.check(output)

        if result.score < min_confidence:
            return AbstentionDecision(
                should_abstain=True,
                confidence=result.score,
                reason=f"Hallucination score {result.score:.2f} below threshold {min_confidence}",
                fallback_response=(
                    "抱歉，我无法确认这个信息。"
                    if result.risk_level in ("critical", "high")
                    else "我对这个问题的把握不够，建议核实相关信息。"
                ),
            )

        return AbstentionDecision(confidence=result.score)


# 模块级单例
_hallucination_guard: HallucinationGuard | None = None


def get_hallucination_guard() -> HallucinationGuard:
    global _hallucination_guard
    if _hallucination_guard is None:
        _hallucination_guard = HallucinationGuard()
    return _hallucination_guard
