"""短期记忆管理：上下文窗口压缩。"""

from __future__ import annotations


class ContextCompressor:
    """分层压缩：近期保持原文，中期做摘要，早期只保留关键事实。

    对齐 AI Agent 记忆系统设计文档规格：
    - 128K tokens 上下文窗口
    - 20 轮甜点区（LoCoMo 验证）
    - 80% 阈值触发压缩（留 20% buffer）
    - 2 层摘要深度
    - 上下文记忆占用不超过 30%
    """

    def __init__(
        self,
        max_tokens: int = 128_000,
        recent_turns: int = 20,
        system_prompt_reserved: int = 4_000,
        compress_threshold: float = 0.8,
        max_context_budget: float = 0.3,
    ) -> None:
        self.max_tokens = max_tokens
        self.recent_turns = recent_turns
        self.system_prompt_reserved = system_prompt_reserved
        self.compress_threshold = compress_threshold
        self.max_context_budget = max_context_budget
        self._summary_layer1: str = ""  # 最近一次摘要
        self._summary_layer2: str = ""  # 元摘要（更早的摘要再压缩）
        self._compression_count: int = 0

    @property
    def available_tokens(self) -> int:
        """扣除 system prompt 后的可用 token 数。"""
        return self.max_tokens - self.system_prompt_reserved

    @property
    def context_budget_tokens(self) -> int:
        """上下文记忆最大 token 预算。"""
        return int(self.max_tokens * self.max_context_budget)

    def should_compress(self, messages: list[dict]) -> bool:
        """判断是否需要压缩（80% 阈值）。使用统一的 token 估算。"""
        text = " ".join(str(m.get("content", "")) for m in messages)
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens > self.available_tokens * self.compress_threshold

    def compress(self, messages: list[dict]) -> list[dict]:
        """压缩消息列表。近期 20 轮保持原文，早期合并为摘要。

        2 层摘要深度：
        - Layer 1：最近压缩的摘要
        - Layer 2：更早摘要的元摘要（防关键细节丢失）
        """
        if not self.should_compress(messages):
            return messages

        if len(messages) <= self.recent_turns:
            return messages

        self._compression_count += 1

        recent = messages[-self.recent_turns:]
        older = messages[: -self.recent_turns]

        # 提取早期消息关键信息
        key_points: list[str] = []
        for m in older:
            content = str(m.get("content", ""))
            if len(content) > 200:
                key_points.append(content[:200] + "...")
            else:
                key_points.append(content)

        # Layer 1 摘要
        new_summary = (
            "[上下文摘要] 以下为此前对话的关键信息：\n"
            + "\n".join(f"- {p}" for p in key_points[-20:])
        )

        # 如果有旧摘要，压缩为 Layer 2 元摘要
        if self._summary_layer1:
            self._summary_layer2 = (
                "[更早的上下文] "
                + self._summary_layer1[:300]
                + "..."
            )

        self._summary_layer1 = new_summary

        # 构建压缩后的消息列表
        result: list[dict] = []
        if self._summary_layer2:
            result.append({"role": "system", "content": self._summary_layer2})
        result.append({"role": "system", "content": self._summary_layer1})
        result.extend(recent)

        # 上下文预算保护：不超过 30%
        while self.estimate_tokens(str(result)) > self.context_budget_tokens and len(recent) > 5:
            recent = recent[-5:]
            result = []
            if self._summary_layer2:
                result.append({"role": "system", "content": self._summary_layer2})
            result.append({"role": "system", "content": self._summary_layer1})
            result.extend(recent)

        return result

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """估算文本的 token 数量。

        优先使用 tiktoken（精确），不可用时回退到 CJK/Latin 加权估算。
        """
        if not text:
            return 0
        try:
            import tiktoken
            enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(text))
        except Exception:
            pass
        # 回退：CJK 约 1.5 字符/token，Latin 约 4 字符/token
        cjk = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿')
        latin = len(text) - cjk
        est = cjk / 1.5 + latin / 4.0
        return max(1, int(est))
