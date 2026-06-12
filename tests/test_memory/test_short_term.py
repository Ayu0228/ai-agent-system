"""ContextCompressor 单元测试。"""

from src.memory.short_term import ContextCompressor


class TestContextCompressor:
    """测试上下文压缩器。"""

    def test_no_compress_when_under_limit(self):
        compressor = ContextCompressor(max_tokens=128000)
        messages = [
            {"role": "user", "content": "short message"},
            {"role": "assistant", "content": "short reply"},
        ]
        assert compressor.should_compress(messages) is False

    def test_compress_preserves_recent(self):
        compressor = ContextCompressor(max_tokens=100, recent_turns=3)
        messages = [{"role": "user", "content": "x" * 1000}] * 20 + [
            {"role": "user", "content": "recent 1"},
            {"role": "assistant", "content": "recent 2"},
            {"role": "user", "content": "recent 3"},
        ]

        result = compressor.compress(messages)
        # 应有摘要 + 最近 3 条
        assert len(result) >= 3
        assert result[-1]["content"] == "recent 3"

    def test_not_enough_messages_to_compress(self):
        compressor = ContextCompressor(max_tokens=10, recent_turns=100)
        messages = [{"role": "user", "content": "hi"}] * 5

        # should_compress 根据总字符数判断
        result = compressor.compress(messages)
        assert len(result) == 5  # 不够压缩

    def test_estimate_tokens(self):
        compressor = ContextCompressor()
        assert compressor.estimate_tokens("hello world") == 2
        assert compressor.estimate_tokens("") == 0

    def test_compressed_messages_have_summary(self):
        compressor = ContextCompressor(max_tokens=50, recent_turns=2)
        messages = [{"role": "user", "content": "very long " + "x" * 500}] * 10 + [
            {"role": "user", "content": "final question"},
            {"role": "assistant", "content": "final answer"},
        ]

        result = compressor.compress(messages)
        # 第一条应该是摘要
        assert result[0]["role"] == "system"
        assert "上下文摘要" in result[0]["content"]
