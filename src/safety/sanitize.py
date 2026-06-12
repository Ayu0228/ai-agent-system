"""输入过滤 + Prompt 注入防御。"""

from __future__ import annotations

import re

from src.shared.constants import BLOCKED_PATTERNS


def sanitize_input(user_input: str) -> tuple[str, bool]:
    """检查用户输入是否包含注入 payload。返回 (sanitized_text, was_blocked)。"""
    compiled = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]
    for pattern in compiled:
        if pattern.search(user_input):
            return ("[输入被安全策略拦截 - 检测到潜在注入攻击]", True)
    return (user_input, False)


def contains_pii(text: str) -> bool:
    """检查是否包含 PII（简化版）。"""
    pii_patterns = [
        r"\b\d{11}\b",                          # 手机号（11位）
        r"\b\d{17}[\dXx]\b",                    # 身份证号
        r"\b[\w.-]+@[\w.-]+\.\w{2,}\b",         # 邮箱
        r"\b(?:AKIA|sk-|sk_live_)[\w-]{16,}\b", # AWS/OpenAI API Key
        r"\bghp_[a-zA-Z0-9]{36}\b",              # GitHub Personal Access Token
    ]
    for p in pii_patterns:
        if re.search(p, text):
            return True
    return False


def contains_system_prompt_leak(text: str) -> bool:
    """检查输出是否可能泄露 System Prompt。"""
    leak_markers = [
        "system prompt", "系统提示词", "system instructions",
        "<system>", "<|im_start|>system", "[SYSTEM]",
        "you are a helpful", "you are an AI",
    ]
    return any(m.lower() in text.lower() for m in leak_markers)
