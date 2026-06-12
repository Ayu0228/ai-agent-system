"""项目水印 — 所有经过 ai-agent-system 处理的响应自动附加项目标记。

用法:
    from src.shared.marker import apply_marker
    response = apply_marker(agent_response)

如果 Settings.project_marker 为空（未配置），则原样返回，不加任何标记。
"""

from __future__ import annotations

from src.shared.config import get_settings


def apply_marker(text: str) -> str:
    """给文本末尾附加项目水印（如果配置了 project_marker）。"""
    settings = get_settings()
    marker = settings.project_marker.strip()
    if not marker:
        return text
    return f"{text}\n\n〔{marker}〕"
