"""共享 pytest fixtures。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """临时目录，测试结束自动清理。"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
async def mock_llm():
    """Mock LLM 客户端。"""
    with patch("src.shared.llm.AsyncOpenAI") as mock_openai:
        client = AsyncMock()
        mock_openai.return_value = client

        # 默认 mock 响应
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="mock response"))
        ]
        mock_response.usage = AsyncMock(total_tokens=10)
        client.chat.completions.create.return_value = mock_response

        yield client


@pytest.fixture
def sample_memory_entry():
    """示例记忆条目。"""
    from src.shared.models import MemoryEntry, MemoryType

    return MemoryEntry(
        agent_id="researcher",
        content="youtube-dl download timeout increased to 600s for long videos",
        memory_type=MemoryType.EXPERIENCE,
        tags=["youtube", "timeout", "transcription"],
        importance=0.8,
    )


@pytest.fixture
def sample_workflow_definition():
    """示例工作流定义。"""
    from src.shared.models import StepConfig, StepType, WorkflowDefinition

    return WorkflowDefinition(
        name="test-workflow",
        version="1.0",
        steps=[
            StepConfig(
                id="search",
                agent="researcher",
                type=StepType.TASK,
                config={"prompt": "search for {{topic}}", "timeout": 60},
            ),
            StepConfig(
                id="analyze",
                agent="data-analyst",
                type=StepType.TASK,
                depends_on=["search"],
                input={"data": "$search.output"},
            ),
        ],
    )
