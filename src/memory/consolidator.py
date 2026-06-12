"""记忆整合器：会话结束 / 定时触发，将工作记忆提升到长期记忆。"""

from __future__ import annotations

import json as _json

import structlog

from src.memory.gateway import MemoryGateway
from src.shared.llm import get_llm
from src.shared.models import MemoryEntry, MemoryType

logger = structlog.get_logger()


class MemoryConsolidator:
    """用 LLM 从工作记忆中提取关键信息，写入长期记忆。"""

    def __init__(self, gateway: MemoryGateway) -> None:
        self._gateway = gateway
        self._llm = get_llm()

    async def consolidate_session(
        self,
        agent_id: str,
        session_id: str,
        working_memories: list[MemoryEntry],
        *,
        trace_id: str = "",
    ) -> list[MemoryEntry]:
        """整合一次会话的工作记忆。"""
        if not working_memories:
            return []

        # 用 LLM 提取关键信息
        prompt = self._build_extraction_prompt(working_memories)
        try:
            raw = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                trace_id=f"{trace_id}_consolidate",
            )
            items = self._parse_extraction(raw)
        except Exception as e:
            logger.warning("consolidation_llm_failed", error=str(e))
            return []

        consolidated: list[MemoryEntry] = []
        for item in items:
            entry = MemoryEntry(
                agent_id=agent_id,
                content=item.get("content", ""),
                memory_type=MemoryType(item.get("category", "fact")),
                tags=item.get("tags", []),
                importance=float(item.get("importance", 0.5)),
                source_trace_id=trace_id,
            )
            await self._gateway.write(entry, trace_id=trace_id)
            consolidated.append(entry)

        logger.info("session_consolidated", count=len(consolidated), session_id=session_id)
        return consolidated

    @staticmethod
    def _build_extraction_prompt(memories: list[MemoryEntry]) -> str:
        items = "\n".join(f"- [{e.created_at}] {e.content}" for e in memories)
        return f"""分析以下会话记忆，提取值得长期保留的信息。

会话记忆：
{items}

提取类别：factual_knowledge / decision_experience / user_preference / error_lesson / success_strategy / procedural_rule

如果有可复用的 if-then 规则模式，请用 procedural_rule 类别输出，content 格式为 "WHEN <条件> THEN <动作>"。

输出 JSON 数组（最多 5 条）：
[{{"category": "类别", "content": "具体内容", "importance": 0.5-1.0, "tags": ["标签"]}}]"""

    @staticmethod
    def _parse_extraction(raw: str) -> list[dict]:
        raw = raw.strip()
        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0]
                break
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            return []
