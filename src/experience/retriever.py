"""经验检索与注入。Agent 启动时自动检索相关经验并注入到上下文。"""

from __future__ import annotations

from src.memory.gateway import MemoryGateway
from src.shared.models import Experience, MemoryQuery, MemoryType


class ExperienceRetriever:
    """从记忆库中检索与当前任务相关的经验。"""

    def __init__(self, gateway: MemoryGateway) -> None:
        self._gateway = gateway

    async def retrieve(
        self,
        agent_id: str,
        task_type: str,
        *,
        top_k: int = 3,
        trace_id: str = "",
    ) -> list[Experience]:
        """检索相关经验。返回已验证的优先。"""
        results = await self._gateway.search(
            MemoryQuery(
                query_text=task_type,
                agent_id=agent_id,
                memory_types=[MemoryType.EXPERIENCE],
                top_k=top_k * 2,
            ),
            trace_id=trace_id,
        )

        experiences: list[Experience] = []
        for r in results:
            entry = r.entry
            # 从记忆条目构建经验对象
            tags = entry.tags
            exp = Experience(
                id=entry.id,
                agent_id=entry.agent_id,
                task_type=next((t for t in tags if t.startswith("type:")), task_type),
                trigger="",
                symptom="",
                root_cause="",
                solution=entry.content,
                outcome="",
                confidence=entry.importance,
                validated=entry.memory_type == MemoryType.EXPERIENCE,
                last_applied_at=entry.last_accessed_at,
                apply_count=entry.access_count,
            )
            experiences.append(exp)

        # 已验证优先 → confidence 降序 → top_k
        experiences.sort(key=lambda e: (e.validated, e.confidence), reverse=True)
        return experiences[:top_k]

    @staticmethod
    def format_context_block(experiences: list[Experience]) -> str:
        """将经验格式化为注入上下文的文本块。"""
        if not experiences:
            return ""

        lines = ["## 相关历史经验（自动注入）"]
        for i, exp in enumerate(experiences, 1):
            lines.append(
                f"{i}. [{exp.task_type}] {exp.solution}"
                f"（置信度: {exp.confidence:.0%}"
                f"{' ✓已验证' if exp.validated else ''}"
                f", 引用: {exp.apply_count}次）"
            )
        return "\n".join(lines) + "\n"
