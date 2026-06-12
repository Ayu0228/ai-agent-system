"""经验人工确认流程。新经验需要阿禹确认后才能写入记忆库。"""

from __future__ import annotations

import structlog

from src.memory.gateway import MemoryGateway
from src.shared.models import Experience, MemoryEntry, MemoryType, Rule

logger = structlog.get_logger()


class ExperienceValidator:
    """管理经验的人工确认流程。"""

    def __init__(self, gateway: MemoryGateway) -> None:
        self._gateway = gateway

    async def validate_and_store(
        self, experience: Experience, *, approved: bool, trace_id: str = "",
        promote_to_rule: bool = False,
    ) -> tuple[str | None, list[Rule]]:
        """如果确认通过，写入记忆库。promote_to_rule=True 时同时提炼规则。"""
        if not approved:
            logger.info("experience_rejected", id=experience.id, trace_id=trace_id)
            return None, []

        entry = MemoryEntry(
            agent_id=experience.agent_id,
            content=f"[{experience.task_type}] 触发: {experience.trigger} | "
            f"根因: {experience.root_cause} | 方案: {experience.solution} | "
            f"结果: {experience.outcome}",
            memory_type=MemoryType.EXPERIENCE,
            tags=[
                f"type:{experience.task_type}",
                "validated",
                *([f"trigger:{experience.trigger}"] if experience.trigger else []),
            ],
            importance=experience.confidence,
            source_trace_id=trace_id,
        )
        entry_id = await self._gateway.write(entry, trace_id=trace_id)

        rules: list[Rule] = []
        if promote_to_rule:
            from src.experience.extractor import ExperienceExtractor
            extractor = ExperienceExtractor()
            rules = await extractor.extract_rules([experience], agent_id=experience.agent_id)

        logger.info("experience_validated", id=entry_id, trace_id=trace_id,
                    rules_extracted=len(rules))
        return entry_id, rules
