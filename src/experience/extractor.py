"""经验自动提取：任务完成/失败后触发，LLM 分析 → 生成经验条目。"""

from __future__ import annotations

import structlog

from src.shared.llm import get_llm
from src.shared.models import Experience, Rule, ScopeLevel, StepResult

logger = structlog.get_logger()


class ExperienceExtractor:
    """从任务日志和轨迹中提取可复用的经验。"""

    def __init__(self) -> None:
        self._llm = get_llm()

    async def extract(
        self,
        agent_id: str,
        task_type: str,
        task_success: bool,
        steps: list[StepResult],
        *,
        trace_id: str = "",
    ) -> list[Experience]:
        """分析任务执行轨迹，提取经验。"""
        prompt = self._build_prompt(agent_id, task_type, task_success, steps)
        try:
            raw = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                trace_id=f"{trace_id}_extract_exp",
            )
            return self._parse(raw, agent_id, task_type)
        except Exception as e:
            logger.warning("experience_extraction_failed", error=str(e), trace_id=trace_id)
            return []

    @staticmethod
    def _build_prompt(
        agent_id: str, task_type: str, success: bool, steps: list[StepResult]
    ) -> str:
        steps_text = "\n".join(
            f"- {s.step_id}: {s.status}" + (f" error={s.error}" if s.error else "")
            for s in steps
        )
        outcome = "成功" if success else "失败"
        return f"""从以下 Agent 任务执行轨迹中提取可复用的经验。

Agent: {agent_id}
任务类型: {task_type}
任务结果: {outcome}

执行步骤:
{steps_text}

请提取：
- 如果失败：trigger（触发条件）、symptom（现象）、root_cause（根因）、solution（解决方案）
- 如果成功：key decisions（关键决策）、successful strategies（成功策略）

输出 JSON 数组（最多 3 条）：
[{{"trigger": "条件", "symptom": "现象", "root_cause": "根因", "solution": "方案", "outcome": "结果", "confidence": 0.5-1.0, "tags": ["标签"]}}]"""

    @staticmethod
    def _parse(raw: str, agent_id: str, task_type: str) -> list[Experience]:
        import json as _json

        raw = raw.strip()
        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0]
                break
        try:
            items = _json.loads(raw)
        except _json.JSONDecodeError:
            return []

        experiences: list[Experience] = []
        for item in items:
            experiences.append(Experience(
                agent_id=agent_id,
                task_type=task_type,
                trigger=item.get("trigger", ""),
                symptom=item.get("symptom", ""),
                root_cause=item.get("root_cause", ""),
                solution=item.get("solution", ""),
                outcome=item.get("outcome", ""),
                confidence=float(item.get("confidence", 0.5)),
                validated=False,
            ))
        return experiences

    async def extract_rules(self, experiences: list[Experience],
                            agent_id: str = "",
                            scope: ScopeLevel = ScopeLevel.AGENT) -> list[Rule]:
        """从已验证经验中提炼 WHEN-THEN 规则。"""
        if not experiences:
            return []

        exp_text = "\n".join(
            f"- [{e.id[:8]}] trigger={e.trigger}, symptom={e.symptom}, "
            f"root_cause={e.root_cause}, solution={e.solution}"
            for e in experiences
        )

        prompt = f"""从以下经验中提取可复用的 WHEN-THEN 规则。

经验列表：
{exp_text}

提取要求：
- 每条规则格式：WHEN <条件> THEN <动作>
- 每条约经验最多产出 1 条规则
- confidence 根据经验的 confidence 设定（0.5-1.0）

输出 JSON 数组：
[{{"name": "规则名", "trigger_condition": "WHEN 条件", "action": "THEN 动作", "confidence": 0.5-1.0, "description": "说明"}}]"""

        try:
            raw = await self._llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                trace_id=f"{agent_id}_extract_rules",
            )
            return self._parse_rules(raw, experiences, agent_id, scope)
        except Exception as e:
            logger.warning("rule_extraction_failed", error=str(e))
            return []

    @staticmethod
    def _parse_rules(raw: str, experiences: list[Experience],
                     agent_id: str, scope: ScopeLevel) -> list[Rule]:
        import json as _json

        raw = raw.strip()
        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0]
                break
        try:
            items = _json.loads(raw)
        except _json.JSONDecodeError:
            return []

        # 取经验 confidence 最小值作为规则初始 confidence
        avg_confidence = sum(e.confidence for e in experiences) / max(len(experiences), 1)

        rules: list[Rule] = []
        for i, item in enumerate(items):
            source_id = experiences[min(i, len(experiences) - 1)].id if experiences else None
            rules.append(Rule(
                name=item.get("name", f"Rule {i+1}"),
                description=item.get("description", ""),
                trigger_condition=item.get("trigger_condition", ""),
                action=item.get("action", ""),
                scope=scope,
                agent_id=agent_id,
                confidence=float(item.get("confidence", avg_confidence * 0.8)),
                source_experience_id=source_id,
            ))
        return rules
