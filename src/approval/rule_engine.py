"""ASK FIRST 规则引擎。从 AGENTS.md 提取规则，程序化判断是否需要审批。"""

from __future__ import annotations

import re
from pathlib import Path

import structlog

logger = structlog.get_logger()


class RuleEngine:
    """基于规则的审批决策。不依赖 LLM，确定性逻辑。"""

    # 红线规则——完全禁止的操作
    RED_LINES = [
        (r"rm\s+-rf", "rm -rf 是禁止操作"),
        (r"DROP\s+TABLE", "DROP TABLE 是禁止操作"),
        (r"modify.*(?:system.?prompt|AGENTS\.md)", "修改系统 Prompt 是禁止操作"),
        (r"elevate.*(?:privilege|permission|autonomy)", "提权操作是禁止操作"),
        (r"impersonate.*agent", "伪造 Agent 身份是禁止操作"),
        (r"delete.*(?:memory|experience).*(?:other|别的|其他)", "删除其他 Agent 的记忆是禁止操作"),
    ]

    def check_action(self, action: str, params: dict | None = None) -> tuple[bool, str]:
        """检查操作是否命中红线规则。返回 (blocked, reason)。"""
        params_str = str(params or {})
        combined = f"{action} {params_str}"

        for pattern, reason in self.RED_LINES:
            if re.search(pattern, combined, re.IGNORECASE):
                logger.warning("red_line_triggered", action=action, pattern=pattern)
                return (True, reason)

        return (False, "")

    @staticmethod
    def load_agent_rules(agent_id: str) -> list[str]:
        """从 config/agents/{agent_id}.yaml 加载 Agent 的 ASK FIRST 规则。"""
        project_root = Path(__file__).resolve().parents[2]
        rules_path = project_root / "config" / "agents" / f"{agent_id}.yaml"
        if not rules_path.exists():
            return []
        import yaml
        try:
            with open(rules_path) as f:
                config = yaml.safe_load(f)
                return config.get("ask_first_rules", []) if config else []
        except Exception:
            return []
