"""Agent 路由器：根据任务类型和内容语义路由到最合适的 Agent。"""

from __future__ import annotations

from src.shared.constants import AGENT_IDS

# 任务类型 → 首选 Agent 映射
TASK_ROUTING: dict[str, str] = {
    "search": "researcher",
    "research": "researcher",
    "collect": "researcher",
    "code": "tech-dev",
    "develop": "tech-dev",
    "debug": "tech-dev",
    "write": "copywriter",
    "copy": "copywriter",
    "script": "script-editor",
    "storyboard": "script-editor",
    "analyze": "data-analyst",
    "data": "data-analyst",
    "chart": "data-analyst",
    "design": "visual-designer",
    "visual": "visual-designer",
    "image": "visual-designer",
    "product": "product-designer",
    "prd": "product-designer",
    "ux": "product-designer",
    "monitor": "ops-monitor",
    "ops": "ops-monitor",
    "alert": "ops-monitor",
    "invest": "investment-analyst",
    "finance": "investment-analyst",
    "strategy": "content-strategist",
    "content_plan": "content-strategist",
    "orchestrate": "main",
    "coordinate": "main",
}

# Agent 能力标签 → Agent ID
CAPABILITY_ROUTING: dict[str, str] = {
    "web_search": "researcher",
    "web_fetch": "researcher",
    "code_generation": "tech-dev",
    "architecture_design": "tech-dev",
    "content_writing": "copywriter",
    "script_writing": "script-editor",
    "data_analysis": "data-analyst",
    "image_generation": "visual-designer",
    "product_planning": "product-designer",
    "system_monitoring": "ops-monitor",
    "industry_research": "investment-analyst",
    "content_strategy": "content-strategist",
}


class AgentRouter:
    """根据任务描述路由到最合适的 Agent。"""

    task_routing = TASK_ROUTING
    capability_routing = CAPABILITY_ROUTING

    def route(self, task_description: str, preferred_agent: str | None = None) -> str:
        """返回应该处理此任务的 Agent ID。"""
        if preferred_agent and preferred_agent in AGENT_IDS:
            return preferred_agent

        text_lower = task_description.lower()

        # 匹配任务类型关键词
        for keyword, agent_id in self.task_routing.items():
            if keyword in text_lower:
                return agent_id

        # 默认：让 main 处理和分发
        return "main"

    def get_capable_agents(self, capability: str) -> list[str]:
        """返回具有指定能力的所有 Agent。"""
        return [
            agent_id
            for cap, agent_id in self.capability_routing.items()
            if cap == capability
        ]

    def list_agents(self) -> list[dict]:
        """列出所有 Agent 及其能力。"""
        import yaml
        from pathlib import Path

        agents = []
        project_root = Path(__file__).resolve().parents[2]
        agents_dir = project_root / "config" / "agents"
        for f in sorted(agents_dir.glob("*.yaml")):
            with open(f) as fh:
                data = yaml.safe_load(fh)
                agent_data = data.get("agent", {})
                agents.append({
                    "id": agent_data.get("id", ""),
                    "name": agent_data.get("name", ""),
                    "model": agent_data.get("model", ""),
                    "capabilities": data.get("capabilities", []),
                })
        return agents
