"""Prompt 模板加载器。从 YAML 文件加载可复用的 prompt 模板。"""

from __future__ import annotations

from pathlib import Path


class PromptLoader:
    """加载和管理 Agent 的 prompt 模板。"""

    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        from src.shared.config import get_project_root

        self._dir = Path(prompts_dir) if prompts_dir else get_project_root() / "config" / "prompts"
        self._cache: dict[str, dict] = {}

    def load(self, agent_id: str) -> dict:
        """加载指定 Agent 的所有 prompt 模板。"""
        if agent_id in self._cache:
            return self._cache[agent_id]

        yaml_path = self._dir / f"{agent_id}.yaml"
        if not yaml_path.exists():
            return {}

        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
            templates = data.get("prompts", {}) if data else {}
            self._cache[agent_id] = templates
            return templates

    def get(self, agent_id: str, prompt_name: str) -> str:
        """获取指定 Agent 的指定 prompt 模板。"""
        templates = self.load(agent_id)
        prompt_def = templates.get(prompt_name, {})
        return prompt_def.get("template", "") if isinstance(prompt_def, dict) else ""

    def render(self, agent_id: str, prompt_name: str, **variables) -> str:
        """加载模板并替换变量。"""
        template = self.get(agent_id, prompt_name)
        if not template:
            return ""
        result = template
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def reload(self) -> None:
        """清空缓存，重新加载。"""
        self._cache.clear()
