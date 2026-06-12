"""工作流步骤依赖解析。"""

from __future__ import annotations

import re

from src.shared.errors import CircularDependencyError
from src.shared.models import StepConfig


class DependencyResolver:
    """解析步骤间的依赖关系，检测循环依赖，提供拓扑排序。"""

    @staticmethod
    def resolve_inputs(step: StepConfig, context: dict) -> dict:
        """解析步骤输入中的 $变量引用。"""
        resolved: dict = {}
        for key, value in step.input.items():
            if isinstance(value, str) and value.startswith("$"):
                ref_path = value[1:].split(".")
                resolved[key] = DependencyResolver._navigate(context, ref_path)
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _navigate(data: dict, path: list[str]) -> object:
        current: object = data
        for part in path:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def topological_sort(steps: list[StepConfig]) -> list[StepConfig]:
        """拓扑排序，检测循环依赖。"""
        step_map = {s.id: s for s in steps}
        in_degree: dict[str, int] = {s.id: 0 for s in steps}
        adj: dict[str, list[str]] = {s.id: [] for s in steps}

        for s in steps:
            for dep in s.depends_on:
                if dep in step_map:
                    adj[dep].append(s.id)
                    in_degree[s.id] += 1

        # Kahn's algorithm
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        sorted_ids: list[str] = []

        while queue:
            node = queue.pop(0)
            sorted_ids.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_ids) != len(steps):
            raise CircularDependencyError("检测到循环依赖，无法执行工作流")

        return [step_map[sid] for sid in sorted_ids]
