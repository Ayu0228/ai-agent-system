"""Token 成本追踪。每 Agent 每日预算 50K tokens。"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.shared.config import get_settings


class CostTracker:
    """按 Agent 维度追踪 token 消耗。超出预算自动告警。"""

    def __init__(self) -> None:
        self._usage: dict[str, dict[str, int | float]] = {}  # agent_id → {tokens, last_reset}
        self._budget = get_settings().daily_token_budget_per_agent

    def check_budget(self, agent_id: str) -> dict:
        """返回 {tokens_used, tokens_remaining, is_exceeded}。"""
        self._maybe_reset(agent_id)
        used = int(self._usage.get(agent_id, {}).get("tokens", 0))
        return {
            "tokens_used": used,
            "tokens_remaining": max(0, self._budget - used),
            "is_exceeded": used >= self._budget,
            "budget": self._budget,
        }

    def record(self, agent_id: str, tokens: int) -> None:
        """记录 token 消耗。"""
        self._maybe_reset(agent_id)
        if agent_id not in self._usage:
            self._usage[agent_id] = {"tokens": 0, "last_reset": time.time()}
        self._usage[agent_id]["tokens"] = int(self._usage[agent_id]["tokens"]) + tokens

    def _maybe_reset(self, agent_id: str) -> None:
        entry = self._usage.get(agent_id)
        if entry and time.time() - float(entry["last_reset"]) > 86400:
            entry["tokens"] = 0
            entry["last_reset"] = time.time()
