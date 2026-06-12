"""JSONL 审计日志。每条操作一行，不可被 Agent 修改。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.shared.config import get_settings


class AuditLogger:
    """JSONL 格式审计日志。append-only。"""

    def __init__(self) -> None:
        settings = get_settings()
        log_path = settings.resolve_path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = Path(log_path).with_suffix(".audit.jsonl")

    def log(
        self,
        *,
        agent_id: str,
        event: str,
        tool: str | None = None,
        params_hash: str | None = None,
        result_code: str | None = None,
        tokens: int = 0,
        duration_ms: int = 0,
        trace_id: str = "",
        risk_level: str = "L0",
    ) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "event": event,
            "tool": tool,
            "params_hash": params_hash,
            "result_code": result_code,
            "tokens": tokens,
            "duration_ms": duration_ms,
            "trace_id": trace_id,
            "risk_level": risk_level,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def tail(self, n: int = 100) -> list[dict]:
        """读取最近 n 条日志。"""
        if not self._path.exists():
            return []
        lines = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines[-n:]
