"""审计日志 — JSONL 格式持久化，支持轮转和回放。

每行一条审计记录，包含完整的 trace/span 上下文。

ref: Anthropic "Building Effective AI Agents" (2025) — audit trail importance
ref: Google SRE Book — monitoring distributed systems
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.observability.tracer import Span, Trace
from src.shared.config import get_settings

logger = structlog.get_logger()


class AuditLogger:
    """JSONL 审计日志记录器。

    用法:
        audit = get_audit_logger()
        tracer.on_span_complete(audit.on_span)
    """

    def __init__(self, log_dir: str = "") -> None:
        settings = get_settings()
        self._log_dir = Path(log_dir or settings.chromadb_path.replace("chromadb", "audit_logs"))
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: str = ""
        self._line_count: int = 0
        self._max_lines_per_file: int = 10_000
        self._rotate()

    # ── 核心写入 ───────────────────────────────────

    def on_span(self, span: Span) -> None:
        """span 完成时写入审计日志。"""
        self._write(span.to_dict())

    def on_trace(self, trace: Trace) -> None:
        """trace 完成时写入汇总记录。"""
        self._write({
            "type": "trace_summary",
            "trace_id": trace.trace_id,
            "session_id": trace.session_id,
            "span_count": len(trace.spans),
            "total_tokens": trace.total_tokens,
            "total_cost": round(trace.total_cost, 6),
            "duration_ms": round(trace.duration_ms, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log_event(
        self,
        event_type: str,
        *,
        agent_id: str = "",
        trace_id: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        """写入自定义审计事件。"""
        record: dict[str, Any] = {
            "type": event_type,
            "agent_id": agent_id,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if data:
            record["data"] = data
        self._write(record)

    # ── 内部 ───────────────────────────────────────

    def _write(self, record: dict[str, Any]) -> None:
        try:
            with open(self._current_file, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._line_count += 1
            if self._line_count >= self._max_lines_per_file:
                self._rotate()
        except Exception:
            logger.exception("audit_write_failed")

    def _rotate(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_file = str(self._log_dir / f"audit_{ts}.jsonl")
        self._line_count = 0
        logger.info("audit_log_rotated", file=self._current_file)

    # ── 查询 ───────────────────────────────────────

    def get_recent_logs(self, limit: int = 100) -> list[dict[str, Any]]:
        """读取最近的审计记录。"""
        files = sorted(self._log_dir.glob("audit_*.jsonl"), reverse=True)
        records: list[dict[str, Any]] = []
        for fpath in files:
            if len(records) >= limit:
                break
            try:
                lines = fpath.read_text().strip().split("\n")
                for line in reversed(lines):
                    if not line:
                        continue
                    records.append(json.loads(line))
                    if len(records) >= limit:
                        break
            except Exception:
                pass
        return records

    def search_logs(
        self,
        *,
        event_type: str = "",
        agent_id: str = "",
        trace_id: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """按条件搜索审计记录。"""
        files = sorted(self._log_dir.glob("audit_*.jsonl"), reverse=True)
        results: list[dict[str, Any]] = []
        for fpath in files:
            if len(results) >= limit:
                break
            try:
                for line in reversed(fpath.read_text().strip().split("\n")):
                    if not line:
                        continue
                    rec = json.loads(line)
                    if event_type and rec.get("type") != event_type:
                        continue
                    if agent_id and rec.get("agent_id") != agent_id:
                        continue
                    if trace_id and rec.get("trace_id") != trace_id:
                        continue
                    results.append(rec)
                    if len(results) >= limit:
                        break
            except Exception:
                pass
        return results

    def replay_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """回放某个 Trace 的所有 span（按时间排序）。"""
        records = self.search_logs(trace_id=trace_id, limit=1000)
        # 过滤出 span 记录
        spans = [r for r in records if "span_id" in r]
        spans.sort(key=lambda r: r.get("start_time", ""))
        return spans


_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
