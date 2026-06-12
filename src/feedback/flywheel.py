"""Data Flywheel — 生产失败 → 标注队列 → 回归测试集的闭环。

ref: LangChain data flywheel pattern
    "Production monitoring creates a data flywheel:
     failures → annotation queue → human review → regression test set"

阶段流转:
  COLLECT → ANNOTATE → REVIEW → PROMOTE → MONITOR → (循环)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class FlywheelStage(str, Enum):
    COLLECT = "collect"         # 生产环境捕获到失败
    ANNOTATE = "annotate"       # 等待人工标注
    REVIEW = "review"           # 人工审核中
    PROMOTE = "promote"         # 转为回归测试用例
    MONITOR = "monitor"         # 监控回归结果


@dataclass
class FlywheelEntry:
    """数据飞轮条目 — 一个从生产失败转化来的测试候选。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    stage: FlywheelStage = FlywheelStage.COLLECT
    agent_id: str = ""
    workflow_id: str = ""
    input_message: str = ""
    actual_output: str = ""
    expected_output: str = ""                # 由人工标注
    error_type: str = ""                     # hallucination / tool_error / wrong_answer / timeout
    trace_id: str = ""
    severity: str = "medium"
    created_at: float = field(default_factory=time.time)
    annotated_at: float = 0.0
    reviewed_at: float = 0.0
    promoted_at: float = 0.0
    annotator: str = ""                      # 标注人
    reviewer: str = ""
    golden_case_id: str = ""                 # 提升后的 golden case id
    promote_action: str = "golden_case"      # golden_case / rule / both
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_golden_case(self) -> dict[str, Any]:
        """转换为 golden test case 格式。"""
        return {
            "id": f"FW-{self.id}",
            "agent_id": self.agent_id,
            "description": f"飞轮导入: {self.error_type} — {self.notes[:80] if self.notes else 'N/A'}",
            "input": {"message": self.input_message},
            "expected": {
                "task_completed": True,
                "contains_keywords": self._extract_keywords(self.expected_output),
                "min_length": 10,
            },
            "assertion_type": "semantic",
            "soft_fail_threshold": 0.5,
            "tags": ["flywheel", self.error_type],
        }

    @staticmethod
    def _extract_keywords(text: str, max_kw: int = 5) -> list[str]:
        """简单关键词提取。"""
        words = [w.strip("，。！？,.!?") for w in text.split()
                 if len(w.strip("，。！？,.!?")) >= 2]
        return words[:max_kw] if words else [text[:20]]


class DataFlywheel:
    """数据飞轮管理器。

    用法:
        fw = DataFlywheel(storage_dir="./data/flywheel")

        # 从生产失败收集
        entry = fw.collect_failure(
            agent_id="researcher",
            input_message="什么是转化率？",
            actual_output="转化率是...",
            error_type="hallucination",
        )

        # 人工标注期望输出
        fw.annotate(entry.id, expected_output="正确的答案...", annotator="reviewer-1")

        # 审核通过 → 提升为回归测试
        fw.promote(entry.id, reviewer="lead-1")

        # 导出回归测试用例
        cases = fw.export_golden_cases()
    """

    def __init__(self, storage_dir: str = "") -> None:
        self._entries: dict[str, FlywheelEntry] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else None

        if self._storage_dir:
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ── 收集 ───────────────────────────────────────

    def collect_failure(self, agent_id: str, input_message: str,
                        actual_output: str, error_type: str,
                        trace_id: str = "", severity: str = "medium",
                        **kwargs: Any) -> FlywheelEntry:
        """从生产环境收集失败案例。"""
        entry = FlywheelEntry(
            stage=FlywheelStage.COLLECT,
            agent_id=agent_id,
            input_message=input_message,
            actual_output=actual_output,
            error_type=error_type,
            trace_id=trace_id,
            severity=severity,
            metadata=kwargs,
        )
        self._entries[entry.id] = entry
        logger.info("flywheel_collected", id=entry.id, agent=agent_id,
                    error=error_type)
        return entry

    # ── 标注 ───────────────────────────────────────

    def annotate(self, entry_id: str, expected_output: str,
                 annotator: str = "") -> bool:
        """人工标注期望输出。"""
        entry = self._entries.get(entry_id)
        if not entry:
            return False
        if entry.stage not in (FlywheelStage.COLLECT, FlywheelStage.ANNOTATE):
            return False
        entry.expected_output = expected_output
        entry.annotator = annotator
        entry.annotated_at = time.time()
        entry.stage = FlywheelStage.REVIEW
        logger.info("flywheel_annotated", id=entry_id, annotator=annotator)
        return True

    # ── 审核与提升 ─────────────────────────────────

    def promote(self, entry_id: str, reviewer: str = "",
                notes: str = "", promote_action: str = "golden_case") -> FlywheelEntry | None:
        """审核通过 → 提升为回归测试候选 + 可选规则。"""
        entry = self._entries.get(entry_id)
        if not entry:
            return None
        if entry.stage != FlywheelStage.REVIEW:
            return None
        if not entry.expected_output:
            return None

        entry.reviewer = reviewer
        entry.reviewed_at = time.time()
        entry.notes = notes or entry.notes
        entry.stage = FlywheelStage.PROMOTE
        entry.promoted_at = time.time()
        entry.promote_action = promote_action

        if promote_action in ("golden_case", "both"):
            entry.golden_case_id = f"FW-{entry.id}"

        # 持久化
        self._save_entry(entry)

        logger.info("flywheel_promoted", id=entry_id, agent=entry.agent_id,
                    action=promote_action)
        return entry

    def extract_rules_from_promoted(self, entry_id: str) -> list:
        """从已提升的飞轮条目中提取规则。返回 Rule 列表。"""
        entry = self._entries.get(entry_id)
        if not entry or entry.promote_action not in ("rule", "both"):
            return []

        from src.shared.models import Rule
        rule = Rule(
            name=f"FW-Rule-{entry.id[:8]}",
            description=f"Flywheel rule from {entry.error_type}: {entry.notes[:100]}",
            trigger_condition=f"WHEN agent={entry.agent_id} produces {entry.error_type}",
            action=f"THEN correct: {entry.expected_output[:200]}",
            agent_id=entry.agent_id,
            confidence=0.6,
            tags=["flywheel", entry.error_type],
        )
        return [rule]

    def reject(self, entry_id: str, reason: str = "") -> bool:
        """审核驳回 — 标记为不适合提升。"""
        entry = self._entries.get(entry_id)
        if not entry:
            return False
        entry.stage = FlywheelStage.COLLECT  # 回到收集阶段
        entry.notes = f"REJECTED: {reason}"
        logger.info("flywheel_rejected", id=entry_id, reason=reason)
        return True

    # ── 导出 ───────────────────────────────────────

    def export_golden_cases(self) -> list[dict[str, Any]]:
        """导出所有已提升的回归测试用例。"""
        cases = []
        for entry in self._entries.values():
            if entry.stage == FlywheelStage.PROMOTE and entry.golden_case_id:
                cases.append(entry.to_golden_case())
        return cases

    def export_jsonl(self, filepath: str) -> int:
        """导出到 JSONL 文件。"""
        cases = self.export_golden_cases()
        if not cases:
            return 0

        with open(filepath, "a") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

        logger.info("flywheel_exported", path=filepath, count=len(cases))
        return len(cases)

    # ── 查询 ───────────────────────────────────────

    def get_by_stage(self, stage: FlywheelStage) -> list[FlywheelEntry]:
        return [e for e in self._entries.values() if e.stage == stage]

    def get_pending_review(self) -> list[FlywheelEntry]:
        return self.get_by_stage(FlywheelStage.REVIEW)

    def get_stats(self) -> dict[str, Any]:
        stages = {}
        for stage in FlywheelStage:
            stages[stage.value] = sum(
                1 for e in self._entries.values() if e.stage == stage
            )
        return {
            "total": len(self._entries),
            "by_stage": stages,
            "promoted": stages.get("promote", 0),
            "pending_review": stages.get("review", 0),
        }

    # ── 持久化 ─────────────────────────────────────

    def _save_entry(self, entry: FlywheelEntry) -> None:
        if not self._storage_dir:
            return
        fpath = self._storage_dir / f"{entry.id}.json"
        data = {
            "id": entry.id,
            "stage": entry.stage.value,
            "agent_id": entry.agent_id,
            "input_message": entry.input_message,
            "actual_output": entry.actual_output,
            "expected_output": entry.expected_output,
            "error_type": entry.error_type,
            "severity": entry.severity,
            "golden_case_id": entry.golden_case_id,
            "notes": entry.notes,
            "created_at": entry.created_at,
            "promoted_at": entry.promoted_at,
        }
        fpath.write_text(json.dumps(data, ensure_ascii=False, indent=2))
