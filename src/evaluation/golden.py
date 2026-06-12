"""Golden Dataset 管理 — 可回放的测试用例集合。

格式: JSONL，每行一个测试用例。
支持非确定性输出的统计断言和软失败阈值。

ref: Apollo ARS (Agent Regression Suite) — apollo.io, 2025
ref: Letta Evals — open-source stateful agent evaluation
ref: Monte Carlo "Soft Failure" — montecarlodata.com, 2025
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.config import get_project_root


@dataclass
class GoldenCase:
    """单个 Golden Test Case。"""

    id: str                                    # 唯一标识
    agent_id: str                              # Agent ID
    description: str                           # 测试描述
    input: dict[str, Any] = field(default_factory=dict)   # 输入参数
    expected: dict[str, Any] = field(default_factory=dict)  # 期望断言

    # 断言类型
    # - exact: 精确匹配
    # - contains: 包含子串
    # - semantic: LLM 语义判断
    # - norootcause: 不应包含某模式
    assertion_type: str = "semantic"

    # 软失败阈值 (0-1)
    # < threshold → hard fail
    # >= threshold → soft fail
    soft_fail_threshold: float = 0.5

    # 元数据
    category: str = "general"                  # happy_path / edge_cases / adversarial
    tags: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class GoldenDataset:
    """一组 Golden Test Cases。"""

    name: str
    description: str = ""
    cases: list[GoldenCase] = field(default_factory=list)
    version: str = "1.0"

    @property
    def total(self) -> int:
        return len(self.cases)

    def by_category(self, category: str) -> list[GoldenCase]:
        return [c for c in self.cases if c.category == category]

    def by_agent(self, agent_id: str) -> list[GoldenCase]:
        return [c for c in self.cases if c.agent_id == agent_id]

    def add(self, case: GoldenCase) -> None:
        self.cases.append(case)


class GoldenStore:
    """Golden Dataset 持久化存储。

    目录结构:
        tests/data/golden/
        ├── happy_path/
        │   ├── researcher.jsonl
        │   └── copywriter.jsonl
        ├── edge_cases/
        │   └── researcher.jsonl
        └── adversarial/
            └── researcher.jsonl
    """

    def __init__(self, base_dir: str = "") -> None:
        if base_dir:
            self._base = Path(base_dir)
        else:
            self._base = get_project_root() / "tests" / "data" / "golden"
        self._base.mkdir(parents=True, exist_ok=True)

    # ── 读取 ───────────────────────────────────────

    def load(self, name: str, category: str = "happy_path") -> GoldenDataset:
        """从 JSONL 文件加载 Golden Dataset。"""
        path = self._base / category / f"{name}.jsonl"
        cases: list[GoldenCase] = []

        if path.exists():
            for line in path.read_text().strip().split("\n"):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    cases.append(GoldenCase(
                        id=data.get("id", ""),
                        agent_id=data.get("agent_id", name),
                        description=data.get("description", ""),
                        input=data.get("input", {}),
                        expected=data.get("expected", {}),
                        assertion_type=data.get("assertion_type", "semantic"),
                        soft_fail_threshold=data.get("soft_fail_threshold", 0.5),
                        category=category,
                        tags=data.get("tags", []),
                        created_at=data.get("created_at", ""),
                    ))
                except (json.JSONDecodeError, KeyError):
                    pass

        return GoldenDataset(name=name, cases=cases, description=f"Golden dataset: {name}/{category}")

    def load_all(self, agent_id: str = "") -> list[GoldenCase]:
        """加载所有 Golden Cases，可选按 agent 过滤。"""
        all_cases: list[GoldenCase] = []
        for cat_dir in self._base.iterdir():
            if not cat_dir.is_dir():
                continue
            for fpath in cat_dir.glob("*.jsonl"):
                name = fpath.stem
                ds = self.load(name, cat_dir.name)
                if agent_id:
                    all_cases.extend(ds.by_agent(agent_id))
                else:
                    all_cases.extend(ds.cases)
        return all_cases

    # ── 写入 ───────────────────────────────────────

    def save(self, dataset: GoldenDataset, category: str = "happy_path") -> None:
        """保存 Golden Dataset 为 JSONL 文件。"""
        cat_dir = self._base / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        path = cat_dir / f"{dataset.name}.jsonl"

        lines: list[str] = []
        for case in dataset.cases:
            lines.append(json.dumps({
                "id": case.id,
                "agent_id": case.agent_id,
                "description": case.description,
                "input": case.input,
                "expected": case.expected,
                "assertion_type": case.assertion_type,
                "soft_fail_threshold": case.soft_fail_threshold,
                "tags": case.tags,
                "created_at": case.created_at,
            }, ensure_ascii=False))

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def add_case(self, name: str, case: GoldenCase, category: str = "happy_path") -> None:
        """追加一个测试用例到已有 dataset。"""
        ds = self.load(name, category)
        ds.add(case)
        self.save(ds, category)

    # ── 从生产日志导入 ──────────────────────────────

    def import_from_audit(self, audit_log_dir: str, agent_id: str = "", limit: int = 100) -> int:
        """从审计日志中导入失败案例作为新的 golden cases。"""
        audit_dir = Path(audit_log_dir)
        if not audit_dir.exists():
            return 0

        imported = 0
        for fpath in sorted(audit_dir.glob("audit_*.jsonl"), reverse=True)[:5]:
            for line in fpath.read_text().strip().split("\n"):
                if not line or imported >= limit:
                    break
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "error":
                        case = GoldenCase(
                            id=f"imported-{rec.get('trace_id', '')[:12]}",
                            agent_id=agent_id or rec.get("agent_id", "unknown"),
                            description=f"Imported from audit: {rec.get('error_message', '')[:100]}",
                            expected={"task_completed": False},
                            category="edge_cases",
                            tags=["imported", "production-failure"],
                        )
                        self.add_case("imported_failures", case, "edge_cases")
                        imported += 1
                except Exception:
                    pass
        return imported
