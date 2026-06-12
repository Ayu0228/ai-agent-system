"""自进化双轨机制 —— self-improving(P1) + AutoScale(P2)。

对齐文档规格：
- self-improving: 失败/纠正/反馈 → Learnings.md 错题本 → AGENTS.md 或独立 skill
- AutoScale: 复用≥3 次且成功率≥90% → Skill.md 封装
- 生命周期闭环：执行 → 提取 → 失败→self-improving / 成功→AutoScale → memify
- 先开 self-improving 跑 1-2 周积累，再挑高频模式让 AutoScale 封装
"""

from __future__ import annotations

import fcntl
import json as _json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from src.shared.models import EvolutionRecord, ExperienceQuery, SelfImprovingTrack

logger = structlog.get_logger()

# 学习记录文件路径
LEARNINGS_PATH = Path.home() / "self-improving" / "memory.md"
CORRECTIONS_PATH = Path.home() / "self-improving" / "corrections.md"
SKILLS_DIR = Path.home() / "self-improving" / "skills"


class SelfEvolution:
    """自进化双轨管理器。"""

    def __init__(self, experience_memory) -> None:
        self._experience = experience_memory  # ExperienceMemory instance
        self._ensure_dirs()

    @staticmethod
    def _ensure_dirs() -> None:
        LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Self-Improving Track (P1) ─────────────────────────

    def capture_failure(
        self, agent_id: str, scenario: str, error: str, fix: str = ""
    ) -> EvolutionRecord:
        """捕捉失败/错误/纠正 → 写入 Learnings.md 错题本。"""
        now = datetime.now(timezone.utc).isoformat()
        entry = (
            f"## {now[:10]} | {agent_id}\n"
            f"- 场景：{scenario}\n"
            f"- 错误：{error}\n"
            + (f"- 修复：{fix}\n" if fix else "")
            + "\n"
        )
        self._append_learnings(entry)

        record = EvolutionRecord(
            track=SelfImprovingTrack.SELF_IMPROVING,
            content=entry.strip(),
            trigger_reason=f"failure: {error[:100]}",
            created_at=now,
        )
        logger.info("failure_captured", agent_id=agent_id, error=error[:80])
        return record

    def capture_correction(
        self, agent_id: str, wrong: str, correct: str, reason: str = ""
    ) -> EvolutionRecord:
        """捕捉纠正/反馈。"""
        now = datetime.now(timezone.utc).isoformat()
        entry = (
            f"## 纠正 | {now[:10]} | {agent_id}\n"
            f"- 错误理解：{wrong}\n"
            f"- 正确理解：{correct}\n"
            + (f"- 原因：{reason}\n" if reason else "")
            + "\n"
        )
        self._append_corrections(entry)

        record = EvolutionRecord(
            track=SelfImprovingTrack.SELF_IMPROVING,
            content=entry.strip(),
            trigger_reason=f"correction: {wrong[:100]} → {correct[:100]}",
            created_at=now,
        )
        logger.info("correction_captured", agent_id=agent_id)
        return record

    def capture_best_practice(self, agent_id: str, practice: str, context: str = "") -> EvolutionRecord:
        """捕捉最佳实践。"""
        now = datetime.now(timezone.utc).isoformat()
        entry = (
            f"## 最佳实践 | {now[:10]} | {agent_id}\n"
            f"- 做法：{practice}\n"
            + (f"- 上下文：{context}\n" if context else "")
            + "\n"
        )
        self._append_learnings(entry)

        record = EvolutionRecord(
            track=SelfImprovingTrack.SELF_IMPROVING,
            content=entry.strip(),
            trigger_reason=f"best_practice: {practice[:100]}",
            created_at=now,
        )
        return record

    def promote_to_agents_md(self, learning: str) -> str:
        """将 Learnings.md 中的经验 Promote 到 AGENTS.md 或独立 skill。

        返回建议的 skill 文件路径（如有）。
        """
        # 简化的 promote 逻辑：判断是否需要提升
        if "skill:" in learning.lower() or "技能" in learning:
            skill_name = learning.split("skill:")[1].split("\n")[0].strip() if "skill:" in learning.lower() else "auto_skill"
            skill_path = SKILLS_DIR / f"{skill_name}.md"
            skill_path.write_text(
                f"# {skill_name}\n\n"
                f"自动学习技能（来自 self-improving）\n\n"
                f"## 背景\n{learning}\n\n"
                f"## 创建时间\n{datetime.now(timezone.utc).isoformat()}\n",
                encoding="utf-8",
            )
            return str(skill_path)
        return ""

    # ── AutoScale Track (P2) ──────────────────────────────

    def check_autoscale_eligible(self, agent_id: str) -> list[dict[str, Any]]:
        """检查哪些经验满足 AutoScale 封装条件。

        条件：复用≥3 次 且 成功率≥90%。
        """
        eligible = self._experience._db.execute(
            """SELECT experience_id, scenario, approach, lesson, usage_count, success_rate
               FROM experiences
               WHERE owner_agent_id = ? AND autoscale_eligible = 1
               AND usage_count >= 3 AND success_rate >= 0.9""",
            (agent_id,),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in eligible:
            results.append({
                "experience_id": row["experience_id"],
                "scenario": row["scenario"],
                "usage_count": row["usage_count"],
                "success_rate": row["success_rate"],
                "ready": True,
            })
        return results

    def encapsulate_skill(self, experience_id: str) -> str | None:
        """将高频成功经验封装为 Skill.md。"""
        card = self._experience.get(experience_id)
        if card is None:
            return None

        skill_name = f"auto_{card.scenario[:20].replace(' ', '_').replace('/', '_')}"
        skill_content = (
            f"# {skill_name}\n\n"
            f"## 触发场景\n{card.scenario}\n\n"
            f"## 执行方法\n{card.approach}\n\n"
            f"## 关键教训\n{card.lesson}\n\n"
            f"## 来源\n- 经验 ID: {experience_id}\n"
            f"- 复用次数: {card.usage_count}\n"
            f"- 成功率: {card.success_rate:.0%}\n"
            f"- 封装时间: {datetime.now(timezone.utc).isoformat()}\n"
        )

        skill_path = SKILLS_DIR / f"{skill_name}.md"
        skill_path.write_text(skill_content, encoding="utf-8")

        # 标记经验为 autoscale 轨道
        self._experience._db.execute(
            "UPDATE experiences SET self_improving_track = 'autoscale' WHERE experience_id = ?",
            (experience_id,),
        )
        self._experience._db.commit()

        logger.info("skill_encapsulated", experience_id=experience_id, skill=str(skill_path))
        return str(skill_path)

    def run_autoscale_cycle(self, agent_id: str) -> list[str]:
        """执行一次 AutoScale 周期：检查 → 封装 → 返回新 skill 路径列表。"""
        eligible = self.check_autoscale_eligible(agent_id)
        new_skills: list[str] = []

        for item in eligible:
            path = self.encapsulate_skill(item["experience_id"])
            if path:
                new_skills.append(path)

        return new_skills

    # ── Lifecycle Loop ────────────────────────────────────

    async def evolution_cycle(self, agent_id: str) -> dict[str, Any]:
        """自进化生命周期闭环。

        执行 → 经验提取 → 失败→self-improving / 成功→AutoScale → memify。
        """
        report: dict[str, Any] = {
            "self_improving": {},
            "autoscale": {},
        }

        # 1. 检查需要进化的经验
        recent = self._experience.search(
            ExperienceQuery(query_text="", agent_id=agent_id, top_k=10, include_shared=False)
        )
        for card in recent:
            if card.success_rate < 0.5:
                # 失败路径 → self-improving
                self.capture_failure(
                    agent_id, card.scenario,
                    error=f"success_rate={card.success_rate}",
                    fix=card.lesson,
                )
                report["self_improving"][card.experience_id] = "captured_failure"
            elif card.autoscale_eligible:
                report["autoscale"][card.experience_id] = "ready"

        # 2. 运行 AutoScale 封装
        new_skills = self.run_autoscale_cycle(agent_id)
        report["autoscale"]["new_skills"] = new_skills

        # 3. 清理
        self._experience.memify()
        report["memify"] = "done"

        logger.info("evolution_cycle_complete", agent_id=agent_id, **report)
        return report

    # ── Read ──────────────────────────────────────────────

    def get_learnings(self) -> str:
        """读取 Learnings.md 内容。"""
        if LEARNINGS_PATH.exists():
            return LEARNINGS_PATH.read_text(encoding="utf-8")
        return ""

    def get_corrections(self) -> str:
        """读取 corrections.md 内容。"""
        if CORRECTIONS_PATH.exists():
            return CORRECTIONS_PATH.read_text(encoding="utf-8")
        return ""

    # ── Helpers ───────────────────────────────────────────

    def _append_learnings(self, entry: str) -> None:
        with open(LEARNINGS_PATH, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(entry)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _append_corrections(self, entry: str) -> None:
        with open(CORRECTIONS_PATH, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(entry)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
