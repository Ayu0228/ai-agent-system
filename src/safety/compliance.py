"""Compliance & Ethics Governance — 合规审计、偏见检测、伦理红线。

ref: OpenAI — "Practices for Governing Agentic AI Systems" (2024)
ref: Anthropic — safety guardrails and responsible AI deployment
ref: EU AI Act / GDPR — regulatory compliance requirements

审计维度:
  1. 偏见检测 — 输出中的歧视性/偏见性语言
  2. 合规审计 — 完整的操作审计追踪
  3. 伦理红线 — 预定义的禁止行为列表
  4. 敏感信息检测 — PII/PHI/PCI 泄露检测
  5. 合规报告 — 自动生成合规报告
"""

from __future__ import annotations

import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ComplianceLevel(str, Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


@dataclass
class ComplianceCheck:
    """单次合规检查结果。"""
    check_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    rule_name: str = ""
    level: ComplianceLevel = ComplianceLevel.COMPLIANT
    passed: bool = True
    details: str = ""
    evidence: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ComplianceReport:
    """合规报告。"""
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_id: str = ""
    session_id: str = ""
    overall: ComplianceLevel = ComplianceLevel.COMPLIANT
    checks: list[ComplianceCheck] = field(default_factory=list)
    violations: int = 0
    warnings: int = 0
    generated_at: float = field(default_factory=time.time)
    summary: str = ""


class ComplianceEngine:
    """合规与伦理治理引擎。

    用法:
        engine = ComplianceEngine()
        engine.add_rule("no_pii", r"\\b\\d{17}\\b", ComplianceLevel.CRITICAL, "银行卡号")

        report = engine.audit_output(
            text="用户输出内容...",
            agent_id="researcher",
        )
        if report.overall in (ComplianceLevel.VIOLATION, ComplianceLevel.CRITICAL):
            raise ComplianceViolation(report)
    """

    def __init__(self) -> None:
        self._rules: list[dict[str, Any]] = []
        self._audit_log: list[ComplianceReport] = []
        self._add_default_rules()

    # ── 规则管理 ───────────────────────────────────

    def add_rule(self, name: str, pattern: str,
                 level: ComplianceLevel, description: str = "") -> None:
        self._rules.append({
            "name": name,
            "pattern": re.compile(pattern, re.IGNORECASE),
            "level": level,
            "description": description,
        })

    def _add_default_rules(self) -> None:
        """添加默认合规规则。"""
        # PII 检测
        self.add_rule("cn_id_card", r"\b\d{15}(\d{2}[0-9xX])?\b",
                      ComplianceLevel.CRITICAL, "中国身份证号")
        self.add_rule("cn_phone", r"\b1[3-9]\d{9}\b",
                      ComplianceLevel.VIOLATION, "中国手机号")
        self.add_rule("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                      ComplianceLevel.WARNING, "邮箱地址")
        self.add_rule("credit_card", r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                      ComplianceLevel.CRITICAL, "信用卡号")

        # 隐私/安全
        self.add_rule("api_key_leak", r"\b(sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b",
                      ComplianceLevel.CRITICAL, "API Key 泄露")
        self.add_rule("password_leak", r"\b(password|passwd|secret)\s*[:=]\s*\S+",
                      ComplianceLevel.CRITICAL, "密码泄露")

        # 偏见/歧视
        bias_terms = [
            r"(所有|全部|凡?是).*(都|就).*(不行|不会|不能|傻|笨|蠢|差)",
            r"(男的|女的).*(天生|就是|都不)",
        ]
        for i, pat in enumerate(bias_terms):
            self.add_rule(f"bias_pattern_{i}", pat,
                          ComplianceLevel.WARNING, "潜在偏见表述")

        # 伦理红线
        red_lines = [
            (r"(制造|合成|制备).*(毒品|违禁药品)", "非法物品制造"),
            (r"(入侵|破解|盗取).*(账号|密码|系统)", "网络攻击指导"),
            (r"(自杀|自残|伤害自己).*(方法|步骤)", "自伤行为"),
            (r"(如何|怎样).*(洗钱|逃税|诈骗)", "金融犯罪"),
        ]
        for pat, desc in red_lines:
            self.add_rule(f"redline_{desc}", pat, ComplianceLevel.CRITICAL, desc)

        logger.info("compliance_rules_loaded", count=len(self._rules))

    # ── 审计 ───────────────────────────────────────

    def audit_output(self, text: str, agent_id: str = "",
                     session_id: str = "",
                     context: dict[str, Any] | None = None) -> ComplianceReport:
        """对 agent 输出进行全面合规审计。"""
        report = ComplianceReport(
            agent_id=agent_id,
            session_id=session_id,
        )

        for rule in self._rules:
            matches = rule["pattern"].findall(text) if isinstance(text, str) else []
            if matches:
                check = ComplianceCheck(
                    rule_name=rule["name"],
                    level=rule["level"],
                    passed=False,
                    details=f"发现 {len(matches)} 处匹配: {rule.get('description', rule['name'])}",
                    evidence=str(matches[:3]),  # 只记录前3个
                )
                report.checks.append(check)

                if rule["level"] == ComplianceLevel.CRITICAL:
                    report.violations += 1
                    report.overall = ComplianceLevel.CRITICAL
                elif rule["level"] == ComplianceLevel.VIOLATION:
                    report.violations += 1
                    if report.overall != ComplianceLevel.CRITICAL:
                        report.overall = ComplianceLevel.VIOLATION
                elif rule["level"] == ComplianceLevel.WARNING:
                    report.warnings += 1
                    if report.overall == ComplianceLevel.COMPLIANT:
                        report.overall = ComplianceLevel.WARNING
            else:
                report.checks.append(ComplianceCheck(
                    rule_name=rule["name"],
                    passed=True,
                ))

        report.summary = self._summarize(report)
        self._audit_log.append(report)

        if report.overall in (ComplianceLevel.VIOLATION, ComplianceLevel.CRITICAL):
            logger.warning("compliance_violation", agent=agent_id,
                          level=report.overall.value,
                          violations=report.violations,
                          summary=report.summary)

        return report

    def audit_tool_call(self, tool_name: str, tool_input: dict[str, Any],
                        tool_output: Any, agent_id: str = "") -> ComplianceReport:
        """审计工具调用的合规性。"""
        # 审查输入
        input_text = str(tool_input)
        report = self.audit_output(input_text, agent_id=agent_id)

        # 审查输出
        output_text = str(tool_output) if tool_output else ""
        output_report = self.audit_output(output_text, agent_id=agent_id)

        # 合并
        report.checks.extend(output_report.checks)
        report.violations += output_report.violations
        report.warnings += output_report.warnings

        if output_report.overall.value > report.overall.value:
            report.overall = output_report.overall

        return report

    # ── 报告 ───────────────────────────────────────

    def _summarize(self, report: ComplianceReport) -> str:
        if report.overall == ComplianceLevel.COMPLIANT:
            return "所有合规检查通过"
        elif report.overall == ComplianceLevel.WARNING:
            return f"{report.warnings} 个警告需要关注"
        elif report.overall == ComplianceLevel.VIOLATION:
            return f"{report.violations} 个违规需要处理"
        else:
            return f"严重违规: {report.violations} 个发现"

    def get_audit_history(self, agent_id: str = "",
                          limit: int = 50) -> list[ComplianceReport]:
        reports = self._audit_log
        if agent_id:
            reports = [r for r in reports if r.agent_id == agent_id]
        return reports[-limit:]

    def get_stats(self) -> dict[str, Any]:
        level_counts = defaultdict(int)
        for r in self._audit_log:
            level_counts[r.overall.value] += 1

        return {
            "total_audits": len(self._audit_log),
            "pass_rate": f"{level_counts['compliant'] / max(1, len(self._audit_log)):.1%}",
            "by_level": dict(level_counts),
            "total_rules": len(self._rules),
        }
