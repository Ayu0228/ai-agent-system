"""审批路由：按风险等级 L0-L3 路由到不同审批策略。"""

from __future__ import annotations

import asyncio
import uuid

import structlog

from src.safety.hooks import OPERATION_RISK, DEFAULT_RISK, RiskLevel
from src.shared.config import get_settings
from src.shared.models import ApprovalRequest, ApprovalResponse

logger = structlog.get_logger()


class ApprovalRouter:
    """审批路由引擎。管理审批请求的生命周期。"""

    def __init__(self) -> None:
        self._pending: dict[str, ApprovalRequest] = {}
        self._timeout = get_settings().approval_timeout

    async def request(
        self, agent_id: str, action: str, summary: str, *, trace_id: str
    ) -> ApprovalRequest:
        """发起审批请求。L2/L3 操作触发。"""
        risk = OPERATION_RISK.get(action, DEFAULT_RISK)
        req = ApprovalRequest(
            agent_id=agent_id,
            action=action,
            risk_level=risk,
            summary=summary,
        )
        self._pending[req.id] = req

        if risk == RiskLevel.L0:
            # 自动通过
            req.status = "approved"
            del self._pending[req.id]
            logger.debug("auto_approved_l0", id=req.id, action=action)
            return req

        if risk == RiskLevel.L1:
            # 自动通过 + 日志通知
            req.status = "approved"
            del self._pending[req.id]
            logger.info("auto_approved_l1", id=req.id, action=action, summary=summary)
            return req

        # L2 / L3 → 等待人工审批
        logger.info("approval_pending", id=req.id, action=action, risk=f"L{risk.value}")
        return req

    async def wait_for_decision(
        self, request_id: str, timeout: int | None = None
    ) -> ApprovalResponse:
        """阻塞等待审批结果（超时自动拒绝）。"""
        timeout = timeout if timeout is not None else self._timeout
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            req = self._pending.get(request_id)
            if req is None or req.status != "pending":
                return ApprovalResponse(
                    request_id=request_id,
                    approved=req.status == "approved" if req else False,
                    reason="timeout" if req is None else req.status,
                )
            await asyncio.sleep(1)

        # 超时 → 自动拒绝
        if request_id in self._pending:
            self._pending[request_id].status = "timeout"
            del self._pending[request_id]
        return ApprovalResponse(request_id=request_id, approved=False, reason="timeout")

    def approve(self, request_id: str, approver: str = "ayu") -> ApprovalResponse:
        """人工通过审批。"""
        req = self._pending.pop(request_id, None)
        if req is None:
            return ApprovalResponse(request_id=request_id, approved=False, reason="not_found")
        req.status = "approved"
        logger.info("approval_approved", id=request_id, approver=approver)
        return ApprovalResponse(request_id=request_id, approved=True, approver=approver)

    def reject(self, request_id: str, reason: str = "", approver: str = "ayu") -> ApprovalResponse:
        """人工拒绝审批。"""
        req = self._pending.pop(request_id, None)
        if req is None:
            return ApprovalResponse(request_id=request_id, approved=False, reason="not_found")
        req.status = "rejected"
        logger.info("approval_rejected", id=request_id, reason=reason, approver=approver)
        return ApprovalResponse(request_id=request_id, approved=False, reason=reason, approver=approver)
