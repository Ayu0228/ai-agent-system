"""飞书审批通知。通过 Webhook 发送审批卡片。"""

from __future__ import annotations

import httpx
import structlog

from src.shared.config import get_settings

logger = structlog.get_logger()


async def send_approval_card(
    title: str,
    summary: str,
    risk_level: str,
    agent_id: str,
    action: str,
) -> bool:
    """发送飞书审批卡片。"""
    webhook = get_settings().feishu_webhook_url
    if not webhook:
        logger.info("feishu_skipped_no_webhook", title=title)
        return False

    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"[审批] {title}"},
                "template": "red" if risk_level == "L3" else "yellow",
            },
            "elements": [
                {"tag": "div", "text": {"tag": "plain_text", "content": summary}},
                {"tag": "div", "text": {"tag": "lark_md", "content": f"**Agent:** {agent_id}\n**操作:** {action}\n**风险等级:** {risk_level}"}},
            ],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(webhook, json=payload)
            if resp.status_code == 200:
                logger.info("feishu_card_sent", title=title)
                return True
            logger.warning("feishu_card_failed", status=resp.status_code)
            return False
    except Exception as e:
        logger.error("feishu_card_error", error=str(e))
        return False
