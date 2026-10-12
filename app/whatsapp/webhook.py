"""Webhook verification and payload parsing for Meta WhatsApp Cloud API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .schemas import ParsedWhatsAppMessage

logger = logging.getLogger(__name__)


def verify_webhook_challenge(
    mode: Optional[str],
    token: Optional[str],
    challenge: Optional[str],
    expected_token: Optional[str],
) -> Optional[str]:
    """Verify Meta Webhook challenge during initial webhook setup.

    Args:
        mode: hub.mode parameter (should be "subscribe")
        token: hub.verify_token parameter
        challenge: hub.challenge parameter
        expected_token: WHATSAPP_VERIFY_TOKEN from environment

    Returns:
        The challenge string if verification passes, else None.
    """
    if not expected_token:
        logger.warning("whatsapp: verification failed - WHATSAPP_VERIFY_TOKEN is not configured on server")
        return None

    if mode == "subscribe" and token == expected_token and challenge:
        logger.info("whatsapp: webhook verification succeeded")
        return challenge

    logger.warning("whatsapp: webhook verification failed - mode=%s, token_match=%s", mode, token == expected_token)
    return None


def parse_webhook_payload(payload: Dict[str, Any]) -> List[ParsedWhatsAppMessage]:
    """Parse a raw Meta WhatsApp webhook dictionary into normalized ParsedWhatsAppMessage objects.

    Extracts text messages and safely ignores statuses, read receipts, delivery reports, and unsupported media types.
    """
    messages: List[ParsedWhatsAppMessage] = []

    if not isinstance(payload, dict):
        return messages

    entries = payload.get("entry", [])
    if not isinstance(entries, list):
        return messages

    for entry in entries:
        changes = entry.get("changes", [])
        if not isinstance(changes, list):
            continue

        for change in changes:
            value = change.get("value", {})
            if not isinstance(value, dict):
                continue

            metadata = value.get("metadata", {})
            phone_number_id = str(metadata.get("phone_number_id") or "") or None
            display_phone_number = metadata.get("display_phone_number")

            # Check if this change contains incoming user messages
            raw_messages = value.get("messages", [])
            if not isinstance(raw_messages, list):
                continue

            for msg in raw_messages:
                if not isinstance(msg, dict):
                    continue

                msg_id = msg.get("id")
                from_number = msg.get("from")
                msg_type = msg.get("type", "unknown")
                timestamp = str(msg.get("timestamp", ""))

                if not msg_id or not from_number:
                    continue

                if msg_type == "text":
                    text_obj = msg.get("text", {})
                    body = text_obj.get("body", "").strip() if isinstance(text_obj, dict) else ""

                    if body:
                        messages.append(
                            ParsedWhatsAppMessage(
                                external_message_id=msg_id,
                                sender_phone_number=from_number,
                                message_type="text",
                                message_text=body,
                                timestamp=timestamp,
                                phone_number_id=phone_number_id,
                                display_phone_number=display_phone_number,
                                raw_metadata=metadata,
                            )
                        )
                else:
                    logger.info(
                        "whatsapp: skipping non-text message type '%s' (msg_id=%s, from=%s)",
                        msg_type, msg_id, from_number[-4:].rjust(len(from_number), "*") if from_number else "",
                    )

    return messages
