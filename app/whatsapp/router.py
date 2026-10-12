"""FastAPI HTTP routes for Meta WhatsApp Cloud API integration."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Response, status
from fastapi.responses import PlainTextResponse

from . import service, webhook
from .config import get_whatsapp_settings
from .schemas import (
    ParsedWhatsAppMessage,
    WhatsAppWebhookAck,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/whatsapp",
    tags=["whatsapp"],
)


@router.get(
    "/webhook",
    summary="Meta WhatsApp webhook verification",
    response_class=PlainTextResponse,
    responses={
        200: {"description": "Returns hub.challenge when verification token matches"},
        403: {"description": "Verification token mismatch or missing"},
    },
)
async def verify_webhook(
    hub_mode: Optional[str] = Query(default=None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(default=None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(default=None, alias="hub.challenge"),
) -> Response:
    """Endpoint called by Meta when verifying webhook URLs in the Developer Portal."""
    cfg = get_whatsapp_settings()
    challenge = webhook.verify_webhook_challenge(
        mode=hub_mode,
        token=hub_verify_token,
        challenge=hub_challenge,
        expected_token=cfg.whatsapp_verify_token,
    )
    if challenge is not None:
        return PlainTextResponse(content=challenge, status_code=status.HTTP_200_OK)

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="WhatsApp webhook verification failed.",
    )


@router.post(
    "/webhook",
    response_model=WhatsAppWebhookAck,
    summary="Meta WhatsApp incoming webhook event listener",
    status_code=status.HTTP_200_OK,
)
async def receive_webhook(
    background_tasks: BackgroundTasks,
    payload: Dict[str, Any] = Body(...),
) -> WhatsAppWebhookAck:
    """Receives event notifications from Meta WhatsApp Cloud API.

    Parses incoming messages and dispatches them asynchronously to the RAG processor,
    returning an immediate 200 OK acknowledgment to prevent Meta webhook retries.
    """
    cfg = get_whatsapp_settings()
    messages = webhook.parse_webhook_payload(payload)

    for msg in messages:
        background_tasks.add_task(
            service.process_incoming_whatsapp_message, msg, cfg
        )

    return WhatsAppWebhookAck(status="ok")


@router.get(
    "/health",
    summary="WhatsApp integration health and configuration status",
)
async def whatsapp_health() -> Dict[str, Any]:
    """Returns WhatsApp integration health and configuration readiness."""
    cfg = get_whatsapp_settings()
    return {
        "status": "healthy",
        "configuration": cfg.public_state(),
    }
