"""Back-channel into the OpenClaw Gateway's REST API.

This is the only way the agent can affect WhatsApp, and it is deliberately
narrow. The agent never imports Baileys, never opens a socket, never touches a
session, and never calls a client application's own API directly. It asks the
gateway, and the gateway - which owns WhatsApp and owns enforcement - decides.

Block and unblock are passthroughs twice over: agent -> gateway -> the client
server that owns the number and actually stores the decision.

NO ROUTE IS ASSUMED beyond the gateway's documented paths. When GATEWAY_BASE_URL
is unset, the block tools are simply not offered to the model.

Every call logs the URL, the payload, the response status and any error body.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from ..config import settings

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


async def startup() -> None:
    global _client
    if not settings.gateway_base_url:
        logger.info("gateway back-channel: disabled (GATEWAY_BASE_URL is not set)")
        return

    try:
        if _client is None:
            headers = {"accept": "application/json"}
            if settings.gateway_token:
                headers["x-gateway-token"] = settings.gateway_token
            _client = httpx.AsyncClient(
                base_url=settings.gateway_base_url.rstrip("/"),
                timeout=settings.gateway_timeout_s,
                headers=headers,
            )
        logger.info(
            "gateway back-channel: base_url=%s token=%s (block tools %s)",
            settings.gateway_base_url,
            "set" if settings.gateway_token else "MISSING",
            "enabled" if settings.block_tools_available else "disabled",
        )
    except Exception as error:  # noqa: BLE001
        _client = None
        logger.warning("gateway back-channel failed to initialise: %s", error)


async def shutdown() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as error:  # noqa: BLE001
            logger.warning("error closing gateway client: %s", error)
        finally:
            _client = None


def _redact(value: Any, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= 180 else f"{value[:180]}…[{len(value)} chars]"
    if depth > 3:
        return "[deep]"
    if isinstance(value, list):
        return [_redact(item, depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        return {key: _redact(item, depth + 1) for key, item in value.items()}
    return value


async def _call(
    name: str,
    path: str,
    *,
    method: str = "POST",
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _client is None:
        raise RuntimeError(
            "GATEWAY_BASE_URL is not configured; the agent has no way to reach the gateway"
        )

    url = f"{settings.gateway_base_url.rstrip('/')}{path}"
    started = time.perf_counter()
    logger.info(
        "gateway request: %s %s %s payload=%s",
        name,
        method,
        url,
        _redact(json_body or params or {}),
    )

    try:
        response = await _client.request(method, path, params=params, json=json_body)
    except httpx.HTTPError as error:
        logger.error("gateway request failed: %s %s error=%s", name, url, error)
        raise

    duration_ms = int((time.perf_counter() - started) * 1000)

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.is_error:
        logger.error(
            "gateway error: %s %s status=%s duration_ms=%s body=%s",
            name,
            url,
            response.status_code,
            duration_ms,
            _redact(payload or response.text),
        )
        response.raise_for_status()

    logger.info(
        "gateway response: %s %s status=%s duration_ms=%s body=%s",
        name,
        url,
        response.status_code,
        duration_ms,
        _redact(payload),
    )
    return payload if isinstance(payload, dict) else {"result": payload}


async def block_contact(
    whatsapp_id: str,
    contact: str,
    reason: str,
    blocked_by: str = "ai-agent",
) -> dict[str, Any]:
    """POST /whatsapp/block - forwarded by the gateway to the number's owner."""
    return await _call(
        "BLOCK",
        "/whatsapp/block",
        json_body={
            "whatsapp_id": whatsapp_id,
            "contact": contact,
            "reason": reason,
            "blocked_by": blocked_by,
        },
    )


async def unblock_contact(whatsapp_id: str, contact: str, reason: str | None = None) -> dict[str, Any]:
    """POST /whatsapp/unblock."""
    return await _call(
        "UNBLOCK",
        "/whatsapp/unblock",
        json_body={
            "whatsapp_id": whatsapp_id,
            "contact": contact,
            "reason": reason,
            "blocked_by": "ai-agent",
        },
    )


async def check_block(whatsapp_id: str, contact: str) -> dict[str, Any]:
    """GET /whatsapp/block/{contact}?whatsapp_id=..."""
    return await _call(
        "BLOCK_CHECK",
        f"/whatsapp/block/{contact}",
        method="GET",
        params={"whatsapp_id": whatsapp_id},
    )
