"""Read-only back-channel to the existing Laravel application.

Used exclusively by agent tools. This service never caches, mirrors or persists
anything it reads - Laravel stays the single source of truth.

NO ROUTE IS ASSUMED: `KNOWLEDGE_API` and `HANDOFF_API` default to empty and the
matching tool is simply not offered to the model until one is configured. A
value may be a path (joined onto LARAVEL_BASE_URL) or a full URL; `{id}` is
replaced with the whatsapp_id.

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
    if not settings.laravel_base_url and not settings.knowledge_api and not settings.handoff_api:
        logger.info("laravel connector: disabled (LARAVEL_BASE_URL is not set)")
        return

    try:
        if _client is None:
            headers = {"accept": "application/json"}
            if settings.laravel_api_token:
                headers["authorization"] = f"Bearer {settings.laravel_api_token}"
            _client = httpx.AsyncClient(timeout=settings.laravel_timeout_s, headers=headers)

        logger.info(
            "laravel connector: base_url=%s token=%s knowledge_api=%s handoff_api=%s",
            settings.laravel_base_url or "(unset)",
            "set" if settings.laravel_api_token else "MISSING",
            settings.knowledge_api or "(not configured)",
            settings.handoff_api or "(not configured)",
        )
    except Exception as error:  # noqa: BLE001
        _client = None
        logger.warning("laravel connector failed to initialise: %s", error)


async def shutdown() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as error:  # noqa: BLE001
            logger.warning("error closing laravel client: %s", error)
        finally:
            _client = None


def _resolve(endpoint: str, whatsapp_id: str | None) -> str:
    value = endpoint.replace("{id}", str(whatsapp_id or ""))
    if value.startswith("http://") or value.startswith("https://"):
        return value
    base = settings.laravel_base_url.rstrip("/")
    if not base:
        raise RuntimeError("LARAVEL_BASE_URL is not configured")
    return f"{base}{value if value.startswith('/') else '/' + value}"


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
    endpoint: str,
    *,
    method: str,
    whatsapp_id: str | None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _client is None:
        raise RuntimeError("Laravel back-channel is not initialised")
    if not endpoint:
        raise RuntimeError(f"{name} is not configured")

    url = _resolve(endpoint, whatsapp_id)
    started = time.perf_counter()
    logger.info(
        "connector request: %s %s %s payload=%s",
        name,
        method,
        url,
        _redact(json_body or params or {}),
    )

    try:
        response = await _client.request(method, url, params=params, json=json_body)
    except httpx.HTTPError as error:
        logger.error("connector request failed: %s %s error=%s", name, url, error)
        raise

    duration_ms = int((time.perf_counter() - started) * 1000)

    if response.is_error:
        logger.error(
            "connector error: %s %s status=%s duration_ms=%s body=%s",
            name,
            url,
            response.status_code,
            duration_ms,
            _redact(response.text),
        )
        response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    logger.info(
        "connector response: %s %s status=%s duration_ms=%s body=%s",
        name,
        url,
        response.status_code,
        duration_ms,
        _redact(payload),
    )

    return payload if isinstance(payload, dict) else {"results": payload}


async def fetch_knowledge(whatsapp_id: str | None, query: str) -> dict[str, Any]:
    """KNOWLEDGE_API - the account's own catalog/policy/FAQ data."""
    return await _call(
        "KNOWLEDGE_API",
        settings.knowledge_api,
        method="GET",
        whatsapp_id=whatsapp_id,
        params={"q": query},
    )


async def request_handoff(
    whatsapp_id: str | None,
    contact: dict[str, Any],
    reason: str,
    summary: str,
) -> dict[str, Any]:
    """HANDOFF_API - ask Laravel to escalate this conversation to a human."""
    return await _call(
        "HANDOFF_API",
        settings.handoff_api,
        method="POST",
        whatsapp_id=whatsapp_id,
        json_body={
            "whatsapp_id": whatsapp_id,
            "contact": contact,
            "reason": reason,
            "summary": summary,
        },
    )
