"""OpenAI-compatible endpoint, so OpenClaw can use this service as a model provider.

OpenClaw references models as `provider/model` and calls the provider's
`baseUrl` with the OpenAI chat-completions shape. By registering this service as
a custom provider, every WhatsApp turn OpenClaw handles is routed here - which
is where Laravel's rules and Groq's reply get applied.

    WhatsApp -> OpenClaw -> POST /v1/chat/completions (this file)
                               -> Laravel  (authorize / prompt)
                               -> Groq     (reply)
                            <- OpenAI-shaped response -> OpenClaw -> WhatsApp

The first job of this endpoint is DISCOVERY: it logs the complete inbound
request so we can see exactly what OpenClaw sends - particularly whether the
peer's phone number is recoverable, which Laravel needs for per-number rules.
Set LOG_LEVEL=DEBUG to see full bodies.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ..config import settings
from ..schemas import ReplyRequest
from ..security import require_token
from ..services import agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])

# Matches +919..., 15551234567 etc. inside session keys / prompts.
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")
# OpenClaw session keys look like: agent:main:whatsapp:direct:+919...
_SESSION_KEY_RE = re.compile(r"agent:[\w-]+:(?P<channel>[\w-]+):(?P<kind>[\w-]+):(?P<peer>[^\s\"']+)")


def _extract_identity(body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """Best-effort recovery of who we are talking to.

    OpenAI's schema has no field for this, so OpenClaw may surface it via the
    `user` field, a header, or inside its system prompt. We try all three and
    log what we found so the guesswork can be replaced with certainty.
    """
    identity: dict[str, Any] = {
        "session_key": None,
        "channel": None,
        "kind": None,
        "peer": None,
        "source": None,
    }

    # 1. OpenAI's `user` field - the most likely carrier.
    candidate = body.get("user")

    # 2. Any header that looks like a session/conversation id.
    if not candidate:
        for key, value in headers.items():
            if key.lower() in {"x-session-key", "x-openclaw-session", "x-conversation-id"}:
                candidate = value
                identity["source"] = f"header:{key}"
                break
    else:
        identity["source"] = "body.user"

    # 3. Fall back to scanning system messages for a session key.
    haystack = candidate or ""
    if not haystack:
        for message in body.get("messages", []):
            if message.get("role") == "system" and isinstance(message.get("content"), str):
                haystack += message["content"] + "\n"
        if haystack:
            identity["source"] = "system_prompt_scan"

    match = _SESSION_KEY_RE.search(haystack)
    if match:
        identity["session_key"] = match.group(0)
        identity["channel"] = match.group("channel")
        identity["kind"] = match.group("kind")
        identity["peer"] = match.group("peer")
        return identity

    if candidate:
        identity["session_key"] = candidate

    phone = _PHONE_RE.search(haystack)
    if phone:
        identity["peer"] = phone.group(0).strip()

    return identity


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]], str]:
    """Return (system_text, history, latest_user_message)."""
    system_parts: list[str] = []
    history: list[dict[str, str]] = []
    latest = ""

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        # Content can be a list of parts in newer OpenAI schemas.
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            continue

        if role == "system":
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            history.append({"role": role, "content": content})

    for turn in reversed(history):
        if turn["role"] == "user":
            latest = turn["content"]
            break

    # Drop the latest user turn from history - it is the message being answered.
    if latest and history and history[-1]["role"] == "user":
        history = history[:-1]

    return "\n\n".join(system_parts), history, latest


def _completion_response(model: str, text: str, usage: Any) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": (getattr(usage, "input_tokens", 0) or 0)
            + (getattr(usage, "output_tokens", 0) or 0),
        },
    }


def _stream_response(model: str, text: str) -> StreamingResponse:
    """Minimal SSE stream: one content chunk, then a stop chunk, then [DONE]."""
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    def chunk(delta: dict[str, Any], finish: str | None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    def generate():
        yield chunk({"role": "assistant", "content": text}, None)
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """Model discovery. OpenClaw may call this when resolving `provider/model`."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.openai_compat_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "whatsapp-ai-stack",
            }
        ],
    }


@router.post("/chat/completions", dependencies=[Depends(require_token)])
async def chat_completions(request: Request) -> Any:
    """OpenAI-compatible chat completions, answered by Groq under Laravel's rules."""
    try:
        body = await request.json()
    except Exception as error:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Body must be JSON"
        ) from error

    headers = dict(request.headers)
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="messages must be a non-empty array",
        )

    # -- DISCOVERY LOGGING --------------------------------------------------
    # Everything OpenClaw sends, so the identity heuristics above can be
    # replaced with the real field once we have seen a live request.
    safe_headers = {
        k: ("[REDACTED]" if k.lower() in {"authorization", "x-api-key"} else v)
        for k, v in headers.items()
    }
    logger.info(
        "openai-compat request: model=%s stream=%s messages=%d",
        body.get("model"),
        body.get("stream"),
        len(messages),
    )
    logger.debug("openai-compat headers: %s", json.dumps(safe_headers, default=str))
    logger.debug("openai-compat body: %s", json.dumps(body, default=str)[:8000])

    identity = _extract_identity(body, headers)
    logger.info("openai-compat identity: %s", json.dumps(identity, default=str))

    system_text, history, latest = _split_messages(messages)
    if not latest.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="no user message to answer",
        )

    # -- build the internal request ----------------------------------------
    # OpenClaw's own system prompt is preserved as reference context rather
    # than discarded; Laravel's prompt still governs behaviour.
    context: dict[str, Any] = {
        "history": history,
        "contact": {"phone": identity.get("peer")},
        "channel": identity.get("channel") or "whatsapp",
        "session_key": identity.get("session_key"),
        "upstream_system_prompt": system_text or None,
    }

    # OpenClaw is just another client application. It has no way to send a
    # client_id or prompt_id through the OpenAI schema, so those come from
    # configuration: OPENCLAW_CLIENT_ID and OPENCLAW_PROMPT. The prompt is not
    # optional - with the built-in templates gone there is no default persona to
    # answer with, so a blank OPENCLAW_PROMPT is a configuration error, not a
    # silent fallback.
    reply_request = ReplyRequest(
        client_id=settings.openclaw_client_id or None,
        channel=identity.get("channel") or "whatsapp",
        contact_id=identity.get("peer"),
        prompt_id=settings.openclaw_prompt_id or None,
        system_prompt=settings.openclaw_prompt or None,
        message=latest,
        context=context,
        # OpenClaw keeps its own conversation history and replays it in
        # `messages`, so writing a second copy here would double-count turns.
        persist=False,
    )

    if not (settings.openclaw_prompt or settings.openclaw_prompt_id):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Neither OPENCLAW_PROMPT nor OPENCLAW_PROMPT_ID is set. This service "
                "ships no default prompt, so it cannot answer OpenClaw-routed turns."
            ),
        )

    if not settings.api_key_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No provider API key is configured. Set at least one of GROQ_API_KEY, "
                "OPENAI_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY."
            ),
        )

    try:
        result = await agent.generate_reply(reply_request)
    except Exception as error:  # noqa: BLE001
        logger.exception("openai-compat generation failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"reply generation failed: {error}",
        ) from error

    logger.info(
        "openai-compat reply: %d chars, %sms, provider=%s",
        len(result.reply),
        result.latency_ms,
        result.provider,
    )

    model_name = body.get("model") or settings.openai_compat_model_name

    if body.get("stream"):
        return _stream_response(model_name, result.reply)
    return _completion_response(model_name, result.reply, result.usage)
