"""WhatsApp orchestration service.

Handles incoming messages:
1. Idempotency validation
2. Server-side tenant resolution
3. Conversation retrieval & user message storage
4. Invocation of the existing RAG answer pipeline (/api/rag/ask)
5. Assistant response persistence
6. Outgoing message transmission via WhatsApp Cloud API client
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from ..rag import service as rag_service
from . import conversation
from .client import whatsapp_client
from .config import WhatsAppSettings, get_whatsapp_settings
from .schemas import ParsedWhatsAppMessage

logger = logging.getLogger(__name__)


async def process_incoming_whatsapp_message(
    message: ParsedWhatsAppMessage,
    settings: Optional[WhatsAppSettings] = None,
) -> Dict[str, Any]:
    """Execute the full processing pipeline for one incoming WhatsApp message."""
    cfg = settings or get_whatsapp_settings()
    started = time.perf_counter()

    msg_id = message.external_message_id
    phone = message.sender_phone_number
    phone_masked = phone[-4:].rjust(len(phone), "*") if phone else "unknown"

    logger.info("whatsapp: processing incoming message (msg_id=%s, from=%s)", msg_id, phone_masked)

    # 1. Idempotency check: Skip if already processed
    if await conversation.is_message_already_processed(msg_id):
        logger.info("whatsapp: message %s has already been processed. Skipping.", msg_id)
        return {
            "status": "skipped",
            "reason": "already_processed",
            "external_message_id": msg_id,
        }

    # 2. Server-side tenant resolution
    tenant_id = cfg.resolve_tenant(message.phone_number_id)
    logger.info("whatsapp: resolved tenant_id='%s' for phone_number_id='%s'", tenant_id, message.phone_number_id)

    # 3. Conversation identification / creation
    conv = await conversation.get_or_create_conversation(
        tenant_id=tenant_id,
        phone_number=phone,
        channel="whatsapp",
        metadata={
            "display_phone_number": message.display_phone_number,
            "phone_number_id": message.phone_number_id,
        },
    )
    conversation_id = conv["id"]

    # 4. Save incoming user message
    user_msg_doc = await conversation.save_message(
        conversation_id=conversation_id,
        role="user",
        content=message.message_text,
        external_message_id=msg_id,
        tenant_id=tenant_id,
        metadata={
            "phone_number_id": message.phone_number_id,
            "timestamp": message.timestamp,
        },
    )

    # 5. Fetch prior conversation turns for multi-turn RAG context
    history = await conversation.get_conversation_history(
        conversation_id=conversation_id,
        limit=6,
    )

    # 6. Call existing RAG answer pipeline (/api/rag/ask)
    answer = ""
    sources = []
    rag_error = None
    rag_latency = {}

    try:
        rag_outcome = await rag_service.ask(
            query=message.message_text,
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            conversation_history=history,
        )
        answer = rag_outcome.get("answer", "")
        sources = rag_outcome.get("sources", [])
        rag_latency = rag_outcome.get("latency", {})
    except Exception as error:
        rag_error = str(error)
        logger.error("whatsapp: RAG answer generation failed for msg_id=%s: %s", msg_id, error)
        if cfg.whatsapp_send_fallback_on_error:
            answer = cfg.whatsapp_fallback_message

    if not answer:
        answer = cfg.whatsapp_fallback_message

    # 7. Save assistant message
    asst_msg_doc = await conversation.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        tenant_id=tenant_id,
        metadata={
            "sources": sources,
            "sources_count": len(sources),
            "rag_error": rag_error,
            "latency": rag_latency,
        },
    )

    # 8. Send reply back to customer via WhatsApp Cloud API client
    send_status = "not_configured"
    if cfg.is_configured:
        try:
            target_pid = message.phone_number_id or cfg.whatsapp_phone_number_id
            await whatsapp_client.send_text_message(
                recipient_phone_number=phone,
                message=answer,
                phone_number_id=target_pid,
            )
            send_status = "sent"
        except Exception as error:
            send_status = "failed"
            logger.error("whatsapp: could not send WhatsApp reply to %s: %s", phone_masked, error)
    else:
        logger.info("whatsapp: credentials not set; skipping live Meta API dispatch")

    took_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "whatsapp: message processing completed in %d ms (tenant=%s, conv=%s, send=%s)",
        took_ms, tenant_id, conversation_id, send_status,
    )

    return {
        "status": "processed",
        "external_message_id": msg_id,
        "tenant_id": tenant_id,
        "conversation_id": conversation_id,
        "phone_number": phone,
        "query": message.message_text,
        "answer": answer,
        "sources": sources,
        "sources_count": len(sources),
        "send_status": send_status,
        "latency": {
            "retrieval_ms": rag_latency.get("retrieval_ms", 0),
            "rerank_ms": rag_latency.get("rerank_ms", 0),
            "llm_ms": rag_latency.get("llm_ms", 0),
            "total_ms": took_ms,
        },
        "user_message": user_msg_doc,
        "assistant_message": asst_msg_doc,
        "took_ms": took_ms,
    }

