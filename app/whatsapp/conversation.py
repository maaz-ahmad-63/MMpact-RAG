"""Conversation and message management for WhatsApp backed by MongoDB.

Stores complete WhatsApp conversation threads and messages in MongoDB collections:
- `whatsapp_conversations`: Tracks conversation metadata, customer phone numbers, and tenant IDs.
- `whatsapp_messages`: Stores all incoming user turns and generated assistant replies with timestamps, citations, and external WhatsApp message IDs for idempotency.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional

import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from ..config import settings
from ..services.mongodb_logger import mongodb_logger

logger = logging.getLogger(__name__)

# In-memory stores used when MongoDB is unavailable or for testing
_mem_conversations: Dict[str, Dict[str, Any]] = {}
_mem_messages: List[Dict[str, Any]] = []
_mem_processed_message_ids: set[str] = set()

_mongo_client: Optional[AsyncIOMotorClient] = None


def reset_memory_state() -> None:
    """Clear memory caches (primarily for clean test isolation)."""
    global _mem_conversations, _mem_messages, _mem_processed_message_ids
    _mem_conversations.clear()
    _mem_messages.clear()
    _mem_processed_message_ids.clear()


def is_mongodb_configured() -> bool:
    """Check if MongoDB connection URL is provided in configuration."""
    return bool(settings.mongodb_url)


async def get_mongo_db() -> Optional[AsyncIOMotorDatabase]:
    """Retrieve the active MongoDB database instance."""
    global _mongo_client
    if not is_mongodb_configured():
        return None

    # Reuse mongodb_logger client if available
    if mongodb_logger._client is not None and mongodb_logger._db is not None:
        return mongodb_logger._db

    if _mongo_client is None:
        try:
            _mongo_client = AsyncIOMotorClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=5000,
                tlsCAFile=certifi.where(),
            )
        except Exception as error:
            logger.error("whatsapp: could not initialize MongoDB client (%s)", error)
            return None

    return _mongo_client[settings.mongodb_database]


async def init_whatsapp_tables() -> None:
    """Initialize MongoDB collections and indexes for WhatsApp storage."""
    db = await get_mongo_db()
    if db is None:
        logger.info("whatsapp: MongoDB not configured; using in-memory conversation storage.")
        return

    try:
        convs_col = db["whatsapp_conversations"]
        msgs_col = db["whatsapp_messages"]

        # Indexes for conversations
        await convs_col.create_index(
            [("tenant_id", ASCENDING), ("phone_number", ASCENDING), ("channel", ASCENDING)],
            unique=True,
            name="uniq_tenant_phone_channel",
        )
        await convs_col.create_index([("updated_at", DESCENDING)], name="conv_updated_idx")

        # Indexes for messages
        await msgs_col.create_index(
            [("conversation_id", ASCENDING), ("created_at", ASCENDING)],
            name="msg_conv_created_idx",
        )
        # Sparse unique index on external_message_id for webhook idempotency
        await msgs_col.create_index(
            [("external_message_id", ASCENDING)],
            unique=True,
            sparse=True,
            name="uniq_external_msg_id",
        )
        logger.info("whatsapp: MongoDB collections and indexes initialized successfully.")
    except Exception as e:
        logger.error("whatsapp: could not initialize MongoDB indexes (%s)", e)


async def is_message_already_processed(external_message_id: Optional[str]) -> bool:
    """Check if an incoming WhatsApp external message ID has already been recorded."""
    if not external_message_id:
        return False

    db = await get_mongo_db()
    if db is None:
        return external_message_id in _mem_processed_message_ids

    try:
        msgs_col = db["whatsapp_messages"]
        doc = await msgs_col.find_one({"external_message_id": external_message_id}, {"_id": 1})
        return doc is not None
    except Exception as e:
        logger.warning("whatsapp: MongoDB idempotency check error (%s), checking memory fallback", e)
        return external_message_id in _mem_processed_message_ids


async def get_or_create_conversation(
    tenant_id: str,
    phone_number: str,
    channel: str = "whatsapp",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieve an existing conversation from MongoDB or create a new one."""
    clean_phone = "".join(c for c in phone_number if c.isdigit() or c == "+")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    db = await get_mongo_db()
    if db is None:
        key = f"{tenant_id}:{clean_phone}:{channel}"
        if key in _mem_conversations:
            return _mem_conversations[key]
        conv_id = f"conv_{uuid.uuid4().hex}"
        record = {
            "id": conv_id,
            "conversation_id": conv_id,
            "tenant_id": tenant_id,
            "phone_number": clean_phone,
            "channel": channel,
            "metadata": metadata or {},
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        _mem_conversations[key] = record
        return record

    try:
        convs_col = db["whatsapp_conversations"]
        doc = await convs_col.find_one(
            {"tenant_id": tenant_id, "phone_number": clean_phone, "channel": channel}
        )
        if doc:
            doc["id"] = doc.get("conversation_id", str(doc["_id"]))
            return doc

        conv_id = f"conv_{uuid.uuid4().hex}"
        new_doc = {
            "_id": conv_id,
            "conversation_id": conv_id,
            "tenant_id": tenant_id,
            "phone_number": clean_phone,
            "channel": channel,
            "metadata": metadata or {},
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        await convs_col.insert_one(new_doc)
        new_doc["id"] = conv_id
        return new_doc

    except Exception as e:
        logger.error("whatsapp: MongoDB error getting conversation (%s), falling back to memory", e)
        key = f"{tenant_id}:{clean_phone}:{channel}"
        if key not in _mem_conversations:
            conv_id = f"conv_{uuid.uuid4().hex}"
            _mem_conversations[key] = {
                "id": conv_id,
                "conversation_id": conv_id,
                "tenant_id": tenant_id,
                "phone_number": clean_phone,
                "channel": channel,
                "metadata": metadata or {},
                "created_at": now_iso,
                "updated_at": now_iso,
            }
        return _mem_conversations[key]


async def save_message(
    conversation_id: str,
    role: str,
    content: str,
    external_message_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tenant_id: str = "",
) -> Dict[str, Any]:
    """Record a user or assistant message in MongoDB."""
    msg_id = f"msg_{uuid.uuid4().hex}"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    record = {
        "id": msg_id,
        "message_id": msg_id,
        "conversation_id": conversation_id,
        "tenant_id": tenant_id,
        "external_message_id": external_message_id,
        "role": role,
        "content": content,
        "channel": "whatsapp",
        "metadata": metadata or {},
        "created_at": now_iso,
    }

    db = await get_mongo_db()
    if db is None:
        if external_message_id:
            _mem_processed_message_ids.add(external_message_id)
        _mem_messages.append(record)
        return record

    try:
        msgs_col = db["whatsapp_messages"]
        convs_col = db["whatsapp_conversations"]

        mongo_doc = {
            "_id": msg_id,
            "message_id": msg_id,
            "conversation_id": conversation_id,
            "tenant_id": tenant_id,
            "external_message_id": external_message_id,
            "role": role,
            "content": content,
            "channel": "whatsapp",
            "metadata": metadata or {},
            "created_at": now_iso,
        }
        await msgs_col.insert_one(mongo_doc)

        # Update conversation updated_at
        await convs_col.update_one(
            {"$or": [{"_id": conversation_id}, {"conversation_id": conversation_id}]},
            {"$set": {"updated_at": now_iso}},
        )
        return record

    except Exception as e:
        logger.error("whatsapp: MongoDB error saving message (%s), falling back to memory", e)
        if external_message_id:
            _mem_processed_message_ids.add(external_message_id)
        _mem_messages.append(record)
        return record


async def get_conversation_history(
    conversation_id: str,
    limit: int = 6,
) -> List[Dict[str, str]]:
    """Retrieve recent conversation messages in chronological order for RAG multi-turn context."""
    db = await get_mongo_db()
    if db is None:
        matching = [m for m in _mem_messages if m["conversation_id"] == conversation_id]
        recent = matching[-limit:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    try:
        msgs_col = db["whatsapp_messages"]
        cursor = msgs_col.find({"conversation_id": conversation_id}).sort("created_at", DESCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        # Reverse to chronological order (oldest to newest)
        return [{"role": doc["role"], "content": doc["content"]} for doc in reversed(docs)]
    except Exception as e:
        logger.error("whatsapp: MongoDB error fetching conversation history (%s)", e)
        matching = [m for m in _mem_messages if m["conversation_id"] == conversation_id]
        recent = matching[-limit:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]

