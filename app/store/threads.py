"""Thread repository.

A thread is the conversation state for one (client_id, channel, contact_id).
The id is derived from that triple, so a client that only knows "who am I
talking to" can resolve the same thread every time without tracking an id.

Isolation is structural: the client_id is part of the hash, so two applications
messaging the same phone number can never land on the same thread.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from ..schemas import HistoryTurn, Thread, ThreadRef
from .base import JsonStore

_store = JsonStore(Path(settings.data_dir), "threads")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def thread_id_for(ref: ThreadRef) -> str:
    """Deterministic id for a conversation triple.

    Hashed rather than concatenated so a contact_id containing separators cannot
    collide with a different triple, and so ids are a predictable length.
    """
    digest = hashlib.sha256(
        "\x00".join([ref.client_id, ref.channel, ref.contact_id]).encode("utf-8")
    ).hexdigest()
    return f"th_{digest[:32]}"


async def get(thread_id: str) -> Thread | None:
    record = await _store.read(thread_id)
    return Thread.model_validate(record) if record else None


async def get_or_create(ref: ThreadRef) -> Thread:
    thread_id = thread_id_for(ref)
    existing = await get(thread_id)
    if existing:
        return existing

    now = _now()
    thread = Thread(
        thread_id=thread_id,
        client_id=ref.client_id,
        channel=ref.channel,
        contact_id=ref.contact_id,
        created_at=now,
        updated_at=now,
    )
    await _store.write(thread_id, thread.model_dump())
    return thread


async def append(thread: Thread, turns: list[HistoryTurn]) -> Thread:
    """Add turns and trim to the retained window.

    Only the most recent `THREAD_MAX_TURNS` are kept. This is a working memory
    for the model, not an archive - the calling application keeps the durable
    message history in its own system.
    """
    thread.turns.extend(turns)
    if settings.thread_max_turns > 0:
        thread.turns = thread.turns[-settings.thread_max_turns :]
    thread.updated_at = _now()
    await _store.write(thread.thread_id, thread.model_dump())
    return thread


async def reset(thread_id: str) -> Thread | None:
    """Clear the turns but keep the thread and its metadata."""
    thread = await get(thread_id)
    if thread is None:
        return None
    thread.turns = []
    thread.updated_at = _now()
    await _store.write(thread_id, thread.model_dump())
    return thread


async def delete(thread_id: str) -> bool:
    return await _store.delete(thread_id)
