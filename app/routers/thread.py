"""Thread API - conversation state for one contact on one channel.

Threads are keyed by (client_id, channel, contact_id), so two applications
messaging the same phone number keep entirely separate conversations. Every
route requires the client_id and checks it against the stored record: no client
can read or clear another client's thread.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..schemas import Thread, ThreadRef
from ..security import require_token
from ..store import threads as thread_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/thread", tags=["thread"], dependencies=[Depends(require_token)])


async def _owned(thread_id: str, client_id: str) -> Thread:
    thread = await thread_store.get(thread_id)
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No thread {thread_id}"
        )
    if thread.client_id != client_id:
        # 404 rather than 403: whether another client's thread exists is not
        # something a caller should be able to probe for.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"No thread {thread_id}"
        )
    return thread


@router.post("", response_model=Thread, status_code=status.HTTP_200_OK)
async def create_or_get(ref: ThreadRef) -> Thread:
    """Create the thread, or return the existing one.

    Request:  { "client_id": "laravel-prod", "channel": "whatsapp", "contact_id": "923001234567" }
    Response: the full thread, including its turns.

    Idempotent - the id is derived from the three fields, so calling this twice
    returns the same thread rather than a second one.
    """
    return await thread_store.get_or_create(ref)


@router.get("/{thread_id}", response_model=Thread)
async def fetch(
    thread_id: str,
    client_id: str = Query(description="The client that owns this thread."),
) -> Thread:
    """History and state for a thread."""
    return await _owned(thread_id, client_id)


@router.post("/{thread_id}/reset", response_model=Thread)
async def reset(
    thread_id: str,
    client_id: str = Query(description="The client that owns this thread."),
) -> Thread:
    """Clear the turns, keeping the thread and its metadata."""
    await _owned(thread_id, client_id)
    thread = await thread_store.reset(thread_id)
    if thread is None:  # deleted between the two calls
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No thread {thread_id}")
    logger.info("thread %s reset by client %s", thread_id, client_id)
    return thread


@router.delete("/{thread_id}")
async def delete(
    thread_id: str,
    client_id: str = Query(description="The client that owns this thread."),
) -> dict[str, object]:
    """Remove the thread entirely."""
    await _owned(thread_id, client_id)
    deleted = await thread_store.delete(thread_id)
    logger.info("thread %s deleted by client %s", thread_id, client_id)
    return {"ok": True, "thread_id": thread_id, "deleted": deleted}
