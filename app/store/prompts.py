"""Prompt-record repository.

A prompt record is whatever a client app submitted, stored verbatim. This module
has no opinion about its contents and no library of its own to fall back on: if
a client has not registered a prompt and does not send one inline, the reply is
refused rather than answered with a house default.

Records are namespaced by client_id, so two applications may both use the
prompt_id "support" without colliding.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from ..schemas import PromptRecord
from .base import JsonStore, safe_name

_store = JsonStore(Path(settings.data_dir), "prompts")


def _key(client_id: str, prompt_id: str) -> str:
    return f"{safe_name(client_id)}__{safe_name(prompt_id)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def get(client_id: str, prompt_id: str) -> PromptRecord | None:
    record = await _store.read(_key(client_id, prompt_id))
    return PromptRecord.model_validate(record) if record else None


async def upsert(record: PromptRecord) -> PromptRecord:
    key = _key(record.client_id, record.prompt_id)
    existing = await _store.read(key)
    record.created_at = (existing or {}).get("created_at") or _now()
    record.updated_at = _now()
    await _store.write(key, record.model_dump())
    return record


async def delete(client_id: str, prompt_id: str) -> bool:
    return await _store.delete(_key(client_id, prompt_id))


async def list_for(client_id: str) -> list[str]:
    """Prompt ids this client has registered."""
    prefix = f"{safe_name(client_id)}__"
    return [key[len(prefix) :] for key in await _store.keys() if key.startswith(prefix)]
