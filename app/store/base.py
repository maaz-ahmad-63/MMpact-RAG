"""Tiny persistence layer for threads and prompt records.

The service had no database and no ORM, so this adds neither. Records are JSON
files under `DATA_DIR`, one per record, written atomically. That is enough for
what the agent actually needs to remember, and it keeps `requirements.txt`
unchanged.

Everything goes through `JsonStore`, so swapping in SQLite, Redis or Postgres
later means writing one new class - no caller changes.

Note on statelessness: the agent remains stateless *with respect to the client*.
It holds no prompt, persona or business config of its own; it only stores what a
client app explicitly wrote through the Thread and Prompt APIs, keyed by that
client's own `client_id`. Two client apps never share a record.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Filenames are derived from caller-supplied ids. Anything outside this set is
# percent-escaped, so a client_id of "../../etc" cannot escape the data dir.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(value: str) -> str:
    """Filesystem-safe encoding of an arbitrary id, reversible enough to debug."""
    text = str(value)
    if not text:
        raise ValueError("id must not be empty")
    return _SAFE.sub(lambda match: f"%{ord(match.group()):02x}", text)[:180]


class JsonStore:
    """One directory of JSON records, with an in-process write lock per key."""

    def __init__(self, root: Path, namespace: str) -> None:
        self._root = Path(root) / namespace
        self._locks: dict[str, asyncio.Lock] = {}
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._root / f"{safe_name(key)}.json"

    def _lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock

    async def read(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)

        def _read() -> dict[str, Any] | None:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except FileNotFoundError:
                return None
            except (OSError, ValueError) as error:
                logger.error("could not read %s: %s", path, error)
                return None

        return await asyncio.to_thread(_read)

    async def write(self, key: str, record: dict[str, Any]) -> dict[str, Any]:
        path = self._path(key)

        def _write() -> None:
            # Temp file in the same directory, then rename: a crash mid-write
            # leaves the previous record intact rather than a truncated one.
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self._root, delete=False, suffix=".tmp"
            )
            try:
                with handle:
                    json.dump(record, handle, ensure_ascii=False, default=str)
                os.replace(handle.name, path)
            except BaseException:
                Path(handle.name).unlink(missing_ok=True)
                raise

        async with self._lock(key):
            await asyncio.to_thread(_write)
        return record

    async def delete(self, key: str) -> bool:
        path = self._path(key)

        def _delete() -> bool:
            try:
                path.unlink()
                return True
            except FileNotFoundError:
                return False

        async with self._lock(key):
            return await asyncio.to_thread(_delete)

    async def keys(self) -> list[str]:
        def _keys() -> list[str]:
            return [item.stem for item in self._root.glob("*.json")]

        return await asyncio.to_thread(_keys)
