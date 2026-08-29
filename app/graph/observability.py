"""Run ids, step timing and redaction.

The tracer is the ONLY thing nodes talk to about observability - they never
touch graph/store.py directly. That keeps one rule in one place: what gets
written is operational, redacted, and never able to break a reply.

WHAT IS NEVER RECORDED
----------------------
  * Chain of thought. No step passes model deliberation to `step()` - the
    metadata is tool names, latencies, counts and statuses. This is a design
    property, not a filter applied afterwards.
  * Secrets. `redact()` removes anything whose key looks like a credential
    before it reaches JSONB, and long values are truncated.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

from . import store

logger = logging.getLogger(__name__)

#: Crockford base32, as ULIDs use. Time-ordered ids sort by creation, which is
#: what a run list wants, and they carry no PII the way a hash of the message
#: might.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

#: Keys that CONTAIN a secret-looking word but are plainly operational. Checked
#: first, because the broad pattern below would otherwise redact token COUNTS -
#: which are among the few numbers this whole feature exists to show.
_SAFE_KEY = re.compile(
    r"^(input_tokens|output_tokens|total_tokens|max_tokens|tokens_used|"
    r"cache_read_input_tokens|cache_creation_input_tokens|token_count)$",
    re.IGNORECASE,
)

#: Keys whose VALUE must never be written, matched case-insensitively anywhere
#: in the key. Deliberately broad: a false positive costs one redacted field in
#: a debugging view, a false negative writes a credential to a database.
_SECRET_KEY = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|credential|authorization|auth|"
    r"bearer|cookie|session|private[_-]?key|dsn|connection[_-]?string)",
    re.IGNORECASE,
)

REDACTED = "[redacted]"

#: Cap on any single string written to a trace row. A retrieved chunk is
#: recorded in full by graph/store.record_rag_search (that is the point of it);
#: this bounds the incidental strings in `metadata` and tool arguments.
MAX_VALUE_CHARS = 2000


def new_run_id() -> str:
    """A sortable, unique run id: run_ + 26 Crockford-base32 characters.

    ULID-shaped, without taking a dependency for 20 lines. 48 bits of
    millisecond timestamp then 80 bits of randomness, so ids created in the same
    millisecond still differ and lexical order is chronological order.
    """
    timestamp = int(time.time() * 1000)
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (timestamp << 80) | randomness

    characters = []
    for _ in range(26):
        characters.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "run_" + "".join(reversed(characters))


def redact(value: Any, depth: int = 0) -> Any:
    """A structure safe to persist: secrets removed, long strings truncated."""
    if depth > 6:
        return "[deep]"
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if not _SAFE_KEY.match(str(key)) and _SECRET_KEY.search(str(key))
                else redact(item, depth + 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item, depth + 1) for item in list(value)[:50]]
    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
        return f"{value[:MAX_VALUE_CHARS]}…[{len(value)} chars]"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:MAX_VALUE_CHARS]


class Tracer:
    """One run's trace. Created by the runner, passed to every node.

    Not a context manager per node by accident: `step()` is, so a node that
    raises still writes a step row with its error and duration before the
    exception continues on to the runner.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._step_number = 0
        #: Mirrors the persisted steps, so a deployment without Postgres still
        #: returns a live trace on the response and the dashboard can degrade to
        #: "not stored" rather than "nothing happened".
        self.steps: list[dict[str, Any]] = []

        # -- usage accounting for ONE run ----------------------------------
        #
        # One run makes SEVERAL provider calls: the model's decide-and-call
        # turn, then its answer turn once tool results come back. Reporting only
        # the last would ignore the first entirely.
        #
        # dm_ai_logs.prompt_tokens/completion_tokens/total_tokens must be what
        # the run actually COST, or the billing view under-reports every
        # retrieval question by the size of two extra calls. Summed here,
        # because the tracer is the one object every LLM call passes through.
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_ms = 0
        self.provider_calls = 0
        #: Per-node breakdown, for dm_ai_logs.metadata and the dashboard. This
        #: is what makes an expensive run attributable to the node that made it
        #: expensive rather than to the run as a whole.
        self.usage_by_node: list[dict[str, Any]] = []

    def _next(self) -> int:
        self._step_number += 1
        return self._step_number

    def step(self, node_name: str, step_type: str = "node") -> "_Step":
        return _Step(self, node_name, step_type)

    async def llm_call(
        self,
        *,
        node_name: str,
        provider: str | None,
        model: str | None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        self.llm_ms += int(latency_ms or 0)
        self.provider_calls += 1
        self.usage_by_node.append(
            {
                "node": node_name,
                "provider": provider,
                "model": model,
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "latency_ms": latency_ms,
                "status": status,
            }
        )

        logger.info(
            "graph[%s] llm %s | provider=%s model=%s tokens=%s/%s %sms %s",
            self.run_id, node_name, provider, model, input_tokens, output_tokens,
            latency_ms, status,
        )
        await store.record_llm_call(
            run_id=self.run_id, node_name=node_name, provider=provider, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=latency_ms, status=status, error=error,
        )

    async def rag_search(
        self,
        *,
        query: str,
        top_k: int | None,
        threshold: float | None,
        results: list[dict[str, Any]],
        latency_ms: int,
        rewritten: bool = False,
    ) -> None:
        logger.info(
            "graph[%s] rag search | q=%r top_k=%s -> %s results in %sms%s",
            self.run_id, query[:120], top_k, len(results), latency_ms,
            " (after rewrite)" if rewritten else "",
        )
        await store.record_rag_search(
            run_id=self.run_id, query=query, top_k=top_k, threshold=threshold,
            results=results, latency_ms=latency_ms, rewritten=rewritten,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    #: 'provider' when any call reported real numbers, 'unavailable' when none
    #: did. Mirrors LLMTrace.usage_source so the gateway reads one vocabulary.
    @property
    def usage_source(self) -> str:
        return "provider" if self.total_tokens else "unavailable"

    async def tool_call(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        result: str | None,
        status: str = "ok",
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        safe_arguments = redact(arguments or {})
        logger.info(
            "graph[%s] tool %s | args=%s %sms %s",
            self.run_id, tool_name, safe_arguments, latency_ms, status,
        )
        await store.record_tool_call(
            run_id=self.run_id, tool_name=tool_name, arguments=safe_arguments,
            result=(redact(result) if isinstance(result, str) else result),
            status=status, latency_ms=latency_ms, error=error,
        )


class _Step:
    """Times one node and records it, whether it succeeds or raises."""

    def __init__(self, tracer: Tracer, node_name: str, step_type: str) -> None:
        self._tracer = tracer
        self._node = node_name
        self._type = step_type
        self._metadata: dict[str, Any] = {}
        self._started_at = datetime.now(timezone.utc)
        self._started = 0.0

    def note(self, **fields: Any) -> None:
        """Attach operational detail to this step. Redacted before it is written."""
        self._metadata.update(fields)

    async def __aenter__(self) -> "_Step":
        self._started = time.perf_counter()
        logger.info("graph[%s] -> %s", self._tracer.run_id, self._node)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        latency_ms = int((time.perf_counter() - self._started) * 1000)
        completed_at = datetime.now(timezone.utc)
        status = "ok" if exc is None else "error"
        error = None if exc is None else f"{type(exc).__name__}: {exc}"
        metadata = redact(self._metadata)

        record = {
            "step_number": self._tracer._next(),  # noqa: SLF001 - same module
            "node_name": self._node,
            "step_type": self._type,
            "status": status,
            "latency_ms": latency_ms,
            "started_at": self._started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "error": error,
            "metadata": metadata,
        }
        self._tracer.steps.append(record)

        if exc is not None:
            logger.warning("graph[%s] x %s failed: %s", self._tracer.run_id, self._node, error)

        await store.record_step(
            run_id=self._tracer.run_id,
            step_number=record["step_number"],
            node_name=self._node,
            step_type=self._type,
            status=status,
            started_at=self._started_at,
            completed_at=completed_at,
            latency_ms=latency_ms,
            error=error,
            metadata=metadata,
        )
        # Never swallow: the runner decides what a failed node means.
        return False
