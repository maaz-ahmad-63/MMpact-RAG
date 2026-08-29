"""Trace persistence. The only module in this package that writes SQL.

It opens NO connection of its own: `rag/vector_store.get_pool()` already owns
the process's asyncpg pool against the same Postgres, and a second pool to the
same server would be two connection budgets for one service.

Every write is best-effort
--------------------------
`record_*` never raises. A trace is a report about a reply; a reply that
succeeded must not be turned into a 500 because its report could not be filed.
Failures are logged once, at WARNING, with the run id.

Reads are the opposite: the dashboard asking for a run that cannot be fetched
should say so, so those propagate.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..rag import vector_store
from ..rag.config import rag_settings

logger = logging.getLogger(__name__)

#: Applied in order. Each file is idempotent and additive, so re-running the
#: whole set is always safe - there is no version table to keep in step.
MIGRATION_FILES = sorted((Path(__file__).parent / "migrations").glob("*.sql"))


class TraceStoreUnavailable(Exception):
    """No database configured. Traces are logged instead; reads report this."""


def storage_configured() -> bool:
    return bool(rag_settings.database_configured)


def _schema() -> str:
    # Reuses the RAG validator, so RAG_DB_SCHEMA is checked the same way here as
    # it is for the vector tables and cannot mean two different things.
    return vector_store._schema(rag_settings)  # noqa: SLF001 - one validated identifier


def migration_sql() -> str:
    schema = _schema()
    return "\n\n".join(
        path.read_text(encoding="utf-8").replace("{{SCHEMA}}", schema)
        for path in MIGRATION_FILES
    )


async def migrate() -> None:
    """Create the observability tables. Idempotent and additive."""
    if not storage_configured():
        raise TraceStoreUnavailable(
            "RAG_DATABASE_URL (or DATABASE_URL) is not set, so there is no database "
            "to create the agent trace tables in."
        )
    pool = await vector_store.get_pool(rag_settings)
    schema = _schema()
    async with pool.acquire() as connection:
        # One statement per FILE, not one for all of them: a DO $$ block that
        # fails should name the migration it came from.
        for path in MIGRATION_FILES:
            await connection.execute(
                path.read_text(encoding="utf-8").replace("{{SCHEMA}}", schema)
            )
            logger.info("graph: applied %s", path.name)
    logger.info(
        "graph: trace schema ready (agent_runs, agent_steps, rag_searches, "
        "rag_results, llm_calls, tool_calls in schema %s)", _schema(),
    )


async def tables_present() -> bool:
    """Whether the migration has been run. Used by /health and the dashboard."""
    if not storage_configured():
        return False
    pool = await vector_store.get_pool(rag_settings)
    row = await pool.fetchrow(
        "SELECT to_regclass($1) IS NOT NULL AS present", f"{_schema()}.agent_runs"
    )
    return bool(row and row["present"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else (str(value) if value else None)


def _as_dict(value: Any) -> dict[str, Any]:
    """asyncpg returns JSONB as a string unless a codec is registered."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


# -- writes (best-effort) ---------------------------------------------------


async def _pool_or_none():
    if not storage_configured():
        return None
    try:
        return await vector_store.get_pool(rag_settings)
    except Exception as error:  # noqa: BLE001
        logger.warning("graph: trace storage unreachable (%s)", error)
        return None


async def start_run(
    *,
    run_id: str,
    conversation_id: str | None,
    user_id: str | None,
    client_id: str | None,
    user_message: str,
    model_identifier: str | None,
    model_provider: str | None,
    model_name: str | None,
) -> None:
    """Open the run row before the graph runs.

    Written up front rather than at the end so a run that crashes mid-graph -
    the case a developer most wants to see - still appears in the dashboard,
    with status 'running' until finish_run overwrites it.
    """
    pool = await _pool_or_none()
    if pool is None:
        return
    try:
        await pool.execute(
            f"""
            INSERT INTO {_schema()}.agent_runs
                (run_id, conversation_id, user_id, client_id, user_message,
                 model_identifier, model_provider, model_name, status, started_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'running',$9)
            ON CONFLICT (run_id) DO NOTHING
            """,
            run_id, conversation_id, user_id, client_id, user_message,
            model_identifier, model_provider, model_name, _now(),
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("graph: could not open run %s (%s)", run_id, error)


async def finish_run(
    *,
    run_id: str,
    status: str,
    intent: str | None = None,
    action: str | None = None,
    confidence: float | None = None,
    requires_rag: bool = False,
    search_query: str | None = None,
    final_answer: str | None = None,
    handoff_requested: bool = False,
    latency_ms: int | None = None,
    error: str | None = None,
    model_provider: str | None = None,
    model_name: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    pool = await _pool_or_none()
    if pool is None:
        return
    try:
        await pool.execute(
            f"""
            UPDATE {_schema()}.agent_runs SET
                status = $2, intent = $3, action = $4, confidence = $5,
                requires_rag = $6, search_query = $7, final_answer = $8,
                handoff_requested = $9, completed_at = $10, latency_ms = $11,
                error = $12,
                model_provider = COALESCE($13, model_provider),
                model_name = COALESCE($14, model_name),
                input_tokens = $15, output_tokens = $16
            WHERE run_id = $1
            """,
            run_id, status, intent, action, confidence, requires_rag, search_query,
            final_answer, handoff_requested, _now(), latency_ms, error,
            model_provider, model_name, int(input_tokens or 0), int(output_tokens or 0),
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("graph: could not close run %s (%s)", run_id, err)


async def record_step(
    *,
    run_id: str,
    step_number: int,
    node_name: str,
    step_type: str,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    latency_ms: int,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    pool = await _pool_or_none()
    if pool is None:
        return
    try:
        await pool.execute(
            f"""
            INSERT INTO {_schema()}.agent_steps
                (run_id, step_number, node_name, step_type, status,
                 started_at, completed_at, latency_ms, error, metadata)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb)
            ON CONFLICT (run_id, step_number) DO NOTHING
            """,
            run_id, step_number, node_name, step_type, status,
            started_at, completed_at, latency_ms, error,
            json.dumps(metadata or {}, ensure_ascii=False, default=str),
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("graph: could not record step %s/%s (%s)", run_id, node_name, err)


async def record_rag_search(
    *,
    run_id: str,
    query: str,
    top_k: int | None,
    threshold: float | None,
    results: Sequence[dict[str, Any]],
    latency_ms: int,
    rewritten: bool = False,
) -> None:
    """The search and its results, in one transaction so they cannot half-write."""
    pool = await _pool_or_none()
    if pool is None:
        return
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                search_id = await connection.fetchval(
                    f"""
                    INSERT INTO {_schema()}.rag_searches
                        (run_id, query, top_k, threshold, results_count, latency_ms, rewritten)
                    VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING id
                    """,
                    run_id, query, top_k, threshold, len(results), latency_ms, rewritten,
                )
                if results:
                    await connection.executemany(
                        f"""
                        INSERT INTO {_schema()}.rag_results
                            (search_id, run_id, document_id, chunk_id, score,
                             source, content, rank)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                        """,
                        [
                            (
                                search_id, run_id,
                                str(item.get("document_id") or ""),
                                str(item.get("chunk_id") or ""),
                                float(item.get("score") or 0.0),
                                (item.get("source") or None),
                                item.get("content"),
                                rank,
                            )
                            for rank, item in enumerate(results, start=1)
                        ],
                    )
    except Exception as error:  # noqa: BLE001
        logger.warning("graph: could not record rag search for %s (%s)", run_id, error)


async def record_llm_call(
    *,
    run_id: str,
    node_name: str,
    provider: str | None,
    model: str | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int | None = None,
    status: str = "ok",
    error: str | None = None,
) -> None:
    pool = await _pool_or_none()
    if pool is None:
        return
    try:
        await pool.execute(
            f"""
            INSERT INTO {_schema()}.llm_calls
                (run_id, node_name, provider, model, input_tokens, output_tokens,
                 latency_ms, status, error)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            """,
            run_id, node_name, provider, model, int(input_tokens), int(output_tokens),
            latency_ms, status, error,
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("graph: could not record llm call for %s (%s)", run_id, err)


async def record_tool_call(
    *,
    run_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: str | None,
    status: str = "ok",
    latency_ms: int | None = None,
    error: str | None = None,
) -> None:
    pool = await _pool_or_none()
    if pool is None:
        return
    try:
        await pool.execute(
            f"""
            INSERT INTO {_schema()}.tool_calls
                (run_id, tool_name, arguments, result, status, latency_ms, error)
            VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7)
            """,
            run_id, tool_name,
            json.dumps(arguments or {}, ensure_ascii=False, default=str),
            result, status, latency_ms, error,
        )
    except Exception as err:  # noqa: BLE001
        logger.warning("graph: could not record tool call for %s (%s)", run_id, err)


# -- reads (propagate) ------------------------------------------------------


async def _read_pool():
    if not storage_configured():
        raise TraceStoreUnavailable(
            "Agent tracing needs Postgres. Set RAG_DATABASE_URL (or DATABASE_URL) "
            "and run `python -m app.graph.migrate`. The agent itself keeps working "
            "without it; only the dashboard is affected."
        )
    return await vector_store.get_pool(rag_settings)


_RUN_COLUMNS = """
    run_id, conversation_id, user_id, client_id, user_message, intent, action,
    confidence, requires_rag, search_query, status, model_provider, model_name,
    model_identifier, final_answer, handoff_requested, started_at, completed_at,
    latency_ms, error, input_tokens, output_tokens, total_tokens
"""


def _run_row(row: Any) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "conversation_id": row["conversation_id"],
        "user_id": row["user_id"],
        "client_id": row["client_id"],
        "user_message": row["user_message"],
        "intent": row["intent"],
        "action": row["action"],
        "confidence": row["confidence"],
        "used_rag": bool(row["requires_rag"]),
        "status": row["status"],
        "model_provider": row["model_provider"],
        "model_name": row["model_name"],
        "latency_ms": row["latency_ms"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "total_tokens": row["total_tokens"],
        "started_at": _iso(row["started_at"]),
        "completed_at": _iso(row["completed_at"]),
        "error": row["error"],
    }


async def list_runs(
    *,
    limit: int = 50,
    offset: int = 0,
    conversation_id: str | None = None,
    user_id: str | None = None,
    status: str | None = None,
    action: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    pool = await _read_pool()
    filters: list[str] = []
    parameters: list[Any] = []
    for column, value in (
        ("conversation_id", conversation_id),
        ("user_id", user_id),
        ("status", status),
        ("action", action),
    ):
        if value:
            parameters.append(value)
            filters.append(f"{column} = ${len(parameters)}")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    total = await pool.fetchval(
        f"SELECT count(*) FROM {_schema()}.agent_runs {where}", *parameters
    )
    parameters.extend([int(limit), int(offset)])
    rows = await pool.fetch(
        f"""
        SELECT {_RUN_COLUMNS} FROM {_schema()}.agent_runs {where}
        ORDER BY started_at DESC
        LIMIT ${len(parameters) - 1} OFFSET ${len(parameters)}
        """,
        *parameters,
    )
    return [_run_row(row) for row in rows], int(total or 0)


async def get_run(run_id: str) -> dict[str, Any] | None:
    """One run with every child row, for the run-detail page."""
    pool = await _read_pool()
    schema = _schema()

    row = await pool.fetchrow(
        f"SELECT {_RUN_COLUMNS} FROM {schema}.agent_runs WHERE run_id = $1", run_id
    )
    if row is None:
        return None

    steps = await pool.fetch(
        f"""SELECT step_number, node_name, step_type, status, latency_ms,
                   started_at, completed_at, error, metadata
            FROM {schema}.agent_steps WHERE run_id = $1 ORDER BY step_number""",
        run_id,
    )
    searches = await pool.fetch(
        f"""SELECT id, query, top_k, threshold, results_count, latency_ms, rewritten
            FROM {schema}.rag_searches WHERE run_id = $1 ORDER BY id""",
        run_id,
    )
    results = await pool.fetch(
        f"""SELECT search_id, document_id, chunk_id, score, source, content
            FROM {schema}.rag_results WHERE run_id = $1 ORDER BY search_id, rank""",
        run_id,
    )
    llm = await pool.fetch(
        f"""SELECT node_name, provider, model, input_tokens, output_tokens,
                   latency_ms, status, error
            FROM {schema}.llm_calls WHERE run_id = $1 ORDER BY id""",
        run_id,
    )
    tools = await pool.fetch(
        f"""SELECT tool_name, arguments, result, status, latency_ms, error
            FROM {schema}.tool_calls WHERE run_id = $1 ORDER BY id""",
        run_id,
    )

    by_search: dict[Any, list[dict[str, Any]]] = {}
    for item in results:
        by_search.setdefault(item["search_id"], []).append(
            {
                "document_id": item["document_id"],
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "source": item["source"],
                "content": item["content"],
            }
        )

    return {
        "run": _run_row(row),
        "final_answer": row["final_answer"],
        "steps": [
            {
                "step_number": step["step_number"],
                "node_name": step["node_name"],
                "step_type": step["step_type"],
                "status": step["status"],
                "latency_ms": step["latency_ms"],
                "started_at": _iso(step["started_at"]),
                "completed_at": _iso(step["completed_at"]),
                "error": step["error"],
                "metadata": _as_dict(step["metadata"]),
            }
            for step in steps
        ],
        "rag_searches": [
            {
                "query": search["query"],
                "top_k": search["top_k"],
                "threshold": search["threshold"],
                "results_count": search["results_count"],
                "latency_ms": search["latency_ms"],
                "rewritten": search["rewritten"],
                "results": by_search.get(search["id"], []),
            }
            for search in searches
        ],
        "llm_calls": [dict(call) for call in llm],
        "tool_calls": [
            {
                "tool_name": call["tool_name"],
                "arguments": _as_dict(call["arguments"]),
                "result": call["result"],
                "status": call["status"],
                "latency_ms": call["latency_ms"],
                "error": call["error"],
            }
            for call in tools
        ],
    }


async def overview(window_hours: int = 24) -> dict[str, Any]:
    """The counters on the dashboard's front page, computed in one statement."""
    pool = await _read_pool()
    row = await pool.fetchrow(
        f"""
        SELECT
            count(*)                                                    AS total,
            count(*) FILTER (WHERE status = 'completed')                AS successful,
            count(*) FILTER (WHERE status = 'failed')                   AS failed,
            count(*) FILTER (WHERE action = 'rag_search')               AS rag_requests,
            count(*) FILTER (WHERE action = 'direct_answer')            AS direct_answers,
            count(*) FILTER (WHERE action = 'tool')                     AS tool_calls,
            count(*) FILTER (WHERE action = 'human_handoff')            AS handoffs,
            COALESCE(avg(latency_ms) FILTER (WHERE latency_ms IS NOT NULL), 0) AS avg_latency,
            COALESCE(sum(input_tokens), 0)  AS input_tokens,
            COALESCE(sum(output_tokens), 0) AS output_tokens,
            COALESCE(sum(total_tokens), 0)  AS total_tokens
        FROM {_schema()}.agent_runs
        WHERE started_at > now() - ($1::int * INTERVAL '1 hour')
        """,
        int(window_hours),
    )
    return {
        "total_requests": int(row["total"] or 0),
        "successful": int(row["successful"] or 0),
        "failed": int(row["failed"] or 0),
        "rag_requests": int(row["rag_requests"] or 0),
        "direct_answers": int(row["direct_answers"] or 0),
        "tool_calls": int(row["tool_calls"] or 0),
        "human_handoffs": int(row["handoffs"] or 0),
        "average_latency_ms": int(row["avg_latency"] or 0),
        "input_tokens": int(row["input_tokens"] or 0),
        "output_tokens": int(row["output_tokens"] or 0),
        "total_tokens": int(row["total_tokens"] or 0),
        "window_hours": int(window_hours),
    }


async def list_errors(*, limit: int = 100, window_hours: int = 168) -> list[dict[str, Any]]:
    """Failures across runs, steps and LLM calls, newest first.

    Three sources rather than one: a run can fail without any step recording an
    error (a crash between nodes), and an LLM call can fail on a node that later
    recovered, which is exactly the case a developer needs to see.
    """
    pool = await _read_pool()
    schema = _schema()
    rows = await pool.fetch(
        f"""
        SELECT started_at AS at, run_id, 'run' AS node, model_provider AS provider,
               model_name AS model, error, latency_ms
          FROM {schema}.agent_runs
         WHERE error IS NOT NULL AND started_at > now() - ($2::int * INTERVAL '1 hour')
        UNION ALL
        SELECT s.started_at, s.run_id, s.node_name, r.model_provider, r.model_name,
               s.error, s.latency_ms
          FROM {schema}.agent_steps s
          JOIN {schema}.agent_runs r ON r.run_id = s.run_id
         WHERE s.error IS NOT NULL AND s.started_at > now() - ($2::int * INTERVAL '1 hour')
        UNION ALL
        SELECT c.created_at, c.run_id, c.node_name, c.provider, c.model,
               c.error, c.latency_ms
          FROM {schema}.llm_calls c
         WHERE c.error IS NOT NULL AND c.created_at > now() - ($2::int * INTERVAL '1 hour')
        ORDER BY at DESC
        LIMIT $1
        """,
        int(limit), int(window_hours),
    )
    return [
        {
            "at": _iso(row["at"]),
            "run_id": row["run_id"],
            "node": row["node"],
            "provider": row["provider"],
            "model": row["model"],
            "error": row["error"],
            "latency_ms": row["latency_ms"],
        }
        for row in rows
    ]


async def model_usage(window_hours: int = 168) -> list[dict[str, Any]]:
    """Which models actually ran. Proves the Laravel-driven model is dynamic."""
    pool = await _read_pool()
    rows = await pool.fetch(
        f"""
        SELECT provider, model, count(*) AS calls,
               COALESCE(sum(input_tokens), 0)  AS input_tokens,
               COALESCE(sum(output_tokens), 0) AS output_tokens,
               COALESCE(avg(latency_ms), 0)    AS avg_latency,
               count(*) FILTER (WHERE status <> 'ok') AS failures
          FROM {_schema()}.llm_calls
         WHERE created_at > now() - ($1::int * INTERVAL '1 hour')
         GROUP BY provider, model
         ORDER BY calls DESC
        """,
        int(window_hours),
    )
    return [
        {
            "provider": row["provider"],
            "model": row["model"],
            "calls": int(row["calls"]),
            "input_tokens": int(row["input_tokens"]),
            "output_tokens": int(row["output_tokens"]),
            "average_latency_ms": int(row["avg_latency"] or 0),
            "failures": int(row["failures"]),
        }
        for row in rows
    ]


async def rag_activity(*, limit: int = 100, window_hours: int = 168) -> list[dict[str, Any]]:
    pool = await _read_pool()
    rows = await pool.fetch(
        f"""
        SELECT run_id, query, top_k, threshold, results_count, latency_ms,
               rewritten, created_at
          FROM {_schema()}.rag_searches
         WHERE created_at > now() - ($2::int * INTERVAL '1 hour')
         ORDER BY created_at DESC LIMIT $1
        """,
        int(limit), int(window_hours),
    )
    return [
        {
            "run_id": row["run_id"],
            "query": row["query"],
            "top_k": row["top_k"],
            "threshold": row["threshold"],
            "results_count": row["results_count"],
            "latency_ms": row["latency_ms"],
            "rewritten": row["rewritten"],
            "at": _iso(row["created_at"]),
        }
        for row in rows
    ]


async def tool_activity(*, limit: int = 100, window_hours: int = 168) -> list[dict[str, Any]]:
    pool = await _read_pool()
    rows = await pool.fetch(
        f"""
        SELECT run_id, tool_name, arguments, result, status, latency_ms, error, created_at
          FROM {_schema()}.tool_calls
         WHERE created_at > now() - ($2::int * INTERVAL '1 hour')
         ORDER BY created_at DESC LIMIT $1
        """,
        int(limit), int(window_hours),
    )
    return [
        {
            "run_id": row["run_id"],
            "tool_name": row["tool_name"],
            "arguments": _as_dict(row["arguments"]),
            "result": row["result"],
            "status": row["status"],
            "latency_ms": row["latency_ms"],
            "error": row["error"],
            "at": _iso(row["created_at"]),
        }
        for row in rows
    ]
