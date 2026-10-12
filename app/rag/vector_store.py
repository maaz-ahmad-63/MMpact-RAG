"""Postgres + pgvector: the only module in the package that writes SQL.

Connection ownership
--------------------
This service had no database before RAG existed (app/store/base.py: threads and
prompt records are JSON files). So this opens its own asyncpg pool from
RAG_DATABASE_URL / DATABASE_URL and owns it alone. Nothing else in the process
uses it, and when the DSN is unset the pool is never created - the RAG routes
answer 503 and every existing endpoint is untouched.

Similarity happens IN THE DATABASE
----------------------------------
`search` sends the query vector to Postgres and orders by the pgvector distance
operator with a LIMIT. Chunks are never streamed into Python to be compared
here: that would move the whole table across the socket on every query and throw
away the index. `_distance_expression` picks the operator from
RAG_DISTANCE_METRIC, and the migration builds the index with the matching
opclass, so the query the planner sees always has an index it can use.

asyncpg has no vector codec, so vectors cross as the literal text pgvector
parses - '[0.1,0.2,...]' - with an explicit ::vector cast. That is a bound
parameter, not string interpolation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import RagSettings, rag_settings

logger = logging.getLogger(__name__)

MIGRATION_FILE = Path(__file__).parent / "migrations" / "001_rag_tables.sql"

# Identifiers cannot be bound as parameters, so the one identifier that comes
# from configuration is constrained to what an unquoted identifier may contain.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class VectorStoreUnavailable(Exception):
    """No DSN configured, or the driver is missing. -> HTTP 503."""


class VectorStoreError(Exception):
    """The database rejected or failed the operation. -> HTTP 500."""


@dataclass
class SearchHit:
    chunk_id: str
    document_id: str
    content: str
    similarity: float
    distance: float
    chunk_index: int
    metadata: dict[str, Any]
    filename: str
    file_type: str
    vector_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


_pool = None  # asyncpg.Pool


def _schema(settings: RagSettings) -> str:
    name = settings.rag_db_schema.strip()
    if not _IDENTIFIER.match(name):
        raise VectorStoreError(
            f"RAG_DB_SCHEMA must be a plain identifier, got {name!r}"
        )
    return name


def to_vector_literal(vector: Sequence[float]) -> str:
    """A float sequence in the text form pgvector's input function accepts."""
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


async def get_pool(settings: RagSettings | None = None):
    """The process-wide asyncpg pool, opened on first use."""
    global _pool
    config = settings or rag_settings
    if _pool is not None:
        try:
            if hasattr(_pool, "_loop") and _pool._loop.is_closed():
                _pool = None
            else:
                return _pool
        except Exception:
            _pool = None

    if not config.database_configured:
        raise VectorStoreUnavailable(
            "RAG storage is not configured. Set RAG_DATABASE_URL (or DATABASE_URL) "
            "to a Postgres instance with the pgvector extension available. "
            "Until then only the RAG routes are affected."
        )

    try:
        import asyncpg
    except ImportError as error:  # pragma: no cover - depends on the install
        raise VectorStoreUnavailable(
            "RAG storage needs `asyncpg` (pip install asyncpg)"
        ) from error

    try:
        _pool = await asyncpg.create_pool(
            dsn=config.dsn,
            min_size=config.rag_db_pool_min,
            max_size=config.rag_db_pool_max,
            command_timeout=config.rag_db_timeout_s,
        )
    except Exception as error:  # noqa: BLE001 - DSN carries a password; see below
        # The exception text is logged, not the DSN: a bad DSN string in a log is
        # a leaked password.
        logger.error("rag: could not connect to Postgres (%s)", error)
        raise VectorStoreUnavailable(f"Could not connect to Postgres: {error}") from error

    logger.info(
        "rag: Postgres pool ready (schema=%s, pool=%s-%s)",
        config.rag_db_schema, config.rag_db_pool_min, config.rag_db_pool_max,
    )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# -- schema -----------------------------------------------------------------


def migration_sql(settings: RagSettings | None = None) -> str:
    """The migration with its placeholders filled in."""
    config = settings or rag_settings
    return (
        MIGRATION_FILE.read_text(encoding="utf-8")
        .replace("{{SCHEMA}}", _schema(config))
        .replace("{{DIM}}", str(int(config.rag_embedding_dim)))
        .replace("{{OPCLASS}}", config.index_opclass)
    )


async def migrate(settings: RagSettings | None = None) -> None:
    """Create the RAG tables and the vector index. Idempotent, additive."""
    config = settings or rag_settings
    pool = await get_pool(config)
    try:
        async with pool.acquire() as connection:
            await connection.execute(migration_sql(config))
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"Migration failed: {error}") from error
    logger.info(
        "rag: schema ready (rag_documents, rag_chunks, vector(%s), %s)",
        config.rag_embedding_dim, config.rag_distance_metric,
    )


async def column_dimension(settings: RagSettings | None = None) -> int | None:
    """The declared width of rag_chunks.embedding, or None if the table is absent.

    Read from the catalogue rather than trusted from configuration, so a column
    created under an older RAG_EMBEDDING_DIM is caught before any insert.
    """
    config = settings or rag_settings
    pool = await get_pool(config)
    row = await pool.fetchrow(
        """
        SELECT atttypmod
        FROM pg_attribute
        WHERE attrelid = to_regclass($1) AND attname = 'embedding' AND NOT attisdropped
        """,
        f"{_schema(config)}.rag_chunks",
    )
    if row is None or row["atttypmod"] is None or row["atttypmod"] < 0:
        return None
    return int(row["atttypmod"])


# -- writes -----------------------------------------------------------------


async def insert_document(
    *,
    document_id: str,
    filename: str,
    file_type: str,
    file_path: str | None,
    user_id: str | None,
    agent_id: str | None,
    conversation_id: str | None,
    metadata: dict[str, Any],
    settings: RagSettings | None = None,
) -> dict[str, Any]:
    config = settings or rag_settings
    pool = await get_pool(config)
    schema = _schema(config)
    try:
        row = await pool.fetchrow(
            f"""
            INSERT INTO {schema}.rag_documents
                (id, filename, file_type, file_path, user_id, agent_id,
                 conversation_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            RETURNING id, filename, file_type, file_path, user_id, agent_id,
                      conversation_id, metadata, chunk_count, created_at, updated_at
            """,
            document_id, filename, file_type, file_path, user_id, agent_id,
            conversation_id, json.dumps(metadata or {}),
        )
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"Could not store the document record: {error}") from error
    return _document_row(row)


async def insert_chunks(
    document_id: str,
    rows: Iterable[tuple[int, str, Sequence[float], dict[str, Any]]],
    *,
    settings: RagSettings | None = None,
) -> int:
    """Bulk-insert (chunk_index, content, embedding, metadata) for one document.

    One COPY for the whole document, inside the transaction that also updates
    chunk_count. Row-at-a-time inserts are what make ingestion slow: a thousand
    chunks is a thousand round-trips, each one a network wait.
    """
    config = settings or rag_settings
    pool = await get_pool(config)
    schema = _schema(config)

    records = [
        (document_id, index, content, to_vector_literal(vector), json.dumps(meta or {}))
        for index, content, vector, meta in rows
    ]
    if not records:
        return 0

    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                # copy_records_to_table cannot cast, and the target columns are
                # vector and jsonb. A staging TEMP TABLE of text takes the COPY,
                # then one INSERT ... SELECT casts the whole set server-side:
                # still one round-trip for the data, and the types stay strict.
                await connection.execute(
                    """
                    CREATE TEMP TABLE _rag_chunk_stage (
                        document_id TEXT, chunk_index INT, content TEXT,
                        embedding TEXT, metadata TEXT
                    ) ON COMMIT DROP
                    """
                )
                await connection.copy_records_to_table(
                    "_rag_chunk_stage",
                    records=records,
                    columns=["document_id", "chunk_index", "content", "embedding", "metadata"],
                )
                await connection.execute(
                    f"""
                    INSERT INTO {schema}.rag_chunks
                        (document_id, chunk_index, content, embedding, metadata)
                    SELECT document_id, chunk_index, content,
                           embedding::vector, metadata::jsonb
                    FROM _rag_chunk_stage
                    """
                )
                await connection.execute(
                    f"""
                    UPDATE {schema}.rag_documents
                    SET chunk_count = (
                            SELECT count(*) FROM {schema}.rag_chunks WHERE document_id = $1
                        ),
                        updated_at = now()
                    WHERE id = $1
                    """,
                    document_id,
                )
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"Could not store the chunks: {error}") from error

    return len(records)


async def delete_document(
    document_id: str, *, settings: RagSettings | None = None
) -> tuple[bool, int]:
    """Delete a document and its chunks in ONE transaction.

    Returns (existed, chunks_deleted). Chunks are deleted explicitly so the count
    can be reported, and the FK's ON DELETE CASCADE remains as the guarantee that
    no path can leave an orphan.
    """
    config = settings or rag_settings
    pool = await get_pool(config)
    schema = _schema(config)
    try:
        async with pool.acquire() as connection:
            async with connection.transaction():
                chunks = await connection.fetchval(
                    f"WITH gone AS (DELETE FROM {schema}.rag_chunks WHERE document_id = $1 RETURNING 1) "
                    f"SELECT count(*) FROM gone",
                    document_id,
                )
                document = await connection.fetchval(
                    f"DELETE FROM {schema}.rag_documents WHERE id = $1 RETURNING id",
                    document_id,
                )
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"Could not delete the document: {error}") from error
    return bool(document), int(chunks or 0)


# -- reads ------------------------------------------------------------------


def _distance_expression(config: RagSettings) -> str:
    return f"c.embedding {config.distance_operator} $1::vector"


def _similarity_expression(config: RagSettings) -> str:
    """SQL turning the metric's distance into a comparable similarity.

    cosine        - distance is 1 - cosine_similarity, so similarity is 1 - d.
    inner_product - pgvector returns the NEGATIVE inner product, so similarity
                    is -d (equal to cosine similarity for normalised vectors).
    l2            - an unbounded distance; mapped to 1/(1+d) so a threshold in
                    (0, 1] still means "closer is higher". Note that a threshold
                    tuned for cosine does NOT transfer to this scale.
    """
    distance = _distance_expression(config)
    if config.rag_distance_metric == "cosine":
        return f"1 - ({distance})"
    if config.rag_distance_metric == "inner_product":
        return f"-({distance})"
    return f"1 / (1 + ({distance}))"


async def search(
    embedding: Sequence[float],
    *,
    top_k: int,
    tenant_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    document_ids: Sequence[str] | None = None,
    settings: RagSettings | None = None,
) -> list[SearchHit]:
    """Top-k nearest chunks, ranked by Postgres."""
    config = settings or rag_settings
    pool = await get_pool(config)
    schema = _schema(config)

    parameters: list[Any] = [to_vector_literal(embedding)]
    filters: list[str] = []
    if tenant_id:
        parameters.append(tenant_id)
        filters.append(f"(d.user_id = ${len(parameters)} OR d.metadata->>'tenant_id' = ${len(parameters)})")
    # Optional scoping. Each filter is a bound parameter; none is interpolated.
    for column, value in (
        ("user_id", user_id),
        ("agent_id", agent_id),
        ("conversation_id", conversation_id),
    ):
        if value:
            parameters.append(value)
            filters.append(f"d.{column} = ${len(parameters)}")
    if document_ids:
        parameters.append(list(document_ids))
        filters.append(f"d.id = ANY(${len(parameters)}::text[])")

    parameters.append(int(top_k))
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    query = f"""
        SELECT c.id, c.document_id, c.chunk_index, c.content, c.metadata,
               d.filename, d.file_type,
               ({_distance_expression(config)}) AS distance,
               ({_similarity_expression(config)}) AS similarity
        FROM {schema}.rag_chunks c
        JOIN {schema}.rag_documents d ON d.id = c.document_id
        {where}
        ORDER BY {_distance_expression(config)}
        LIMIT ${len(parameters)}
    """

    started = time.perf_counter()
    try:
        rows = await pool.fetch(query, *parameters)
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"Vector search failed: {error}") from error

    logger.info(
        "rag: vector search completed - %s rows in %.0f ms (metric=%s, top_k=%s)",
        len(rows), (time.perf_counter() - started) * 1000,
        config.rag_distance_metric, top_k,
    )

    return [
        SearchHit(
            chunk_id=str(row["id"]),
            document_id=row["document_id"],
            content=row["content"],
            similarity=float(row["similarity"]),
            distance=float(row["distance"]),
            chunk_index=int(row["chunk_index"]),
            metadata=_as_dict(row["metadata"]),
            filename=row["filename"],
            file_type=row["file_type"],
        )
        for row in rows
    ]


async def hybrid_search(
    embedding: Sequence[float],
    query_text: str,
    *,
    vector_candidate_count: int,
    keyword_candidate_count: int,
    tenant_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    document_ids: Sequence[str] | None = None,
    settings: RagSettings | None = None,
    **kwargs: Any,
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Retrieve candidates from both vector search and keyword search.

    Returns a tuple: (vector_hits, keyword_hits)
    """
    config = settings or rag_settings
    pool = await get_pool(config)
    schema = _schema(config)

    # Base parameters: $1 is the embedding, $2 is the query text
    base_parameters: list[Any] = [to_vector_literal(embedding), query_text]
    filters: list[str] = []

    if tenant_id:
        base_parameters.append(tenant_id)
        filters.append(f"(d.user_id = ${len(base_parameters)} OR d.metadata->>'tenant_id' = ${len(base_parameters)})")

    for column, value in (
        ("user_id", user_id),
        ("agent_id", agent_id),
        ("conversation_id", conversation_id),
    ):
        if value:
            base_parameters.append(value)
            filters.append(f"d.{column} = ${len(base_parameters)}")

    if document_ids:
        base_parameters.append(list(document_ids))
        filters.append(f"d.id = ANY(${len(base_parameters)}::text[])")

    # Distinct parameters for each candidate query limit
    vector_parameters = list(base_parameters)
    vector_parameters.append(int(vector_candidate_count))
    vector_limit_param = f"${len(vector_parameters)}"

    keyword_parameters = list(base_parameters)
    keyword_parameters.append(int(keyword_candidate_count))
    keyword_limit_param = f"${len(keyword_parameters)}"

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    query_vector = f"""
        WITH q AS (
            SELECT coalesce(nullif(string_agg(lexeme, ' | '), ''), '') AS val
            FROM unnest(to_tsvector('english', $2::text))
        )
        SELECT c.id, c.document_id, c.chunk_index, c.content, c.metadata,
               d.filename, d.file_type,
               ({_distance_expression(config)}) AS distance,
               ({_similarity_expression(config)}) AS similarity,
               CASE 
                   WHEN q.val = '' THEN 0.0
                   ELSE ts_rank_cd(to_tsvector('english', c.content), to_tsquery('english', q.val))
               END AS keyword_score
        FROM {schema}.rag_chunks c
        JOIN {schema}.rag_documents d ON d.id = c.document_id
        CROSS JOIN q
        {where_clause}
        ORDER BY {_distance_expression(config)}
        LIMIT {vector_limit_param}
    """

    # Keyword search has the FTS filter in its WHERE clause
    keyword_filters = list(filters) + ["q.val <> ''", "to_tsvector('english', c.content) @@ to_tsquery('english', q.val)"]
    keyword_where_clause = f"WHERE {' AND '.join(keyword_filters)}"

    query_keyword = f"""
        WITH q AS (
            SELECT coalesce(nullif(string_agg(lexeme, ' | '), ''), '') AS val
            FROM unnest(to_tsvector('english', $2::text))
        )
        SELECT c.id, c.document_id, c.chunk_index, c.content, c.metadata,
               d.filename, d.file_type,
               ({_distance_expression(config)}) AS distance,
               ({_similarity_expression(config)}) AS similarity,
               CASE 
                   WHEN q.val = '' THEN 0.0
                   ELSE ts_rank_cd(to_tsvector('english', c.content), to_tsquery('english', q.val))
               END AS keyword_score
        FROM {schema}.rag_chunks c
        JOIN {schema}.rag_documents d ON d.id = c.document_id
        CROSS JOIN q
        {keyword_where_clause}
        ORDER BY keyword_score DESC
        LIMIT {keyword_limit_param}
    """

    try:
        vector_rows = await pool.fetch(query_vector, *vector_parameters)
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"Vector search candidates fetch failed: {error}") from error

    try:
        keyword_rows = await pool.fetch(query_keyword, *keyword_parameters)
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"Keyword search candidates fetch failed: {error}") from error

    def _parse_row(row) -> SearchHit:
        return SearchHit(
            chunk_id=str(row["id"]),
            document_id=row["document_id"],
            content=row["content"],
            similarity=float(row["similarity"]),
            distance=float(row["distance"]),
            chunk_index=int(row["chunk_index"]),
            metadata=_as_dict(row["metadata"]),
            filename=row["filename"],
            file_type=row["file_type"],
            vector_score=float(row["similarity"]),
            keyword_score=float(row["keyword_score"] or 0.0),
        )

    vector_hits = [_parse_row(r) for r in vector_rows]
    keyword_hits = [_parse_row(r) for r in keyword_rows]

    return vector_hits, keyword_hits


async def get_document(
    document_id: str, *, settings: RagSettings | None = None
) -> dict[str, Any] | None:
    config = settings or rag_settings
    pool = await get_pool(config)
    row = await pool.fetchrow(
        f"SELECT * FROM {_schema(config)}.rag_documents WHERE id = $1", document_id
    )
    return _document_row(row) if row else None


async def list_documents(
    *,
    limit: int,
    offset: int,
    tenant_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    settings: RagSettings | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """One page of documents, newest first, plus the total for that filter."""
    config = settings or rag_settings
    pool = await get_pool(config)
    schema = _schema(config)

    parameters: list[Any] = []
    filters: list[str] = []
    if tenant_id:
        parameters.append(tenant_id)
        filters.append(f"(user_id = ${len(parameters)} OR metadata->>'tenant_id' = ${len(parameters)})")
    for column, value in (
        ("user_id", user_id),
        ("agent_id", agent_id),
        ("conversation_id", conversation_id),
    ):
        if value:
            parameters.append(value)
            filters.append(f"{column} = ${len(parameters)}")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    total = await pool.fetchval(
        f"SELECT count(*) FROM {schema}.rag_documents {where}", *parameters
    )
    rows = await pool.fetch(
        f"""
        SELECT * FROM {schema}.rag_documents {where}
        ORDER BY created_at DESC
        LIMIT ${len(parameters) + 1} OFFSET ${len(parameters) + 2}
        """,
        *parameters, int(limit), int(offset),
    )
    return [_document_row(row) for row in rows], int(total or 0)


async def stats(settings: RagSettings | None = None) -> dict[str, Any]:
    """Row counts, for /api/rag/health."""
    config = settings or rag_settings
    pool = await get_pool(config)
    schema = _schema(config)
    row = await pool.fetchrow(
        f"""
        SELECT (SELECT count(*) FROM {schema}.rag_documents) AS documents,
               (SELECT count(*) FROM {schema}.rag_chunks)    AS chunks
        """
    )
    return {"documents": int(row["documents"]), "chunks": int(row["chunks"])}


def _as_dict(value: Any) -> dict[str, Any]:
    """asyncpg hands JSONB back as a string unless a codec is registered."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except ValueError:
            return {}
    return {}


def _document_row(row: Any) -> dict[str, Any]:
    record = dict(row)
    record["metadata"] = _as_dict(record.get("metadata"))
    for field in ("created_at", "updated_at"):
        if record.get(field) is not None:
            record[field] = record[field].isoformat()
    return record
