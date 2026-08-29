"""The RAG pipeline, orchestrated. No HTTP, no SQL - just the sequence.

    bytes -> extract -> clean -> chunk -> embed (batched) -> bulk insert
    query -> embed -> pgvector search -> threshold -> results / context

The router below it deals only in HTTP concerns and the modules above it each
do one thing; this is the file that knows the order. It is importable and
testable on its own, and a future in-process caller (the agent's own tool
dispatcher, a background worker) can use it without going through the API.

Deliberately absent: any call to an LLM. `build_context` formats retrieved text
and stops. What to do with it - or with nothing - is the existing agent's
decision, and this task does not make it.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from . import document_loader, embeddings, vector_store
from .chunker import chunk_segments
from .config import RagSettings, rag_settings
from .text_cleaner import clean_text, is_effectively_empty

logger = logging.getLogger(__name__)


class EmptyDocument(Exception):
    """The file parsed, but holds no text worth embedding. -> HTTP 422."""


def new_document_id() -> str:
    return f"doc_{uuid.uuid4().hex}"


async def ingest(
    *,
    content: bytes,
    filename: str,
    document_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    settings: RagSettings | None = None,
) -> dict[str, Any]:
    """One file, all the way to stored embeddings.

    Raises UnsupportedFileType / ExtractionError / EmptyDocument /
    EmbeddingError / VectorStoreError - the router maps each to a status code.
    """
    config = settings or rag_settings
    started = time.perf_counter()
    document_id = document_id or new_document_id()

    file_type = document_loader.file_type_of(filename)
    logger.info(
        "rag: document received (id=%s file=%s type=%s bytes=%s)",
        document_id, filename, file_type, len(content),
    )

    # -- extract ------------------------------------------------------------
    logger.info("rag: text extraction started (id=%s)", document_id)
    segments = document_loader.load(content, filename)
    pages = len({segment.page for segment in segments if segment.page is not None}) or None

    cleaned = [
        document_loader.Segment(text=clean_text(segment.text), page=segment.page)
        for segment in segments
    ]
    cleaned = [segment for segment in cleaned if not is_effectively_empty(segment.text)]
    characters = sum(len(segment.text) for segment in cleaned)
    logger.info(
        "rag: text extraction completed (id=%s segments=%s chars=%s pages=%s)",
        document_id, len(cleaned), characters, pages,
    )

    if not cleaned or characters == 0:
        raise EmptyDocument(
            f"No text could be extracted from '{filename}'. If it is a scanned "
            f"PDF it needs OCR, which is not enabled on this service."
        )

    # -- chunk --------------------------------------------------------------
    chunks = chunk_segments(
        cleaned,
        document_id=document_id,
        source=filename,
        settings=config,
        extra_metadata={
            key: value
            for key, value in (
                ("user_id", user_id),
                ("agent_id", agent_id),
                ("conversation_id", conversation_id),
            )
            if value
        },
    )
    logger.info(
        "rag: %s chunks created (id=%s size=%s overlap=%s)",
        len(chunks), document_id, config.rag_chunk_size, config.rag_chunk_overlap,
    )
    if not chunks:
        raise EmptyDocument(f"'{filename}' produced no chunks worth storing")

    # -- embed --------------------------------------------------------------
    # Before anything is written: the model's real width against the column's
    # declared width. Failing here costs one probe; failing later costs a
    # half-ingested document.
    embedder = await embeddings.get_embedder()
    declared = await vector_store.column_dimension(config)
    if declared is None:
        raise vector_store.VectorStoreError(
            "The rag_chunks table does not exist. Run the migration: "
            "python -m app.rag.migrate"
        )
    await embedder.verify_dimension(declared)

    logger.info(
        "rag: embedding generation started (id=%s chunks=%s model=%s batch=%s)",
        document_id, len(chunks), embedder.model_name, config.embedding_batch_size,
    )
    vectors = await embedder.embed_texts([chunk.text for chunk in chunks])
    logger.info("rag: embedding generation completed (id=%s vectors=%s)", document_id, len(vectors))

    # -- store --------------------------------------------------------------
    file_path = _persist_original(content, document_id, filename, config)
    await vector_store.insert_document(
        document_id=document_id,
        filename=filename,
        file_type=file_type,
        file_path=file_path,
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        metadata=metadata or {},
        settings=config,
    )
    stored = await vector_store.insert_chunks(
        document_id,
        (
            (chunk.index, chunk.text, vector, chunk.metadata)
            for chunk, vector in zip(chunks, vectors)
        ),
        settings=config,
    )
    took_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "rag: embeddings stored (id=%s chunks=%s in %s ms)", document_id, stored, took_ms
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "chunks": stored,
        "characters": characters,
        "pages": pages,
        "embedding_model": embedder.model_name,
        "embedding_dim": declared,
        "took_ms": took_ms,
        "metadata": metadata or {},
    }


def _persist_original(
    content: bytes, document_id: str, filename: str, config: RagSettings
) -> str | None:
    """Keep the uploaded bytes, so a re-chunk later needs no re-upload.

    Best effort: a full or read-only disk must not lose a document that is
    otherwise ingested successfully, so a failure here is logged and the record
    simply carries no file_path.
    """
    if not config.rag_storage_dir:
        return None
    try:
        directory = Path(config.rag_storage_dir)
        directory.mkdir(parents=True, exist_ok=True)
        # The stored name is derived from the document id, never from the
        # client's filename: an uploaded name is attacker-controlled and
        # "../../x" must not decide where bytes land.
        extension = document_loader.file_type_of(filename)
        path = directory / f"{document_id}.{extension}"
        path.write_bytes(content)
        return str(path)
    except OSError as error:
        logger.warning("rag: could not keep the original of %s (%s)", document_id, error)
        return None


# -- retrieval --------------------------------------------------------------


async def search(
    *,
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    conversation_id: str | None = None,
    document_ids: Sequence[str] | None = None,
    settings: RagSettings | None = None,
) -> dict[str, Any]:
    """Embed the query, search pgvector, apply the threshold."""
    config = settings or rag_settings
    started = time.perf_counter()
    limit = min(int(top_k or config.rag_top_k), config.rag_max_top_k)
    cutoff = config.rag_similarity_threshold if threshold is None else float(threshold)

    logger.info(
        "rag: vector search started (chars=%s top_k=%s threshold=%s)",
        len(query), limit, cutoff,
    )
    # Checked before the query is embedded: without a database there is nothing
    # to search, and "storage is not configured" is a far more useful answer
    # than whatever the embedding backend says about itself.
    if not config.database_configured:
        raise vector_store.VectorStoreUnavailable(
            "RAG storage is not configured. Set RAG_DATABASE_URL (or DATABASE_URL) "
            "to a Postgres instance with pgvector. Only the RAG routes are affected."
        )

    embedder = await embeddings.get_embedder()
    vector = await embedder.embed_text(query)

    hits = await vector_store.search(
        vector,
        top_k=limit,
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        document_ids=document_ids,
        settings=config,
    )

    # Filtering here rather than in SQL: the ORDER BY ... LIMIT is what the
    # vector index accelerates, and a WHERE on a computed distance would fight
    # it. At top_k rows the cost of discarding in Python is nil.
    passing = [hit for hit in hits if hit.similarity >= cutoff]
    took_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "rag: search returned %s/%s chunks above %.2f in %s ms",
        len(passing), len(hits), cutoff, took_ms,
    )

    return {
        "query": query,
        "found": bool(passing),
        "results": [
            {
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "content": hit.content,
                "similarity": round(hit.similarity, 6),
                "metadata": {
                    **hit.metadata,
                    "source": hit.metadata.get("source") or hit.filename,
                    "chunk_index": hit.chunk_index,
                },
            }
            for hit in passing
        ],
        "threshold": cutoff,
        "top_k": limit,
        "took_ms": took_ms,
    }


async def build_context(
    *,
    query: str,
    max_chars: int | None = None,
    settings: RagSettings | None = None,
    **search_kwargs: Any,
) -> dict[str, Any]:
    """Retrieved chunks, assembled into one block of text with its sources.

    No model is called. The format - a numbered, labelled excerpt per chunk - is
    chosen so a prompt can cite it ("[1] policy.pdf, page 4") and so the model
    can tell two documents apart inside one block.
    """
    config = settings or rag_settings
    started = time.perf_counter()
    outcome = await search(query=query, settings=config, **search_kwargs)

    budget = int(max_chars or config.rag_context_max_chars)
    blocks: list[str] = []
    sources: list[dict[str, Any]] = []
    used = 0

    for position, result in enumerate(outcome["results"], start=1):
        metadata = result["metadata"]
        source = metadata.get("source") or "document"
        page = metadata.get("page")
        label = f"[{position}] {source}" + (f", page {page}" if page else "")
        block = f"{label}\n{result['content']}"

        if used + len(block) > budget:
            # Stop at the boundary rather than truncating mid-chunk: half a
            # sentence in a prompt reads as a fact that was never written.
            break

        blocks.append(block)
        used += len(block) + 2
        sources.append(
            {
                "document_id": result["document_id"],
                "source": source,
                "page": page,
                "chunk_index": metadata.get("chunk_index"),
                "similarity": result["similarity"],
            }
        )

    return {
        "query": query,
        "found": bool(blocks),
        "context": "\n\n".join(blocks),
        "sources": sources,
        "chunks_used": len(blocks),
        "took_ms": int((time.perf_counter() - started) * 1000),
    }


# -- lifecycle --------------------------------------------------------------


async def startup(settings: RagSettings | None = None) -> None:
    """Warm the model and check the schema. NEVER raises.

    Called from the app's lifespan. A RAG layer that is misconfigured must
    degrade to 503 on its own routes and leave the agent running - the priority
    is the existing service, not this one.
    """
    config = settings or rag_settings
    if not config.database_configured:
        logger.info(
            "rag: RAG_DATABASE_URL is not set - /api/rag/* answers 503. "
            "The agent is unaffected."
        )
        return

    try:
        declared = await vector_store.column_dimension(config)
        if declared is None:
            logger.warning(
                "rag: connected, but rag_chunks does not exist. "
                "Run `python -m app.rag.migrate` before uploading."
            )
        else:
            logger.info("rag: schema present (embedding column is vector(%s))", declared)
    except Exception as error:  # noqa: BLE001 - never block boot
        logger.warning("rag: database not reachable at startup (%s)", error)

    await embeddings.warmup()


async def shutdown() -> None:
    await embeddings.shutdown()
    await vector_store.close_pool()
