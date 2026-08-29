"""RAG HTTP surface, mounted at /api/rag.

A separate router with its own prefix: no existing route changes path, shape or
behaviour, and removing the one `include_router` line in app/main.py removes
this entire feature.

Auth reuses the service's existing dependency (`require_token`), so these routes
are protected by the same bearer token as /agent/* - a document store must not
be the one open door in an otherwise closed service.

Errors are mapped in exactly one place, `_handled`, so every route answers the
same status code for the same failure:

    400 unsupported file type          422 unreadable or empty document
    404 unknown document               413 file over RAG_MAX_UPLOAD_MB
    502 embedding backend failed       503 database or parser unavailable
    500 database failure
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

import datetime
import uuid
from ..security import require_token
from ..services.mongodb_logger import mongodb_logger
from . import document_loader, embeddings, service, vector_store
from .config import rag_settings
from .schemas import (
    ContextRequest,
    ContextResponse,
    DeleteResponse,
    DocumentListResponse,
    DocumentSummary,
    RagHealthResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/rag",
    tags=["rag"],
    dependencies=[Depends(require_token)],
)

logs_router = APIRouter(
    prefix="/api",
    tags=["logs"],
    dependencies=[Depends(require_token)],
)


@contextmanager
def _handled():
    """One failure-to-status-code mapping for every route."""
    try:
        yield
    except document_loader.UnsupportedFileType as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(error)) from error
    except (document_loader.ExtractionError, service.EmptyDocument) as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
    except (document_loader.ParserUnavailable, vector_store.VectorStoreUnavailable) as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    except embeddings.EmbeddingError as error:
        # 502: this service is fine, the thing it depends on is not.
        logger.error("rag: embedding failure - %s", error)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(error)) from error
    except vector_store.VectorStoreError as error:
        logger.error("rag: database failure - %s", error)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(error)) from error


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF, DOCX or TXT document",
    responses={
        400: {"description": "Unsupported file type"},
        413: {"description": "File larger than RAG_MAX_UPLOAD_MB"},
        422: {"description": "The file holds no extractable text"},
        502: {"description": "The embedding backend failed"},
        503: {"description": "RAG storage is not configured"},
    },
)
async def upload_document(
    file: UploadFile = File(description="PDF, DOCX or TXT."),
    document_id: str | None = Form(
        default=None,
        description="Your own id for this document. Generated when omitted.",
    ),
    user_id: str | None = Form(default=None),
    agent_id: str | None = Form(default=None),
    conversation_id: str | None = Form(default=None),
    metadata: str | None = Form(
        default=None, description='Optional JSON object, e.g. {"tenant":"acme"}.'
    ),
) -> UploadResponse:
    """Ingest one document: extract, clean, chunk, embed and store.

    multipart/form-data. Only `file` is required - user_id, agent_id and
    conversation_id are optional scopes a caller may later filter searches by,
    and nothing in this service requires them.

    Synchronous by design: the caller learns how many chunks were stored, and a
    failure is a failed request rather than a job that quietly never finished.
    A very large PDF therefore takes as long as it takes to embed.
    """
    if file is None or not (file.filename or "").strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No file was uploaded")

    with _handled():
        # Validate the extension BEFORE reading the body: no reason to buffer
        # 25 MB of a file that is going to be rejected.
        document_loader.file_type_of(file.filename)

    content = await file.read()
    limit = rag_settings.rag_max_upload_mb * 1024 * 1024
    if len(content) > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{file.filename} is {len(content) / 1_048_576:.1f} MB; the limit is "
            f"{rag_settings.rag_max_upload_mb} MB (RAG_MAX_UPLOAD_MB)",
        )
    if not content:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"{file.filename} is empty (0 bytes)"
        )

    parsed_metadata = _parse_metadata(metadata)

    with _handled():
        result = await service.ingest(
            content=content,
            filename=file.filename,
            document_id=(document_id or "").strip() or None,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            metadata=parsed_metadata,
        )
    return UploadResponse(**result)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Vector similarity search over the stored chunks",
    responses={
        422: {"description": "Empty query or invalid top_k"},
        502: {"description": "The embedding backend failed"},
        503: {"description": "RAG storage is not configured"},
    },
)
async def search(request: SearchRequest = Body()) -> SearchResponse:
    """Embed the query, rank chunks in Postgres, return those above the threshold.

    `found` is false when nothing reaches RAG_SIMILARITY_THRESHOLD. No model is
    called in either case: what to do about an empty result is the calling
    agent's decision, not this endpoint's.
    """
    import time
    import asyncio

    request_id = f"req_{uuid.uuid4().hex}"
    start_time = time.perf_counter()
    error_msg = None
    outcome = None

    _check_top_k(request.top_k)

    try:
        with _handled():
            outcome = await service.search(**request.model_dump())
        return SearchResponse(**outcome)
    except Exception as e:
        error_msg = str(e)
        raise
    finally:
        took_ms = int((time.perf_counter() - start_time) * 1000)
        doc_ids = []
        chunk_ids = []
        scores = []
        found = False
        num_chunks = 0

        if outcome:
            found = outcome.get("found", False)
            results = outcome.get("results", [])
            num_chunks = len(results)
            for r in results:
                chunk_ids.append(r.get("chunk_id"))
                doc_ids.append(r.get("document_id"))
                scores.append(r.get("similarity"))
            # Deduplicate document IDs
            doc_ids = list(set(doc_ids))

        log_data = {
            "query": request.query,
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "request_id": request_id,
            "document_ids": doc_ids,
            "chunks_retrieved_count": num_chunks,
            "chunk_ids": chunk_ids,
            "similarity_scores": scores,
            "threshold": request.threshold if request.threshold is not None else rag_settings.rag_similarity_threshold,
            "top_k": request.top_k if request.top_k is not None else rag_settings.rag_top_k,
            "found": found,
            "took_ms": took_ms,
            "model_name": rag_settings.embedding_model,
            "error": error_msg,
        }
        # Safely queue the database log write without blocking
        asyncio.create_task(mongodb_logger.log_search(log_data))


@router.post(
    "/context",
    response_model=ContextResponse,
    summary="Retrieved chunks, assembled as prompt-ready context",
    responses={
        422: {"description": "Empty query or invalid top_k"},
        502: {"description": "The embedding backend failed"},
        503: {"description": "RAG storage is not configured"},
    },
)
async def context(request: ContextRequest = Body()) -> ContextResponse:
    """The same retrieval as /search, formatted as one block with its sources.

    This endpoint calls NO LLM and touches no agent code. It exists so the
    existing agent can fetch context when it decides to, without this task
    changing how the agent works.
    """
    _check_top_k(request.top_k)
    payload = request.model_dump()
    max_chars = payload.pop("max_chars", None)
    with _handled():
        outcome = await service.build_context(max_chars=max_chars, **payload)
    return ContextResponse(**outcome)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List ingested documents, newest first",
)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user_id: str | None = Query(default=None),
    agent_id: str | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
) -> DocumentListResponse:
    """Paginated. `total` is the count for the same filter, not for the page."""
    with _handled():
        rows, total = await vector_store.list_documents(
            limit=limit,
            offset=offset,
            user_id=user_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
        )
    return DocumentListResponse(
        documents=[DocumentSummary(**row) for row in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/documents/{document_id}",
    response_model=DocumentSummary,
    responses={404: {"description": "No such document"}},
    summary="One document's record",
)
async def get_document(document_id: str) -> DocumentSummary:
    with _handled():
        row = await vector_store.get_document(document_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document {document_id}")
    return DocumentSummary(**row)


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    responses={404: {"description": "No such document"}},
    summary="Delete a document, its chunks and their embeddings",
)
async def delete_document(document_id: str) -> DeleteResponse:
    """One transaction: either the document and all of its chunks go, or nothing
    does. A half-deleted document would leave chunks that still answer searches
    while their document record no longer exists."""
    with _handled():
        existed, chunks = await vector_store.delete_document(document_id)
    if not existed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No document {document_id}")
    logger.info("rag: document %s deleted (%s chunks)", document_id, chunks)
    return DeleteResponse(document_id=document_id, chunks_deleted=chunks)


@router.get(
    "/health",
    response_model=RagHealthResponse,
    summary="Whether the RAG layer is usable, and why not if it is not",
)
async def rag_health() -> RagHealthResponse:
    """Reports configuration names and numbers only - never a DSN or a key.

    Always 200, including when the layer is unusable: this describes RAG, and a
    non-200 here would be read as the service being down.
    """
    database: dict[str, Any] = {"configured": rag_settings.database_configured}
    if rag_settings.database_configured:
        try:
            database["column_dimension"] = await vector_store.column_dimension()
            database["schema_ready"] = database["column_dimension"] is not None
            database.update(await vector_store.stats())
            database["connected"] = True
        except Exception as error:  # noqa: BLE001 - a probe reports, never raises
            database["connected"] = False
            database["error"] = str(error)

    embedding_state: dict[str, Any] = {
        "model": rag_settings.embedding_model,
        "backend": rag_settings.resolved_backend,
    }
    try:
        embedder = await embeddings.get_embedder()
        embedding_state["loaded"] = True
        embedding_state["dimension"] = embedder.dimension
    except Exception as error:  # noqa: BLE001
        embedding_state["loaded"] = False
        embedding_state["error"] = str(error)

    ok = bool(
        database.get("connected")
        and database.get("schema_ready")
        and embedding_state.get("loaded")
        and embedding_state.get("dimension") == database.get("column_dimension")
    )
    return RagHealthResponse(
        ok=ok,
        configuration=rag_settings.public_state(),
        database=database,
        embeddings=embedding_state,
    )


def _check_top_k(top_k: int | None) -> None:
    if top_k is not None and top_k > rag_settings.rag_max_top_k:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"top_k must not exceed {rag_settings.rag_max_top_k} (RAG_MAX_TOP_K)",
        )


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    """The metadata form field, which multipart can only carry as a string."""
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"metadata must be valid JSON: {error}",
        ) from error
    if not isinstance(parsed, dict):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "metadata must be a JSON object",
        )
    return parsed


@logs_router.get(
    "/logs/searches",
    summary="Retrieve recent search logs",
)
async def get_search_logs(
    limit: int = Query(default=50, ge=1, le=200),
    document_id: str | None = Query(default=None),
    found: bool | None = Query(default=None),
) -> list[dict[str, Any]]:
    """Retrieve recent queries and metrics from the MongoDB logs collection."""
    return await mongodb_logger.get_search_logs(
        limit=limit,
        document_id=document_id,
        found=found,
    )

