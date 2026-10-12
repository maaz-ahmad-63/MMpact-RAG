"""Cloudflare Workers AI client for embeddings and reranking.

Integrates with Cloudflare Workers AI REST API:
- Embeddings: @cf/baai/bge-m3 (1024-dimensional dense vectors)
- Reranking:  @cf/baai/bge-reranker-base (cross-encoder relevance scoring)

Security & Reliability Guarantees:
- Never logs API tokens or account IDs.
- Validates dimensions (strictly 1024-d).
- Ensures L2 vector normalization for cosine similarity compatibility with pgvector.
- Controlled handling for HTTP 429 rate limit / quota exhaustion.
- Reusable HTTP client connections and configurable timeouts.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Sequence

import httpx

if TYPE_CHECKING:
    from app.rag.config import RagSettings
    from app.rag.vector_store import SearchHit

logger = logging.getLogger(__name__)

CF_API_BASE = "https://api.cloudflare.com/client/v4/accounts"


class EmbeddingError(Exception):
    """The embedding backend could not produce vectors. -> HTTP 502."""


class CloudflareAIError(Exception):
    """Base exception for Cloudflare Workers AI failures."""


class CloudflareQuotaExhaustedError(CloudflareAIError, EmbeddingError):
    """Raised when Cloudflare returns HTTP 429 quota exhaustion or rate limits."""


class CloudflareAuthError(CloudflareAIError):
    """Raised when Cloudflare returns HTTP 401 or 403 authentication failures."""


class CloudflareModelError(CloudflareAIError):
    """Raised when Cloudflare model returns malformed, unexpected, or incompatible data."""


def normalize_l2(vector: Sequence[float]) -> list[float]:
    """Normalize a vector to unit length (L2 norm = 1.0) for cosine similarity."""
    squared_sum = sum(x * x for x in vector)
    if squared_sum <= 0.0:
        return [float(x) for x in vector]
    norm = math.sqrt(squared_sum)
    return [float(x) / norm for x in vector]


def _build_url(account_id: str, model_name: str) -> str:
    """Build the official Cloudflare Workers AI execution endpoint."""
    cleaned_account = account_id.strip()
    cleaned_model = model_name.strip()
    return f"{CF_API_BASE}/{cleaned_account}/ai/run/{cleaned_model}"


def _build_headers(api_token: str) -> dict[str, str]:
    """Build request headers without logging or exposing the secret token."""
    return {
        "Authorization": f"Bearer {api_token.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "MMpact-RAG-Inference-Client/1.0",
    }


def _handle_http_error(response: httpx.Response, action_description: str) -> None:
    """Transform HTTP errors into controlled exceptions without leaking sensitive credentials."""
    status_code = response.status_code
    
    if status_code == 429:
        raise CloudflareQuotaExhaustedError(
            "Cloudflare Workers AI quota exhausted or rate limit reached (HTTP 429). "
            "Please verify your Cloudflare Workers AI plan and allocation."
        )
    if status_code in (401, 403):
        raise CloudflareAuthError(
            f"Cloudflare Workers AI authentication failed (HTTP {status_code}). "
            "Please check CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in your environment."
        )

    # Sanitize and truncate response body for safety
    try:
        body = response.json()
        errors = body.get("errors", [])
        if errors and isinstance(errors, list):
            detail_msg = "; ".join(str(e.get("message", e)) for e in errors)
        else:
            detail_msg = response.text[:200]
    except Exception:
        detail_msg = response.text[:200]

    raise CloudflareAIError(
        f"Cloudflare Workers AI {action_description} failed (HTTP {status_code}): {detail_msg}"
    )


# ============================================================================
# EMBEDDINGS (@cf/baai/bge-m3)
# ============================================================================

def _parse_embedding_response(
    body: dict[str, Any] | list[Any],
    expected_count: int,
    expected_dim: int,
) -> list[list[float]]:
    """Extract and validate 1024-dimensional vectors from Cloudflare Workers AI response."""
    raw_data: Any = None
    if isinstance(body, dict):
        if not body.get("success", True) and "errors" in body:
            errors = body.get("errors", [])
            raise CloudflareModelError(f"Cloudflare Workers AI returned success=false: {errors}")
        result = body.get("result")
        if isinstance(result, dict) and "data" in result:
            raw_data = result["data"]
        elif isinstance(result, list):
            raw_data = result
        elif "data" in body:
            raw_data = body["data"]
    elif isinstance(body, list):
        raw_data = body

    if raw_data is None:
        raise CloudflareModelError(
            f"Unrecognized response shape from Cloudflare embedding endpoint: {type(body).__name__}"
        )

    # Handle single string return shape: [1024 floats] instead of [[1024 floats]]
    if raw_data and isinstance(raw_data, list) and isinstance(raw_data[0], (int, float)):
        raw_data = [raw_data]

    if len(raw_data) != expected_count:
        raise CloudflareModelError(
            f"Cloudflare Workers AI returned {len(raw_data)} vectors, expected {expected_count}"
        )

    normalized_vectors: list[list[float]] = []
    for idx, row in enumerate(raw_data):
        if not isinstance(row, list) or len(row) != expected_dim:
            actual_dim = len(row) if isinstance(row, list) else type(row).__name__
            raise CloudflareModelError(
                f"Vector at index {idx} has invalid dimension {actual_dim} (expected {expected_dim})"
            )
        # Apply strict L2 normalization for pgvector cosine distance compatibility
        normalized_vectors.append(normalize_l2(row))

    return normalized_vectors


async def embed_texts_cloudflare(
    texts: Sequence[str],
    settings: RagSettings,
    client: httpx.AsyncClient | None = None,
) -> list[list[float]]:
    """Generate dense embeddings via Cloudflare Workers AI (@cf/baai/bge-m3).
    
    Batches inputs according to settings.embedding_batch_size, validates dimension (1024),
    and normalizes vectors to unit L2 norm.
    """
    items = list(texts)
    if not items:
        return []

    settings.validate_cloudflare_config("embeddings")

    url = _build_url(settings.cloudflare_account_id, settings.cloudflare_embedding_model)
    headers = _build_headers(settings.cloudflare_api_token)
    batch_size = max(1, settings.embedding_batch_size)
    expected_dim = settings.rag_embedding_dim

    vectors: list[list[float]] = []
    own_client = client is None

    async_client = client or httpx.AsyncClient(
        timeout=settings.cloudflare_timeout_s,
        headers=headers,
    )

    try:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            payload = {"text": batch}

            try:
                response = await async_client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as error:
                raise CloudflareAIError(
                    f"Cloudflare Workers AI embedding request timed out after {settings.cloudflare_timeout_s}s"
                ) from error
            except httpx.RequestError as error:
                raise CloudflareAIError(
                    f"Failed to connect to Cloudflare Workers AI: {error}"
                ) from error

            if response.status_code != 200:
                _handle_http_error(response, "embedding")

            try:
                body = response.json()
            except ValueError as error:
                raise CloudflareModelError(
                    "Cloudflare Workers AI returned non-JSON response"
                ) from error

            batch_vectors = _parse_embedding_response(body, len(batch), expected_dim)
            vectors.extend(batch_vectors)

    finally:
        if own_client:
            await async_client.aclose()

    return vectors


# ============================================================================
# RERANKER (@cf/baai/bge-reranker-base)
# ============================================================================

def _parse_reranker_response(
    body: dict[str, Any] | list[Any],
    expected_count: int,
) -> list[dict[str, Any]]:
    """Extract candidate scores and indices from Cloudflare Workers AI reranker response."""
    items: Any = None
    if isinstance(body, dict):
        if not body.get("success", True) and "errors" in body:
            errors = body.get("errors", [])
            raise CloudflareModelError(f"Cloudflare Workers AI reranker returned success=false: {errors}")
        result = body.get("result")
        if isinstance(result, dict):
            # Official Cloudflare Workers AI schema for @cf/baai/bge-reranker-base:
            # {"result": {"response": [{"id": 0, "score": 0.999}, ...], "usage": {...}}}
            if "response" in result and isinstance(result["response"], list):
                items = result["response"]
            elif "data" in result and isinstance(result["data"], list):
                items = result["data"]
            elif "results" in result and isinstance(result["results"], list):
                items = result["results"]
        elif isinstance(result, list):
            items = result
        elif "response" in body and isinstance(body["response"], list):
            items = body["response"]
        elif "data" in body and isinstance(body["data"], list):
            items = body["data"]
    elif isinstance(body, list):
        items = body

    if not isinstance(items, list):
        shape_info = list(body.keys()) if isinstance(body, dict) else type(body).__name__
        raise CloudflareModelError(
            f"Unrecognized response shape from Cloudflare reranker endpoint. Available keys: {shape_info}"
        )

    return items


def rerank_cloudflare(
    query: str,
    candidates: list[SearchHit],
    settings: RagSettings,
    client: httpx.Client | None = None,
) -> tuple[list[SearchHit], int | None]:
    """Score and order candidate chunks using Cloudflare Workers AI (@cf/baai/bge-reranker-base).
    
    Returns:
        (reranked_candidates, rerank_ms)
    """
    from app.rag.reranker import evaluate_chunk_evidence

    if not candidates:
        return [], 0

    settings.validate_cloudflare_config("reranker")

    candidates_to_rerank = candidates[: settings.rag_rerank_candidates]
    remaining_candidates = candidates[settings.rag_rerank_candidates :]

    url = _build_url(settings.cloudflare_account_id, settings.cloudflare_reranker_model)
    headers = _build_headers(settings.cloudflare_api_token)

    # Prepare contextualized text passages matching local reranker convention
    contexts = []
    for hit in candidates_to_rerank:
        source = hit.metadata.get("source") if hit.metadata else None
        contextualized_text = f"Document: {source}\n\n{hit.content}" if source else hit.content
        contexts.append({"text": contextualized_text})

    payload = {
        "query": query,
        "contexts": contexts,
    }

    started = time.perf_counter()
    own_client = client is None
    sync_client = client or httpx.Client(
        timeout=settings.cloudflare_timeout_s,
        headers=headers,
    )

    try:
        try:
            response = sync_client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as error:
            raise CloudflareAIError(
                f"Cloudflare Workers AI reranker request timed out after {settings.cloudflare_timeout_s}s"
            ) from error
        except httpx.RequestError as error:
            raise CloudflareAIError(
                f"Failed to connect to Cloudflare Workers AI reranker: {error}"
            ) from error

        if response.status_code != 200:
            _handle_http_error(response, "reranker")

        try:
            body = response.json()
        except ValueError as error:
            raise CloudflareModelError(
                "Cloudflare Workers AI reranker returned non-JSON response"
            ) from error

        raw_items = _parse_reranker_response(body, len(candidates_to_rerank))

        # Build map: index -> raw_score
        # Cloudflare returns [{'id': int, 'score': float}, ...]
        # Also supports [{'index': int, 'score': float}, ...]
        score_by_index: dict[int, float] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            idx = item.get("id") if "id" in item else item.get("index")
            if idx is None:
                continue
            try:
                idx_int = int(idx)
            except (ValueError, TypeError):
                continue

            score_val = item.get("score") if "score" in item else item.get("relevance_score")
            if score_val is None:
                continue
            try:
                score_by_index[idx_int] = float(score_val)
            except (ValueError, TypeError):
                continue

        if not score_by_index and candidates_to_rerank:
            raise CloudflareModelError(
                "Cloudflare Workers AI reranker response contains no valid candidate scores or indices"
            )

        # Map back to original candidate chunks and apply evidence evaluation
        for idx, hit in enumerate(candidates_to_rerank):
            raw_score = score_by_index.get(idx, 0.0)
            if math.isnan(raw_score) or math.isinf(raw_score):
                raw_score = 0.0
            
            # If cross-encoder returned logit outside [0, 1], map via sigmoid
            if raw_score < 0.0 or raw_score > 1.0:
                prob = 1.0 / (1.0 + math.exp(-raw_score))
            else:
                prob = raw_score

            final_ev_score, meta = evaluate_chunk_evidence(query, hit.content, prob)
            hit.rerank_score = round(final_ev_score, 6)
            if hit.metadata is None:
                hit.metadata = {}
            hit.metadata["raw_rerank_score"] = round(prob, 6)
            hit.metadata["matched_targets"] = meta.get("targets_matched", [])
            hit.metadata["is_evidence"] = meta.get("is_evidence", False)

        # Chunks beyond candidate limit receive 0.0 rerank score
        for hit in remaining_candidates:
            hit.rerank_score = 0.0

        # Sort candidate chunks by final rerank score descending
        reranked = sorted(candidates_to_rerank, key=lambda h: h.rerank_score or 0.0, reverse=True)
        reranked.extend(remaining_candidates)

        rerank_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "rag: reranked %d candidates via Cloudflare Workers AI in %d ms (top=%.4f)",
            len(candidates_to_rerank),
            rerank_ms,
            reranked[0].rerank_score if reranked else 0.0,
        )
        return reranked, rerank_ms

    finally:
        if own_client:
            sync_client.close()
