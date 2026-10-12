"""Unit tests for Cloudflare Workers AI inference (embeddings and reranking).

Tests all requirements:
1. Successful embedding generation and L2 normalization with @cf/baai/bge-m3.
2. 1024-dimension enforcement.
3. Batching across requests.
4. HTTP 429 quota exhaustion controlled error handling.
5. HTTP 401/403 authentication failure handling.
6. Successful reranking and candidate ordering with @cf/baai/bge-reranker-base.
7. Verification that local ML models are NOT loaded when Cloudflare mode is active.
8. Validation of missing credentials.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pytest

from app.rag.cloudflare_ai import (
    CloudflareAIError,
    CloudflareAuthError,
    CloudflareModelError,
    CloudflareQuotaExhaustedError,
    embed_texts_cloudflare,
    normalize_l2,
    rerank_cloudflare,
)
from app.rag.config import RagSettings
from app.rag.embeddings import Embedder, EmbeddingError
from app.rag.reranker import RerankError, rerank
from app.rag.vector_store import SearchHit


def _mock_settings(**kwargs) -> RagSettings:
    defaults = {
        "embedding_provider": "cloudflare",
        "reranker_provider": "cloudflare",
        "cloudflare_account_id": "test_account_123",
        "cloudflare_api_token": "test_token_xyz",
        "cloudflare_embedding_model": "@cf/baai/bge-m3",
        "cloudflare_reranker_model": "@cf/baai/bge-reranker-base",
        "rag_embedding_dim": 1024,
        "embedding_batch_size": 2,
        "rag_rerank_candidates": 10,
    }
    defaults.update(kwargs)
    return RagSettings(**defaults)


def _mock_vector(val: float = 0.05, dim: int = 1024) -> list[float]:
    return [val] * dim


def _hit(chunk_id: str, content: str = "Test content", similarity: float = 0.5) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="doc_1",
        content=content,
        similarity=similarity,
        distance=1.0 - similarity,
        chunk_index=0,
        metadata={"source": "test.txt"},
        filename="test.txt",
        file_type="txt",
        vector_score=similarity,
        keyword_score=0.0,
    )


# ============================================================================
# L2 NORMALIZATION TESTS
# ============================================================================

def test_normalize_l2():
    raw = [3.0, 4.0]  # norm = 5.0
    norm = normalize_l2(raw)
    assert pytest.approx(norm[0], 0.0001) == 0.6
    assert pytest.approx(norm[1], 0.0001) == 0.8
    # Length of unit vector is 1.0
    assert pytest.approx(math.sqrt(sum(x * x for x in norm)), 0.0001) == 1.0


def test_normalize_l2_zero_vector():
    raw = [0.0, 0.0, 0.0]
    norm = normalize_l2(raw)
    assert norm == [0.0, 0.0, 0.0]


# ============================================================================
# EMBEDDING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_cloudflare_embedding_success():
    """Verify that Cloudflare returns 1024-d L2-normalized embeddings."""
    settings = _mock_settings()
    texts = ["First passage", "Second passage"]

    mock_vectors = [_mock_vector(0.03, 1024), _mock_vector(0.04, 1024)]
    mock_response = httpx.Response(
        status_code=200,
        json={"result": {"shape": [2, 1024], "data": mock_vectors}, "success": True},
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.AsyncClient()
    with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        result = await embed_texts_cloudflare(texts, settings, client=client)

    assert len(result) == 2
    assert len(result[0]) == 1024
    assert len(result[1]) == 1024

    # Verify unit length
    norm0 = math.sqrt(sum(x * x for x in result[0]))
    norm1 = math.sqrt(sum(x * x for x in result[1]))
    assert pytest.approx(norm0, 0.0001) == 1.0
    assert pytest.approx(norm1, 0.0001) == 1.0


@pytest.mark.asyncio
async def test_cloudflare_embedding_dimension_mismatch():
    """Verify that vectors not matching 1024 dimensions are loudly refused."""
    settings = _mock_settings(rag_embedding_dim=1024)
    texts = ["Test passage"]

    # Model returns 512 dimensions instead of 1024
    mock_vectors = [_mock_vector(0.05, 512)]
    mock_response = httpx.Response(
        status_code=200,
        json={"result": {"data": mock_vectors}, "success": True},
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.AsyncClient()
    with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(CloudflareModelError, match="invalid dimension 512"):
            await embed_texts_cloudflare(texts, settings, client=client)


@pytest.mark.asyncio
async def test_cloudflare_embedding_quota_exhausted():
    """Verify HTTP 429 returns CloudflareQuotaExhaustedError with helpful message."""
    settings = _mock_settings()
    texts = ["Test passage"]

    mock_response = httpx.Response(
        status_code=429,
        text="Quota exceeded",
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.AsyncClient()
    with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(CloudflareQuotaExhaustedError, match="HTTP 429"):
            await embed_texts_cloudflare(texts, settings, client=client)


@pytest.mark.asyncio
async def test_cloudflare_embedding_auth_failure():
    """Verify HTTP 401 returns CloudflareAuthError."""
    settings = _mock_settings()
    texts = ["Test passage"]

    mock_response = httpx.Response(
        status_code=401,
        text="Unauthorized",
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.AsyncClient()
    with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(CloudflareAuthError, match="HTTP 401"):
            await embed_texts_cloudflare(texts, settings, client=client)


@pytest.mark.asyncio
async def test_cloudflare_embedding_batching():
    """Verify multiple chunks are split into batches of embedding_batch_size."""
    settings = _mock_settings(embedding_batch_size=2)
    texts = ["chunk 1", "chunk 2", "chunk 3"]

    def mock_post_side_effect(url, json=None, headers=None):
        batch_len = len(json["text"])
        return httpx.Response(
            status_code=200,
            json={"result": {"data": [_mock_vector(0.01, 1024)] * batch_len}},
            request=httpx.Request("POST", url),
        )

    client = httpx.AsyncClient()
    with patch.object(client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = mock_post_side_effect
        result = await embed_texts_cloudflare(texts, settings, client=client)

    assert len(result) == 3
    # 3 texts with batch size 2 -> 2 calls (batch of 2, batch of 1)
    assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_embedder_cloudflare_mode_skips_local_model():
    """Verify that in Cloudflare mode, sentence_transformers is NOT loaded."""
    settings = _mock_settings(embedding_provider="cloudflare")
    embedder = Embedder(settings=settings)

    mock_response = httpx.Response(
        status_code=200,
        json={"result": {"data": [_mock_vector(0.02, 1024)]}},
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            dim = await embedder.load()

        assert dim == 1024
        assert embedder.ready is True
        assert embedder._model is None
        # SentenceTransformer was never instantiated
        mock_st.assert_not_called()


@pytest.mark.asyncio
async def test_cloudflare_missing_credentials_fails_fast():
    """Verify that missing credentials raise descriptive error without calling network."""
    settings = RagSettings(
        embedding_provider="cloudflare",
        cloudflare_account_id="",
        cloudflare_api_token="",
    )
    embedder = Embedder(settings=settings)
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN"):
        await embedder.load()


# ============================================================================
# RERANKER TESTS
# ============================================================================

def test_cloudflare_reranker_success():
    """Verify Cloudflare reranker orders candidates, maps scores, and sets metadata."""
    settings = _mock_settings(rag_rerank_candidates=2)
    query = "What is the course code?"

    candidates = [
        _hit("chunk_1", "Introductory overview", 0.8),
        _hit("chunk_2", "Course Code is CS101", 0.6),
        _hit("chunk_3", "Unranked candidate", 0.4),
    ]

    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "data": [
                    {"index": 0, "score": 0.2, "text": "Introductory overview"},
                    {"index": 1, "score": 0.95, "text": "Course Code is CS101"},
                ]
            },
            "success": True,
        },
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.Client()
    with patch.object(client, "post", return_value=mock_response):
        reranked, latency_ms = rerank_cloudflare(query, candidates, settings, client=client)

    assert len(reranked) == 3
    # chunk_2 had score 0.95 and matches code slot -> ranked first
    assert reranked[0].chunk_id == "chunk_2"
    assert reranked[0].rerank_score > reranked[1].rerank_score
    assert reranked[0].metadata["raw_rerank_score"] == 0.95
    assert reranked[0].metadata["is_evidence"] is True

    # chunk_3 was beyond candidates_to_rerank limit -> 0.0
    assert reranked[2].chunk_id == "chunk_3"
    assert reranked[2].rerank_score == 0.0
    assert latency_ms is not None


def test_cloudflare_reranker_quota_exhausted():
    """Verify HTTP 429 in reranker raises controlled error."""
    settings = _mock_settings()
    query = "Test query"
    candidates = [_hit("chunk_1")]

    mock_response = httpx.Response(
        status_code=429,
        text="Quota exceeded",
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.Client()
    with patch.object(client, "post", return_value=mock_response):
        with pytest.raises(CloudflareQuotaExhaustedError, match="HTTP 429"):
            rerank_cloudflare(query, candidates, settings, client=client)


def test_cloudflare_reranker_official_schema():
    """Verify official Cloudflare Workers AI schema ({result: {response: [{id, score}]}})."""
    settings = _mock_settings(rag_rerank_candidates=3)
    query = "What is the course code?"

    candidates = [
        _hit("chunk_0", "General introduction", 0.7),
        _hit("chunk_1", "Course Code is R1UC631B", 0.6),
        _hit("chunk_2", "Campus details", 0.5),
    ]

    # Cloudflare returns items sorted by score descending with "id" key
    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": [
                    {"id": 1, "score": 0.9994},
                    {"id": 0, "score": 0.0210},
                    {"id": 2, "score": 0.0015},
                ],
                "usage": {"total_tokens": 85},
            },
            "success": True,
            "errors": [],
            "messages": [],
        },
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.Client()
    with patch.object(client, "post", return_value=mock_response):
        reranked, latency_ms = rerank_cloudflare(query, candidates, settings, client=client)

    assert len(reranked) == 3
    # chunk_1 (id: 1) had top score and matched code slot
    assert reranked[0].chunk_id == "chunk_1"
    assert reranked[0].metadata["raw_rerank_score"] == 0.9994
    assert reranked[0].metadata["is_evidence"] is True

    # chunk_0 (id: 0) mapped accurately despite arriving 2nd in the response list
    assert reranked[1].chunk_id == "chunk_0"
    assert reranked[1].metadata["raw_rerank_score"] == 0.0210

    # chunk_2 (id: 2) mapped accurately
    assert reranked[2].chunk_id == "chunk_2"
    assert reranked[2].metadata["raw_rerank_score"] == 0.0015


def test_cloudflare_reranker_malformed_shape():
    """Verify malformed reranker response raises CloudflareModelError."""
    settings = _mock_settings()
    query = "Test query"
    candidates = [_hit("chunk_1")]

    mock_response = httpx.Response(
        status_code=200,
        json={"result": {"unexpected_key": 123}, "success": True},
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.Client()
    with patch.object(client, "post", return_value=mock_response):
        with pytest.raises(CloudflareModelError, match="Unrecognized response shape"):
            rerank_cloudflare(query, candidates, settings, client=client)


def test_cloudflare_reranker_missing_candidate_ids():
    """Verify response missing id/index raises CloudflareModelError."""
    settings = _mock_settings()
    query = "Test query"
    candidates = [_hit("chunk_1")]

    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": [{"missing_id_and_index": True, "score": 0.5}]
            },
            "success": True,
        },
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.Client()
    with patch.object(client, "post", return_value=mock_response):
        with pytest.raises(CloudflareModelError, match="no valid candidate scores or indices"):
            rerank_cloudflare(query, candidates, settings, client=client)


def test_cloudflare_reranker_logit_scores():
    """Verify logit scores outside [0, 1] are converted via sigmoid."""
    settings = _mock_settings()
    query = "Topical query"
    candidates = [_hit("chunk_1", "Relevant snippet")]

    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                # Raw logit = 2.5 -> sigmoid(2.5) ~ 0.92414
                "response": [{"id": 0, "score": 2.5}]
            },
            "success": True,
        },
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    client = httpx.Client()
    with patch.object(client, "post", return_value=mock_response):
        reranked, _ = rerank_cloudflare(query, candidates, settings, client=client)

    assert len(reranked) == 1
    # Check that score is bounded in [0, 1]
    assert 0.0 <= reranked[0].rerank_score <= 1.0
    expected_prob = 1.0 / (1.0 + math.exp(-2.5))
    assert pytest.approx(reranked[0].metadata["raw_rerank_score"], 0.0001) == round(expected_prob, 6)


def test_reranker_service_cloudflare_mode_skips_crossencoder():
    """Verify that in Cloudflare mode, local CrossEncoder is never loaded."""
    settings = _mock_settings(reranker_provider="cloudflare")
    query = "Test query"
    candidates = [_hit("chunk_1")]

    mock_response = httpx.Response(
        status_code=200,
        json={
            "result": {
                "response": [{"id": 0, "score": 0.88}]
            }
        },
        request=httpx.Request("POST", "https://api.cloudflare.com"),
    )

    with patch("app.rag.reranker.get_reranker_model") as mock_local_model:
        with patch.object(httpx.Client, "post", return_value=mock_response):
            reranked, _ = rerank(query, candidates, settings=settings)

        assert len(reranked) == 1
        assert reranked[0].rerank_score == 0.88
        # Local model was NEVER requested
        mock_local_model.assert_not_called()
