import pytest
import sys
from unittest.mock import MagicMock, patch
from app.rag import service, vector_store, reranker
from app.rag.config import RagSettings

def _hit(chunk_id, similarity, keyword_score=0.0):
    return vector_store.SearchHit(
        chunk_id=chunk_id,
        document_id="doc_1",
        content=f"Content for {chunk_id}",
        similarity=similarity,
        distance=1.0 - similarity,
        chunk_index=0,
        metadata={"source": "test.txt"},
        filename="test.txt",
        file_type="txt",
        vector_score=similarity,
        keyword_score=keyword_score
    )

@pytest.fixture(autouse=True)
def cleanup_reranker_cache():
    """Reset global reranker cache between tests."""
    reranker._model = None
    reranker._model_name = None
    yield
    reranker._model = None
    reranker._model_name = None

@pytest.mark.asyncio
async def test_reranker_ordering_and_query_passing(monkeypatch):
    """Reranker should correctly re-order chunks based on relevance scores and receive the query."""
    config = RagSettings(
        rag_rerank_enabled=True,
        rag_rerank_candidates=10,
        rag_rerank_top_k=5,
        rag_similarity_threshold=0.1,
    )

    vector_hits = [_hit("chunk_A", 0.9, 0.0), _hit("chunk_B", 0.8, 0.0)]
    keyword_hits = []

    # Mock hybrid_search as an async function
    async def mock_hybrid_search(*args, **kwargs):
        return (vector_hits, keyword_hits)

    monkeypatch.setattr(vector_store, "hybrid_search", mock_hybrid_search)

    # Mock the CrossEncoder model
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.1, 0.95]
    monkeypatch.setattr(reranker, "get_reranker_model", lambda name: mock_encoder)

    query = "original target query"
    outcome = await service.search(query=query, settings=config)

    assert outcome["found"] is True
    assert len(outcome["results"]) == 2
    
    mock_encoder.predict.assert_called_once_with([
        (query, "Document: test.txt\n\nContent for chunk_A"),
        (query, "Document: test.txt\n\nContent for chunk_B"),
    ])

    assert outcome["results"][0]["chunk_id"] == "chunk_B"
    assert outcome["results"][0]["rerank_score"] == 0.95
    assert outcome["results"][1]["chunk_id"] == "chunk_A"
    assert outcome["results"][1]["rerank_score"] == 0.1

    assert "vector_score" in outcome["results"][0]
    assert "keyword_score" in outcome["results"][0]
    assert "rrf_score" in outcome["results"][0]
    assert "rerank_score" in outcome["results"][0]


@pytest.mark.asyncio
async def test_candidate_count_and_top_k_respected(monkeypatch):
    """Reranker should respect the candidates limits and top_k parameter."""
    config = RagSettings(
        rag_rerank_enabled=True,
        rag_rerank_candidates=2,
        rag_rerank_top_k=1,
        rag_similarity_threshold=0.1,
    )

    vector_hits = [_hit("chunk_A", 0.9), _hit("chunk_B", 0.8), _hit("chunk_C", 0.7)]
    keyword_hits = []

    async def mock_hybrid_search(*args, **kwargs):
        return (vector_hits, keyword_hits)

    monkeypatch.setattr(vector_store, "hybrid_search", mock_hybrid_search)

    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = [0.1, 0.95]
    monkeypatch.setattr(reranker, "get_reranker_model", lambda name: mock_encoder)

    outcome = await service.search(query="query text", top_k=1, settings=config)

    assert len(outcome["results"]) == 1
    mock_encoder.predict.assert_called_once_with([
        ("query text", "Document: test.txt\n\nContent for chunk_A"),
        ("query text", "Document: test.txt\n\nContent for chunk_B"),
    ])



@pytest.mark.asyncio
async def test_reranker_disabled_keeps_rrf_intact(monkeypatch):
    """When reranker is disabled, ranking remains in RRF order, and rerank_score is None."""
    config = RagSettings(
        rag_rerank_enabled=False,
        rag_similarity_threshold=0.1,
    )

    vector_hits = [_hit("chunk_A", 0.9, 0.0), _hit("chunk_B", 0.8, 0.0)]
    keyword_hits = []

    async def mock_hybrid_search(*args, **kwargs):
        return (vector_hits, keyword_hits)

    monkeypatch.setattr(vector_store, "hybrid_search", mock_hybrid_search)

    mock_encoder = MagicMock()
    monkeypatch.setattr(reranker, "get_reranker_model", lambda name: mock_encoder)

    outcome = await service.search(query="query text", settings=config)
    
    mock_encoder.predict.assert_not_called()
    assert outcome["results"][0]["chunk_id"] == "chunk_A"
    assert outcome["results"][0]["rerank_score"] is None
    assert outcome["rerank_ms"] is None


@pytest.mark.asyncio
async def test_reranker_failure_graceful_fallback(monkeypatch):
    """When reranking fails during execution, fallback gracefully to RRF with null rerank_scores."""
    config = RagSettings(
        rag_rerank_enabled=True,
        rag_similarity_threshold=0.1,
    )

    vector_hits = [_hit("chunk_A", 0.9, 0.0), _hit("chunk_B", 0.8, 0.0)]
    keyword_hits = []

    async def mock_hybrid_search(*args, **kwargs):
        return (vector_hits, keyword_hits)

    monkeypatch.setattr(vector_store, "hybrid_search", mock_hybrid_search)

    mock_encoder = MagicMock()
    mock_encoder.predict.side_effect = RuntimeError("Out of Memory")
    monkeypatch.setattr(reranker, "get_reranker_model", lambda name: mock_encoder)

    outcome = await service.search(query="query text", settings=config)

    assert len(outcome["results"]) == 2
    assert outcome["results"][0]["chunk_id"] == "chunk_A"
    assert outcome["results"][0]["rerank_score"] is None
    assert outcome["rerank_ms"] is None


@pytest.mark.asyncio
async def test_reranker_unavailable_graceful_fallback(monkeypatch):
    """When sentence-transformers library cannot be imported, fallback gracefully to RRF."""
    config = RagSettings(
        rag_rerank_enabled=True,
        rag_similarity_threshold=0.1,
    )

    vector_hits = [_hit("chunk_A", 0.9), _hit("chunk_B", 0.8)]
    keyword_hits = []

    async def mock_hybrid_search(*args, **kwargs):
        return (vector_hits, keyword_hits)

    monkeypatch.setattr(vector_store, "hybrid_search", mock_hybrid_search)

    with patch.dict(sys.modules, {"sentence_transformers": None}):
        outcome = await service.search(query="query text", settings=config)

        assert len(outcome["results"]) == 2
        assert outcome["results"][0]["chunk_id"] == "chunk_A"
        assert outcome["results"][0]["rerank_score"] is None
        assert outcome["rerank_ms"] is None
