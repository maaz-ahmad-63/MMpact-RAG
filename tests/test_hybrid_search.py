import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from app.rag import service, vector_store
from app.rag.config import RagSettings, get_rag_settings
from app.rag.vector_store import SearchHit
import uuid


def _hit(
    chunk_id: str,
    similarity: float,
    keyword_score: float = 0.0,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id="doc_1",
        content=f"Content of {chunk_id}",
        similarity=similarity,
        distance=1.0 - similarity,
        chunk_index=0,
        metadata={},
        filename="test.pdf",
        file_type="pdf",
        vector_score=similarity,
        keyword_score=keyword_score,
    )


@pytest.fixture
def mock_embedder(monkeypatch):
    embedder = AsyncMock()
    embedder.embed_text.return_value = [0.1] * 8
    monkeypatch.setattr(service.embeddings, "get_embedder", AsyncMock(return_value=embedder))
    return embedder


@pytest.mark.asyncio
async def test_vector_only_match(monkeypatch, mock_embedder):
    """Should verify that vector-only matches are retrieved and scored with vector rank contribution."""
    config = RagSettings(
        rag_vector_candidates=10,
        rag_keyword_candidates=10,
        rag_rrf_k=60,
        rag_similarity_threshold=0.5,
    )

    # Vector score 0.9 is above cutoff, keyword score 0.0
    vector_hits = [_hit("chunk_v1", 0.9, 0.0)]
    keyword_hits = []

    monkeypatch.setattr(
        vector_store,
        "hybrid_search",
        AsyncMock(return_value=(vector_hits, keyword_hits)),
    )

    outcome = await service.search(query="query text", settings=config)
    assert outcome["found"] is True
    assert len(outcome["results"]) == 1
    assert outcome["results"][0]["chunk_id"] == "chunk_v1"
    # RRF score: 1.0 / (60 + 1) = 0.016393
    assert outcome["results"][0]["metadata"]["rrf_score"] == pytest.approx(0.016393, abs=1e-5)


@pytest.mark.asyncio
async def test_keyword_only_match(monkeypatch, mock_embedder):
    """Should verify that keyword-only matches are retrieved and scored with FTS rank contribution."""
    config = RagSettings(
        rag_vector_candidates=10,
        rag_keyword_candidates=10,
        rag_rrf_k=60,
        rag_similarity_threshold=0.7,
    )

    # Vector score 0.2 is below cutoff, but FTS hit bypasses cutoff
    vector_hits = []
    keyword_hits = [_hit("chunk_k1", 0.2, 0.8)]

    monkeypatch.setattr(
        vector_store,
        "hybrid_search",
        AsyncMock(return_value=(vector_hits, keyword_hits)),
    )

    outcome = await service.search(query="query text", settings=config)
    assert outcome["found"] is True
    assert len(outcome["results"]) == 1
    assert outcome["results"][0]["chunk_id"] == "chunk_k1"
    # RRF score: 1.0 / (60 + 1) = 0.016393
    assert outcome["results"][0]["metadata"]["rrf_score"] == pytest.approx(0.016393, abs=1e-5)


@pytest.mark.asyncio
async def test_match_found_by_both_and_deduplicated(monkeypatch, mock_embedder):
    """Should verify that duplicate chunks returned by both search systems are merged and sum their ranks."""
    config = RagSettings(
        rag_vector_candidates=10,
        rag_keyword_candidates=10,
        rag_rrf_k=60,
        rag_similarity_threshold=0.5,
    )

    # chunk_both ranks 1st in vector and 1st in keyword
    vector_hits = [_hit("chunk_both", 0.8, 0.6)]
    keyword_hits = [_hit("chunk_both", 0.8, 0.6)]

    monkeypatch.setattr(
        vector_store,
        "hybrid_search",
        AsyncMock(return_value=(vector_hits, keyword_hits)),
    )

    outcome = await service.search(query="query text", settings=config)
    assert outcome["found"] is True
    # Should only return one result due to deduplication
    assert len(outcome["results"]) == 1
    assert outcome["results"][0]["chunk_id"] == "chunk_both"
    # RRF score: 1.0 / (60 + 1) + 1.0 / (60 + 1) = 0.032787
    assert outcome["results"][0]["metadata"]["rrf_score"] == pytest.approx(0.032787, abs=1e-5)


@pytest.mark.asyncio
async def test_ranking_after_score_fusion(monkeypatch, mock_embedder):
    """Should verify that RRF ranks chunks that appear in both searches higher than those in only one."""
    config = RagSettings(
        rag_vector_candidates=10,
        rag_keyword_candidates=10,
        rag_rrf_k=60,
        rag_similarity_threshold=0.3,
    )

    # Chunk A: Rank 1 in vector search. Not in keyword. RRF = 1 / (60+1) = 0.016393
    # Chunk B: Rank 2 in vector search, Rank 1 in keyword. RRF = 1/(60+2) + 1/(60+1) = 0.032522
    # So Chunk B must outrank Chunk A even though Chunk A has a higher vector score!
    vector_hits = [_hit("chunk_A", 0.70, 0.0), _hit("chunk_B", 0.65, 0.8)]
    keyword_hits = [_hit("chunk_B", 0.65, 0.8)]

    monkeypatch.setattr(
        vector_store,
        "hybrid_search",
        AsyncMock(return_value=(vector_hits, keyword_hits)),
    )

    outcome = await service.search(query="query text", settings=config)
    assert len(outcome["results"]) == 2
    assert outcome["results"][0]["chunk_id"] == "chunk_B"
    assert outcome["results"][0]["metadata"]["rrf_score"] == pytest.approx(0.032522, abs=1e-5)
    assert outcome["results"][1]["chunk_id"] == "chunk_A"
    assert outcome["results"][1]["metadata"]["rrf_score"] == pytest.approx(0.016393, abs=1e-5)


@pytest.mark.asyncio
async def test_empty_results_scenarios(monkeypatch, mock_embedder):
    """Should verify RRF handles empty results gracefully."""
    config = RagSettings(
        rag_vector_candidates=10,
        rag_keyword_candidates=10,
        rag_rrf_k=60,
        rag_similarity_threshold=0.5,
    )

    monkeypatch.setattr(
        vector_store,
        "hybrid_search",
        AsyncMock(return_value=([], [])),
    )

    outcome = await service.search(query="query text", settings=config)
    assert outcome["found"] is False
    assert len(outcome["results"]) == 0


@pytest.mark.asyncio
async def test_real_database_fts_matching():
    """Integration test checking that exact terms in actual database FTS matches correctly."""
    settings = get_rag_settings()
    if not settings.rag_database_url:
        pytest.skip("No database configured, skipping real FTS tests")

    pool = await vector_store.get_pool(settings)
    doc_id = str(uuid.uuid4())
    c1 = uuid.uuid4()
    c2 = uuid.uuid4()
    c3 = uuid.uuid4()
    c4 = uuid.uuid4()
    
    await pool.execute(f"DELETE FROM {vector_store._schema(settings)}.rag_documents WHERE id = $1", doc_id)
    await pool.execute(
        f"INSERT INTO {vector_store._schema(settings)}.rag_documents (id, filename, file_type) VALUES ($1, $2, $3)",
        doc_id, "test_fts.pdf", "pdf"
    )
    
    chunks = [
        (c1, doc_id, 0, "This candidate got a CGPA: 7.5+ in B.Tech Computer Science."),
        (c2, doc_id, 1, "Deep experience deploying workloads in AWS Cloud."),
        (c3, doc_id, 2, "Building premium responsive frontends with React.js framework."),
        (c4, doc_id, 3, "Completed a 6-month software engineering internship."),
    ]
    
    for cid, did, idx, content in chunks:
        embedding = [0.1] * settings.rag_embedding_dim
        await pool.execute(
            f"""
            INSERT INTO {vector_store._schema(settings)}.rag_chunks 
            (id, document_id, chunk_index, content, embedding, metadata)
            VALUES ($1, $2, $3, $4, $5::vector, '{{}}'::jsonb)
            """,
            cid, did, idx, content, vector_store.to_vector_literal(embedding)
        )
        
    try:
        test_cases = [
            ("CGPA", str(c1)),
            ("AWS", str(c2)),
            ("React.js", str(c3)),
            ("internship", str(c4)),
            ("What is Maaz’s CGPA?", str(c1))
        ]
        
        dummy_embedding = [0.1] * settings.rag_embedding_dim
        for query, expected_chunk_id in test_cases:
            vector_hits, keyword_hits = await vector_store.hybrid_search(
                dummy_embedding,
                query,
                vector_candidate_count=5,
                keyword_candidate_count=5,
                document_ids=[doc_id],
                settings=settings,
            )
            
            all_hits = vector_hits + keyword_hits
            matched_hit = next((h for h in all_hits if h.chunk_id == expected_chunk_id), None)
            
            assert matched_hit is not None, f"Query '{query}' did not retrieve chunk '{expected_chunk_id}'"
            assert matched_hit.keyword_score > 0.0, f"Query '{query}' matched chunk but keyword_score was {matched_hit.keyword_score}"
            
    finally:
        await pool.execute(f"DELETE FROM {vector_store._schema(settings)}.rag_chunks WHERE document_id = $1", doc_id)
        await pool.execute(f"DELETE FROM {vector_store._schema(settings)}.rag_documents WHERE id = $1", doc_id)

