"""Regression tests for RAG Answerability, Slot-Evidence Evaluation, and Reranking.

Verifies:
1. Course code queries rank chunk with the actual code (e.g. R1UC631B) #1.
2. Semester queries rank course pack chunk with semester details (VI Sem) #1, not the student's resume.
3. Course type queries rank the course type chunk (Integrated) #1.
4. Combined course code, type, and semester queries rank the multi-attribute chunk #1.
5. Unrelated resume chunks do not outrank course syllabus chunks for course-specific questions.
6. Queries with no answers in the documentation produce insufficient evidence rather than hallucinations.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import service as rag_service
from app.rag.config import get_rag_settings
from app.rag.llm import MockLLMProvider
from app.rag.reranker import evaluate_chunk_evidence, rerank
from app.rag.vector_store import SearchHit


# Unit tests for evidence evaluation layer
def test_evaluate_chunk_evidence_code_slot():
    query = "What is the code assigned to the Mobile Application Development course?"
    
    # Chunk with code
    chunk_with_code = "Course Title | Mobile Application Development | Course Code | R1UC631B | Class | B. Tech (CSE) VI Sem"
    score_with_code, meta_with_code = evaluate_chunk_evidence(query, chunk_with_code, 0.60)
    
    # Overview chunk without code
    chunk_without_code = "COURSE OVERVIEW The Mobile Application Development course is designed to provide learners with Android skills..."
    score_without_code, meta_without_code = evaluate_chunk_evidence(query, chunk_without_code, 0.85)

    assert "code" in meta_with_code["targets_matched"]
    assert "code" not in meta_without_code["targets_matched"]
    # Chunk with code is boosted above the generic overview
    assert score_with_code > score_without_code


def test_evaluate_chunk_evidence_semester_slot():
    query = "Which semester is Mobile Application Development offered in?"
    
    chunk_with_sem = "Course Code | R1UC631B | Class | B. Tech (CSE) VI Sem | Credits | 3"
    score_with_sem, meta_with_sem = evaluate_chunk_evidence(query, chunk_with_sem, 0.30)

    chunk_without_sem = "COURSE OVERVIEW The Mobile Application Development course is designed..."
    score_without_sem, meta_without_sem = evaluate_chunk_evidence(query, chunk_without_sem, 0.60)

    assert "semester" in meta_with_sem["targets_matched"]
    assert "semester" not in meta_without_sem["targets_matched"]
    assert score_with_sem > score_without_sem


def test_evaluate_chunk_evidence_off_topic_floor_guard():
    query = "What is the prerequisite for Quantum Cryptography 901?"
    chunk = "PREREQUISITE COURSE: Object Oriented Programming in Java"
    
    # When CrossEncoder gives near-zero score (off-topic document), do not boost
    score, meta = evaluate_chunk_evidence(query, chunk, 0.0001)
    assert score == 0.0001
    assert meta["is_evidence"] is False


@pytest.mark.asyncio
async def test_regression_1_course_code_retrieval():
    """Verify course code query returns chunk with R1UC631B as #1."""
    settings = get_rag_settings()
    res = await rag_service.search(
        query="What is the code assigned to the Mobile Application Development course?",
        top_k=5,
        threshold=0.0,
        settings=settings,
    )
    assert len(res["results"]) > 0
    top_result = res["results"][0]
    assert "R1UC631B" in top_result["content"]
    assert "Course Code" in top_result["content"] or "R1UC631B" in top_result["content"]


@pytest.mark.asyncio
async def test_regression_2_semester_retrieval():
    """Verify semester query ranks chunk with VI Sem as #1, outranking the resume."""
    settings = get_rag_settings()
    res = await rag_service.search(
        query="Which semester is Mobile Application Development offered in?",
        top_k=5,
        threshold=0.0,
        settings=settings,
    )
    assert len(res["results"]) > 0
    top_result = res["results"][0]
    # Must be from Course Pack, NOT Resume
    assert top_result["metadata"]["source"] != "Maaz Resume.pdf"
    assert any(term in top_result["content"] for term in ("Sem", "VI Sem", "Semester", "Classes per Semester"))


@pytest.mark.asyncio
async def test_regression_3_course_type_retrieval():
    """Verify course type query ranks chunk with Integrated as #1."""
    settings = get_rag_settings()
    res = await rag_service.search(
        query="What is the course type for Mobile Application Development?",
        top_k=5,
        threshold=0.0,
        settings=settings,
    )
    assert len(res["results"]) > 0
    top_result = res["results"][0]
    assert "Course Pack" in top_result["metadata"]["source"]
    assert any(t in top_result["content"].lower() for t in ("integrated", "course type", "theory"))


@pytest.mark.asyncio
async def test_regression_4_combined_attributes_retrieval():
    """Verify multi-attribute query ranks comprehensive chunk with code, type, semester #1."""
    settings = get_rag_settings()
    res = await rag_service.search(
        query="What is the course code, course type, and semester for Mobile Application Development?",
        top_k=5,
        threshold=0.0,
        settings=settings,
    )
    assert len(res["results"]) > 0
    top_result = res["results"][0]
    assert "R1UC631B" in top_result["content"]
    assert "VI Sem" in top_result["content"] or "Integrated" in top_result["content"]


@pytest.mark.asyncio
async def test_regression_5_unrelated_resume_chunk_demoted():
    """Verify that a course syllabus question is not dominated by an unrelated resume chunk."""
    settings = get_rag_settings()
    res = await rag_service.search(
        query="What are the modules in the Mobile Application Development course?",
        top_k=5,
        threshold=0.0,
        settings=settings,
    )
    assert len(res["results"]) > 0
    top_result = res["results"][0]
    assert top_result["metadata"]["source"] == "Course Pack-Mobile Application Developement-Updated.docx"


@pytest.mark.asyncio
async def test_regression_6_unanswerable_query_insufficient_evidence():
    """Verify unanswerable queries return no context and produce grounded missing-info response."""
    settings = get_rag_settings()
    mock_llm = MockLLMProvider()
    mock_llm.generate = AsyncMock(
        return_value=("I could not find information about that in the provided documents.", {"total_tokens": 30})
    )

    with patch("app.rag.service.llm.get_llm_provider", return_value=mock_llm):
        outcome = await rag_service.ask(
            query="What is the prerequisite for Quantum Cryptography 901?",
            settings=settings,
        )

        assert outcome["sources"] == []
        assert "could not find" in outcome["answer"].lower() or "no information" in outcome["answer"].lower()
