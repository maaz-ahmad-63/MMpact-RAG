import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from app.rag.config import RagSettings, get_rag_settings
from app.rag.vector_store import SearchHit

logger = logging.getLogger(__name__)

# Cached model instance
_model = None
_model_name = None

class RerankError(Exception):
    """Custom exception raised when local reranking fails or dependency is missing."""
    pass


# Target entity / slot patterns for query-to-chunk answerability evaluation
TARGET_SLOT_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "code": {
        "query_triggers": [
            r"\bcode\b",
            r"\bcodes\b",
            r"\bcourse\s*code\b",
            r"\bassigned\s+to\b",
            r"\b[a-zA-Z]{1,4}\d{2,6}[a-zA-Z0-9]*\b",
            r"\b\d{3,6}\b",
        ],
        "chunk_evidence": [
            r"course\s*code",
            r"\b[a-zA-Z]{1,4}\d{2,6}[a-zA-Z0-9]*\b",
            r"\bcode\s*[:\|]",
            r"\bcode\s+is\b",
        ],
    },
    "semester": {
        "query_triggers": [r"\bsem\b", r"\bsemester\b", r"\bsemesters\b", r"\boffered\s+in\b"],
        "chunk_evidence": [
            r"\b(?:I|II|III|IV|V|VI|VII|VIII|\d+(?:st|nd|rd|th))\s*sem\b",
            r"\bclasses\s+per\s+semester\b",
            r"\bsemester\s*[:\|]\s*\w+",
        ],
    },
    "type": {
        "query_triggers": [r"\bcourse\s*type\b", r"\btype\s+of\s+course\b", r"\bcategory\b", r"\bclassification\b"],
        "chunk_evidence": [
            r"course\s*type",
            r"\bintegrated\b",
            r"\btheory\s+course\b",
            r"\blab\s+course\b",
            r"\belective\b",
        ],
    },
    "credit": {
        "query_triggers": [r"\bcredit\b", r"\bcredits\b", r"\bcredit\s*hours\b"],
        "chunk_evidence": [
            r"\bcredits?\b",
            r"\bcredit\s*hours?\b",
            r"\bL-T-P\b",
            r"\bL-T-P-C\b",
        ],
    },
    "prerequisite": {
        "query_triggers": [r"\bprerequisite\b", r"\bprerequisites\b", r"\bpre-requisite\b", r"\beligibility\b"],
        "chunk_evidence": [
            r"prerequisite\s*course",
            r"\bprerequisites?\b",
            r"\beligibility\b",
            r"\bprior\s+knowledge\b",
        ],
    },
    "cgpa": {
        "query_triggers": [r"\bcgpa\b", r"\bgpa\b", r"\bpercentage\b", r"\bmarks\b", r"\bgrade\b"],
        "chunk_evidence": [
            r"\bcgpa\s*[:\|]?\s*[\d\.]+",
            r"\bgpa\s*[:\|]?\s*[\d\.]+",
            r"\b\d{2,3}%\b",
            r"\bgrade\b",
        ],
    },
    "fee": {
        "query_triggers": [r"\bfee\b", r"\bfees\b", r"\btuition\b", r"\bcost\b", r"\bprice\b"],
        "chunk_evidence": [
            r"\bfees?\b",
            r"\btuition\b",
            r"₹\s*[\d,]+",
            r"\$\s*[\d,]+",
            r"\brs\.?\s*[\d,]+",
        ],
    },
}



def evaluate_chunk_evidence(
    query: str,
    chunk_text: str,
    raw_rerank_score: float,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate whether candidate chunk contains concrete answer evidence for the query intent."""
    detected_targets: List[str] = []
    for target_name, patterns in TARGET_SLOT_PATTERNS.items():
        if any(re.search(pat, query, re.IGNORECASE) for pat in patterns["query_triggers"]):
            detected_targets.append(target_name)

    matched_targets: List[str] = []
    for target in detected_targets:
        if any(re.search(pat, chunk_text, re.IGNORECASE) for pat in TARGET_SLOT_PATTERNS[target]["chunk_evidence"]):
            matched_targets.append(target)

    # Floor guard: if semantic rerank score is near zero (< 0.02), chunk is completely off-topic
    if raw_rerank_score < 0.02:
        return raw_rerank_score, {
            "targets_queried": detected_targets,
            "targets_matched": [],
            "is_evidence": False,
        }

    # Modulate score based on target slot matching
    if detected_targets:
        target_coverage = len(matched_targets) / len(detected_targets)
        if target_coverage > 0:
            # Boost chunk providing exact answer slot
            final_score = raw_rerank_score * 0.6 + 0.4 * target_coverage
        else:
            # Topic match but completely missing requested slot/attribute
            final_score = raw_rerank_score * 0.20
    else:
        # Open-ended topical query
        final_score = raw_rerank_score

    final_score = max(0.0, min(1.0, final_score))
    return final_score, {
        "targets_queried": detected_targets,
        "targets_matched": matched_targets,
        "is_evidence": final_score >= 0.05,
    }



def get_reranker_model(model_name: str) -> Any:
    """Lazy-load and cache the local CrossEncoder reranking model."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RerankError(
            "The 'sentence-transformers' library is not installed or available. "
            "Please install it to use local reranking."
        ) from error

    try:
        logger.info("rag: initializing cross-encoder reranking model '%s'", model_name)
        _model = CrossEncoder(model_name)
        _model_name = model_name
        return _model
    except Exception as error:
        raise RerankError(f"Failed to load cross-encoder model '{model_name}': {error}") from error


def rerank(
    query: str,
    candidates: list[SearchHit],
    settings: RagSettings | None = None,
) -> tuple[list[SearchHit], int | None]:
    """Score and order the candidate chunks using the contextualized cross-encoder and answerability layer.

    Returns a tuple: (reranked_candidates, rerank_ms)
    """
    if not candidates:
        return [], 0

    config = settings or get_rag_settings()

    if config.reranker_provider == "cloudflare":
        from .cloudflare_ai import CloudflareAIError, rerank_cloudflare
        try:
            return rerank_cloudflare(query, candidates, settings=config)
        except Exception as error:
            logger.error("rag: cloudflare reranker failed: %s", error, exc_info=True)
            for hit in candidates:
                hit.rerank_score = None
            raise RerankError(f"Cloudflare reranking failed: {error}") from error

    # Local mode:
    # Reranking is computationally heavy, limit the number of candidates sent
    candidates_to_rerank = candidates[:config.rag_rerank_candidates]
    remaining_candidates = candidates[config.rag_rerank_candidates:]

    started = time.perf_counter()
    try:
        model = get_reranker_model(config.rag_rerank_model)
        
        # Batch query/chunk text pairs with document source context
        pairs = []
        for hit in candidates_to_rerank:
            source = hit.metadata.get("source") if hit.metadata else None
            contextualized_text = f"Document: {source}\n\n{hit.content}" if source else hit.content
            pairs.append((query, contextualized_text))

        raw_scores = model.predict(pairs)
        
        # Apply Answerability & Slot Evidence Evaluation Layer
        for hit, raw_score in zip(candidates_to_rerank, raw_scores):
            raw_s = float(raw_score)
            final_ev_score, meta = evaluate_chunk_evidence(query, hit.content, raw_s)
            hit.rerank_score = round(final_ev_score, 6)
            if hit.metadata is None:
                hit.metadata = {}
            hit.metadata["raw_rerank_score"] = round(raw_s, 6)
            hit.metadata["matched_targets"] = meta.get("targets_matched", [])
            hit.metadata["is_evidence"] = meta.get("is_evidence", False)

        # Chunks not reranked get a 0.0 rerank score
        for hit in remaining_candidates:
            hit.rerank_score = 0.0

        # Sort all candidates by final rerank evidence score descending
        reranked = sorted(candidates_to_rerank, key=lambda h: h.rerank_score, reverse=True)
        # Append the remaining ones at the end
        reranked.extend(remaining_candidates)
        
        rerank_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "rag: reranked and evaluated %d candidates in %d ms (top_score=%.4f)",
            len(candidates_to_rerank),
            rerank_ms,
            reranked[0].rerank_score if reranked else 0.0,
        )
        return reranked, rerank_ms

    except Exception as error:
        logger.error("rag: reranker failed, falling back to RRF: %s", error, exc_info=True)
        # Clear rerank scores on failure
        for hit in candidates:
            hit.rerank_score = None
        raise RerankError(str(error)) from error
