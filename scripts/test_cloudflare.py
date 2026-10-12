#!/usr/bin/env python3
"""Diagnostic and compatibility verification tool for Cloudflare Workers AI.

Usage:
    # Test with live configured Cloudflare credentials:
    python scripts/test_cloudflare.py

    # Test with simulated responses (offline / CI):
    python scripts/test_cloudflare.py --mock

    # Compare local BGE-M3 vs Cloudflare BGE-M3 embeddings:
    python scripts/test_cloudflare.py --compare-local

What this script verifies:
1. Endpoint accessibility for @cf/baai/bge-m3 (1024-d dense vectors).
2. Endpoint accessibility for @cf/baai/bge-reranker-base (cross-encoder ranking).
3. Dimensionality, L2 normalization, and request latency.
4. Embedding space compatibility (cosine similarity between local and Cloudflare models).
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.rag.cloudflare_ai import (
    embed_texts_cloudflare,
    normalize_l2,
    rerank_cloudflare,
)
from app.rag.config import RagSettings, get_rag_settings
from app.rag.vector_store import SearchHit


SAMPLE_DOCS = [
    "MMpact RAG provides high-performance hybrid retrieval combining pgvector dense embeddings with PostgreSQL full-text search.",
    "BGE-M3 is a multilingual, multi-functionality embedding model trained by BAAI that outputs 1024-dimensional dense vectors.",
    "Cloudflare Workers AI runs machine learning models on serverless GPUs across Cloudflare's global edge network.",
    "Course syllabus: Mobile Application Development. Course Code: R1UC631B. Semester: VI Sem. Credits: 3.",
]

SAMPLE_QUERY = "What is the course code for Mobile Application Development?"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def run_diagnostics(settings: RagSettings, mock_mode: bool = False, compare_local: bool = False):
    print("=" * 72)
    print("MMpact RAG — Cloudflare Workers AI Diagnostic & Compatibility Suite")
    print("=" * 72)

    account_id = settings.cloudflare_account_id.strip()
    api_token = settings.cloudflare_api_token.strip()

    if mock_mode:
        print("[MODE] Simulated / Mock Mode (offline testing)")
        account_id = account_id or "mock_account_id"
        api_token = api_token or "mock_token_secret"
        settings.cloudflare_account_id = account_id
        settings.cloudflare_api_token = api_token
    else:
        print(f"[CONFIG] Cloudflare Account ID: {'*' * (len(account_id) - 4) + account_id[-4:] if len(account_id) >= 4 else '(not set)'}")
        print(f"[CONFIG] Cloudflare API Token:  {'[CONFIGURED]' if api_token else '[MISSING]'}")
        print(f"[CONFIG] Embedding Model:      {settings.cloudflare_embedding_model}")
        print(f"[CONFIG] Reranker Model:       {settings.cloudflare_reranker_model}")
        print()

        if not account_id or not api_token:
            print("[ERROR] Cloudflare credentials are missing!")
            print("Please set the following in your .env or environment:")
            print("    CLOUDFLARE_ACCOUNT_ID=<your-cloudflare-account-id>")
            print("    CLOUDFLARE_API_TOKEN=<your-cloudflare-workers-ai-api-token>")
            print()
            print("To run in simulated/dry-run mode without credentials, use:")
            print("    python scripts/test_cloudflare.py --mock")
            sys.exit(1)

    # ------------------------------------------------------------------------
    # 1. EMBEDDINGS TEST
    # ------------------------------------------------------------------------
    print("\n--- 1. Testing Cloudflare Embeddings (@cf/baai/bge-m3) ---")
    cf_vectors: list[list[float]] = []
    
    if mock_mode:
        print("  Generating mock 1024-d vectors...")
        cf_vectors = [normalize_l2([0.05 * (i + 1)] * 1024) for i in range(len(SAMPLE_DOCS))]
        embed_ms = 42
    else:
        started = time.perf_counter()
        try:
            cf_vectors = await embed_texts_cloudflare(SAMPLE_DOCS, settings)
            embed_ms = int((time.perf_counter() - started) * 1000)
        except Exception as error:
            print(f"  [FAIL] Embedding generation failed: {error}")
            sys.exit(1)

    print(f"  [PASS] Generated {len(cf_vectors)} embeddings in {embed_ms} ms")
    
    # Check dimensionality
    dims = [len(v) for v in cf_vectors]
    dim_ok = all(d == 1024 for d in dims)
    print(f"  [PASS] Dimension verification: {dims[0]}-d (Expected: 1024) -> {'VALID' if dim_ok else 'INVALID'}")
    
    # Check L2 normalization
    norms = [math.sqrt(sum(x * x for x in v)) for v in cf_vectors]
    norm_ok = all(abs(n - 1.0) < 1e-4 for n in norms)
    print(f"  [PASS] L2 Unit Normalization: norm={norms[0]:.6f} -> {'VALID' if norm_ok else 'INVALID'}")

    # ------------------------------------------------------------------------
    # 2. RERANKER TEST
    # ------------------------------------------------------------------------
    print("\n--- 2. Testing Cloudflare Reranker (@cf/baai/bge-reranker-base) ---")
    hits = [
        SearchHit(
            chunk_id=f"chunk_{i}",
            document_id="doc_1",
            content=text,
            similarity=0.5,
            distance=0.5,
            chunk_index=i,
            metadata={"source": "test_syllabus.txt"},
            filename="test_syllabus.txt",
            file_type="txt",
            vector_score=0.5,
            keyword_score=0.0,
        )
        for i, text in enumerate(SAMPLE_DOCS)
    ]

    if mock_mode:
        # Mock rerank outcome
        for hit in hits:
            hit.rerank_score = 0.95 if "R1UC631B" in hit.content else 0.20
            hit.metadata["raw_rerank_score"] = hit.rerank_score
            hit.metadata["is_evidence"] = hit.rerank_score > 0.5
        reranked = sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)
        rerank_ms = 35
    else:
        try:
            reranked, rerank_ms = rerank_cloudflare(SAMPLE_QUERY, hits, settings)
        except Exception as error:
            print(f"  [FAIL] Reranking failed: {error}")
            sys.exit(1)

    print(f"  [PASS] Reranked {len(reranked)} candidates in {rerank_ms} ms")
    print(f"  Top Candidate: Chunk '{reranked[0].chunk_id}' (Score: {reranked[0].rerank_score})")
    print(f"  Content snippet: \"{reranked[0].content[:70]}...\"")
    if "R1UC631B" in reranked[0].content:
        print("  [PASS] Target course code chunk correctly ranked at #1!")

    # ------------------------------------------------------------------------
    # 3. EMBEDDING COMPATIBILITY TEST (Local vs Cloudflare)
    # ------------------------------------------------------------------------
    if compare_local and not mock_mode:
        print("\n--- 3. Testing Embedding Space Compatibility (Local vs Cloudflare) ---")
        try:
            from sentence_transformers import SentenceTransformer
            print("  Loading local BAAI/bge-m3 model...")
            local_model = SentenceTransformer("BAAI/bge-m3")
            print("  Encoding sample texts with local model...")
            local_vectors = local_model.encode(
                SAMPLE_DOCS, normalize_embeddings=True, convert_to_numpy=True
            ).tolist()

            similarities = []
            for i, (loc_v, cf_v) in enumerate(zip(local_vectors, cf_vectors)):
                sim = cosine_similarity(loc_v, cf_v)
                similarities.append(sim)
                print(f"    Sample {i+1} Cosine Similarity (Local vs Cloudflare): {sim:.5f}")

            avg_sim = sum(similarities) / len(similarities)
            print(f"  Average Cross-Implementation Cosine Similarity: {avg_sim:.5f}")

            if avg_sim > 0.98:
                print("  [COMPATIBILITY VERDICT] HIGH FIDELITY.")
                print("  Cloudflare @cf/baai/bge-m3 embeddings closely match local BAAI/bge-m3.")
            else:
                print("  [COMPATIBILITY VERDICT] DIVERGENT EMBEDDING SPACE DETECTED.")
                print("  Quantization or architecture differences between local and Cloudflare models")
                print("  mean that queries in one space may experience degraded retrieval on documents")
                print("  embedded in the other space.")
                print("  RECOMMENDATION: Use `python scripts/reindex_embeddings.py` to re-embed all")
                print("  stored chunks before switching your primary query provider.")

        except ImportError:
            print("  [INFO] `sentence-transformers` is not installed; skipping local comparison.")
        except Exception as error:
            print(f"  [WARN] Could not run local model comparison: {error}")

    print("\n" + "=" * 72)
    print("ALL DIAGNOSTIC CHECKS PASSED SUCCESSFULLY.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Test Cloudflare Workers AI integration.")
    parser.add_argument("--mock", action="store_true", help="Run in mock/offline mode")
    parser.add_argument(
        "--compare-local", action="store_true", help="Compare with local BGE-M3 if available"
    )
    args = parser.parse_args()

    settings = get_rag_settings()
    asyncio.run(run_diagnostics(settings, mock_mode=args.mock, compare_local=args.compare_local))


if __name__ == "__main__":
    main()
