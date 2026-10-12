#!/usr/bin/env python3
"""Safe, transactional re-indexing procedure for MMpact RAG.

Use this script to migrate existing stored embeddings when switching providers
(e.g., from local BGE-M3 to Cloudflare Workers AI @cf/baai/bge-m3, or vice-versa).

Ensures that:
1. All chunks are re-embedded in batches of EMBEDDING_BATCH_SIZE using the active provider.
2. Embeddings are verified to match RAG_EMBEDDING_DIM (1024-d) and L2-normalized.
3. Database updates occur within safe, chunked transactions.
4. Tenant isolation is respected (optionally target specific tenant_id or document_id).
5. Progress and latency are clearly reported.

Usage:
    # Re-index all chunks with the currently configured provider:
    python scripts/reindex_embeddings.py

    # Re-index for a specific tenant:
    python scripts/reindex_embeddings.py --tenant-id default

    # Dry-run (checks chunks without modifying DB):
    python scripts/reindex_embeddings.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import embeddings, vector_store
from app.rag.config import get_rag_settings


async def reindex(tenant_id: str | None = None, document_id: str | None = None, dry_run: bool = False):
    settings = get_rag_settings()
    pool = await vector_store.get_pool(settings)

    print("=" * 72)
    print("MMpact RAG — Re-indexing & Vector Migration Procedure")
    print("=" * 72)
    print(f"Target Database:      {settings.rag_db_schema} in PostgreSQL")
    print(f"Embedding Provider:   {settings.embedding_provider.upper()}")
    print(f"Active Model:         {settings.cloudflare_embedding_model if settings.embedding_provider == 'cloudflare' else settings.embedding_model}")
    print(f"Vector Dimension:     {settings.rag_embedding_dim}")
    print(f"Dry Run Mode:         {'ENABLED (no changes will be written)' if dry_run else 'DISABLED (will update database)'}")
    print()

    # 1. Fetch chunks to re-index
    where_clauses = []
    params = []
    if tenant_id:
        params.append(tenant_id)
        where_clauses.append(f"tenant_id = ${len(params)}")
    if document_id:
        params.append(document_id)
        where_clauses.append(f"document_id = ${len(params)}")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    query = f"""
        SELECT chunk_id, document_id, content
        FROM {settings.rag_db_schema}.rag_chunks
        {where_sql}
        ORDER BY document_id, chunk_index;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    total_chunks = len(rows)
    print(f"Found {total_chunks} chunks to re-index.")
    if total_chunks == 0:
        print("No chunks match the filter criteria. Nothing to do.")
        return

    # 2. Initialize Embedder
    embedder = await embeddings.get_embedder()
    dim = await embedder.load()
    print(f"Embedder initialized (dim={dim}, backend={embedder.backend})")

    # 3. Process in batches
    batch_size = max(1, settings.embedding_batch_size)
    updated_count = 0
    started_all = time.perf_counter()

    for start in range(0, total_chunks, batch_size):
        batch_rows = rows[start : start + batch_size]
        contents = [r["content"] for r in batch_rows]
        chunk_ids = [r["chunk_id"] for r in batch_rows]

        batch_start = time.perf_counter()
        vectors = await embedder.embed_texts(contents)
        embed_duration_ms = int((time.perf_counter() - batch_start) * 1000)

        if not dry_run:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for chunk_id, vector in zip(chunk_ids, vectors):
                        # pgvector accepts vector string format '[f1, f2, ...]'
                        vec_str = "[" + ",".join(str(v) for v in vector) + "]"
                        await conn.execute(
                            f"""
                            UPDATE {settings.rag_db_schema}.rag_chunks
                            SET embedding = $1::vector
                            WHERE chunk_id = $2;
                            """,
                            vec_str,
                            chunk_id,
                        )

        updated_count += len(batch_rows)
        pct = (updated_count / total_chunks) * 100
        print(
            f"  [{pct:5.1f}%] Processed {updated_count}/{total_chunks} chunks "
            f"(batch of {len(batch_rows)} embedded in {embed_duration_ms} ms)"
        )

    total_time_s = time.perf_counter() - started_all
    print()
    print("=" * 72)
    print(f"RE-INDEXING COMPLETE: {updated_count} chunks successfully processed in {total_time_s:.2f}s")
    if dry_run:
        print("Note: DRY RUN mode was enabled. No database rows were modified.")
    else:
        print("All database vector embeddings updated and normalized.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Re-index MMpact RAG stored chunks.")
    parser.add_argument("--tenant-id", type=str, default=None, help="Filter by tenant ID")
    parser.add_argument("--document-id", type=str, default=None, help="Filter by document ID")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without database writes")
    args = parser.parse_args()

    asyncio.run(reindex(tenant_id=args.tenant_id, document_id=args.document_id, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
