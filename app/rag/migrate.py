"""Create the RAG tables. Run it, or print the SQL and run that yourself.

    python -m app.rag.migrate            # apply
    python -m app.rag.migrate --sql      # print, apply nothing
    docker compose exec -T ai-agent python -m app.rag.migrate

Every statement is IF NOT EXISTS. It adds `rag_documents` and `rag_chunks` and
touches nothing else in the database - existing tables, the pgvector extension
if it is already installed, and anything the gateway owns are all left alone.

`--sql` exists because a deployment where the service's database role cannot
CREATE EXTENSION needs a DBA to run one line first; printing is then more useful
than failing.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import rag_settings
from .embeddings import get_embedder
from .vector_store import close_pool, column_dimension, migrate, migration_sql


async def _run(check_model: bool) -> int:
    if not rag_settings.database_configured:
        print(
            "RAG_DATABASE_URL (or DATABASE_URL) is not set. Nothing to migrate.",
            file=sys.stderr,
        )
        return 2

    if check_model:
        # Better to learn the real width now than to have Postgres reject every
        # insert later. Advisory only: an unreachable model must not stop the
        # tables from being created.
        try:
            embedder = await get_embedder()
            if embedder.dimension != rag_settings.rag_embedding_dim:
                print(
                    f"WARNING: {embedder.model_name} produces "
                    f"{embedder.dimension}-d vectors but RAG_EMBEDDING_DIM is "
                    f"{rag_settings.rag_embedding_dim}. Set RAG_EMBEDDING_DIM="
                    f"{embedder.dimension} before migrating.",
                    file=sys.stderr,
                )
                return 3
        except Exception as error:  # noqa: BLE001
            print(f"NOTE: could not check the model's width ({error}).", file=sys.stderr)

    await migrate()
    dimension = await column_dimension()
    print(
        f"rag_documents and rag_chunks are ready "
        f"(embedding vector({dimension}), metric={rag_settings.rag_distance_metric})."
    )
    await close_pool()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", action="store_true", help="print the SQL and exit")
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="do not load the embedding model to verify the vector width",
    )
    arguments = parser.parse_args()

    if arguments.sql:
        print(migration_sql())
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    return asyncio.run(_run(check_model=not arguments.skip_model_check))


if __name__ == "__main__":
    raise SystemExit(main())
