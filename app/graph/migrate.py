"""Create the agent observability tables. Same shape as app/rag/migrate.py.

    python -m app.graph.migrate            # apply
    python -m app.graph.migrate --sql      # print, apply nothing

Additive and idempotent: it creates agent_runs, agent_steps, rag_searches,
rag_results, llm_calls and tool_calls in the SAME database and schema the RAG
tables already live in, and touches nothing that exists.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from ..rag.vector_store import close_pool
from . import store


async def _run() -> int:
    if not store.storage_configured():
        print(
            "RAG_DATABASE_URL (or DATABASE_URL) is not set. Nothing to migrate.\n"
            "The agent and its graph run without it - only the developer "
            "dashboard needs these tables.",
            file=sys.stderr,
        )
        return 2

    await store.migrate()
    print(
        "agent_runs, agent_steps, rag_searches, rag_results, llm_calls and "
        "tool_calls are ready."
    )
    await close_pool()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", action="store_true", help="print the SQL and exit")
    arguments = parser.parse_args()

    if arguments.sql:
        print(store.migration_sql())
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
