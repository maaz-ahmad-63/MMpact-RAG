"""RAG layer: document ingestion, embeddings, pgvector storage and retrieval.

This package is DELIBERATELY SELF-CONTAINED. It imports nothing from the agent,
the LLM providers, the tool dispatcher or the JSON store, and none of those
import it. The only wiring into the running service is one `include_router` line
in app/main.py; delete that line and the agent behaves exactly as it did before.

Why its own settings and its own database
-----------------------------------------
The service had no database (see app/store/base.py) - threads and prompt records
are JSON files, and app/config.py deliberately holds no DSN. Rather than push a
Postgres dependency into the shared configuration that every existing code path
reads, the RAG layer carries its own `RagSettings` and its own connection pool.
Nothing outside this package changes shape because retrieval exists.

Everything here is lazy. No pool is opened, no model is loaded and no import of
an optional dependency happens until a /api/rag route is actually called, so an
unconfigured deployment pays nothing and cannot fail at boot.
"""
