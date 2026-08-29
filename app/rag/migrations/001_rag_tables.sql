-- RAG storage: documents, chunks and their embeddings.
--
-- Run through app/rag/migrate.py, which substitutes {{SCHEMA}}, {{DIM}} and
-- {{OPCLASS}} from the environment. The placeholders exist because a vector
-- column's width and an index's operator class are part of the DDL and cannot
-- be bound as parameters.
--
-- Every statement is idempotent. This migration ADDS two tables and touches
-- nothing that already exists in the database.

-- pgvector, only if it is not already there. CREATE EXTENSION IF NOT EXISTS is
-- a no-op against an existing installation and never re-creates or upgrades it.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {{SCHEMA}}.rag_documents (
    -- TEXT rather than UUID: a caller may supply its own document_id (a Laravel
    -- media id, a file hash) and should not have to make it a UUID first.
    id              TEXT PRIMARY KEY,
    filename        TEXT        NOT NULL,
    file_type       TEXT        NOT NULL,
    file_path       TEXT,
    user_id         TEXT,
    agent_id        TEXT,
    conversation_id TEXT,
    -- Free-form, caller-owned. JSONB so it can be filtered on later.
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    chunk_count     INTEGER     NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {{SCHEMA}}.rag_chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- ON DELETE CASCADE is the second guard behind the delete transaction: even
    -- a direct SQL delete of a document cannot leave orphaned chunks.
    document_id TEXT        NOT NULL REFERENCES {{SCHEMA}}.rag_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER     NOT NULL,
    content     TEXT        NOT NULL,
    embedding   VECTOR({{DIM}}) NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Scoping filters. Search may be restricted to one user/agent/conversation, and
-- without these that filter is a sequential scan over every chunk.
CREATE INDEX IF NOT EXISTS rag_documents_user_idx         ON {{SCHEMA}}.rag_documents (user_id);
CREATE INDEX IF NOT EXISTS rag_documents_agent_idx        ON {{SCHEMA}}.rag_documents (agent_id);
CREATE INDEX IF NOT EXISTS rag_documents_conversation_idx ON {{SCHEMA}}.rag_documents (conversation_id);
CREATE INDEX IF NOT EXISTS rag_documents_created_idx      ON {{SCHEMA}}.rag_documents (created_at DESC);
CREATE INDEX IF NOT EXISTS rag_chunks_document_idx        ON {{SCHEMA}}.rag_chunks (document_id);

-- The vector index.
--
-- Both pgvector index types are APPROXIMATE: they trade a little recall for a
-- large speed gain, and they only pay off once the table is big. Below roughly
-- ten thousand chunks an exact scan is both faster and perfectly recalled, so an
-- index here is not "free correctness" - it is a deliberate choice for scale.
-- Creating it early is still right: it costs nothing to maintain at small sizes
-- and the planner ignores it when a sequential scan wins.
--
-- HNSW where the server has it (pgvector >= 0.5): better recall/speed than
-- IVFFlat and, unlike IVFFlat, it needs no representative data at build time -
-- an IVFFlat index built on an empty table has useless centroids forever.
-- The opclass MUST match the distance operator the queries use; both come from
-- RAG_DISTANCE_METRIC so they cannot drift apart.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_am WHERE amname = 'hnsw') THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx '
                'ON {{SCHEMA}}.rag_chunks USING hnsw (embedding {{OPCLASS}})';
    ELSIF EXISTS (SELECT 1 FROM pg_am WHERE amname = 'ivfflat') THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx '
                'ON {{SCHEMA}}.rag_chunks USING ivfflat (embedding {{OPCLASS}}) '
                'WITH (lists = 100)';
    ELSE
        RAISE NOTICE 'No hnsw/ivfflat access method: searches run as exact scans.';
    END IF;
END
$$;
