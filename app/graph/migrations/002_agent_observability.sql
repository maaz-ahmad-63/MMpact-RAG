-- Agent observability: one run, its steps, and what each step actually did.
--
-- Run through app/graph/migrate.py, which substitutes {{SCHEMA}}. Placed in the
-- SAME database and schema as rag_documents/rag_chunks and opened through the
-- SAME pool - this task adds tables, it does not add a datastore.
--
-- Every statement is idempotent and additive. Nothing here touches rag_*, and
-- no foreign key points at a RAG table: a document may be deleted long after a
-- run cited it, and losing the run's history to that would defeat the purpose.
--
-- WHAT IS NOT STORED HERE
-- Chain of thought. There is no `reasoning`, `thoughts` or `deliberation`
-- column anywhere below, and adding one is out of scope by policy rather than
-- by oversight. Nor are API keys, bearer tokens or DSNs - `metadata` is written
-- only by graph/observability.py, which redacts before it writes.

CREATE TABLE IF NOT EXISTS {{SCHEMA}}.agent_runs (
    -- Caller-visible id, e.g. run_01JXYZ.... TEXT, matching rag_documents.id.
    run_id          TEXT PRIMARY KEY,
    conversation_id TEXT,
    user_id         TEXT,
    client_id       TEXT,
    user_message    TEXT,

    intent          TEXT,
    action          TEXT,
    confidence      DOUBLE PRECISION,
    requires_rag    BOOLEAN     NOT NULL DEFAULT FALSE,
    search_query    TEXT,

    status          TEXT        NOT NULL DEFAULT 'running',
    model_provider  TEXT,
    model_name      TEXT,
    -- Laravel's `text_llm` as sent, before services/models.resolve() ran.
    model_identifier TEXT,

    final_answer    TEXT,
    handoff_requested BOOLEAN   NOT NULL DEFAULT FALSE,

    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    latency_ms      INTEGER,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS {{SCHEMA}}.agent_steps (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT        NOT NULL
                 REFERENCES {{SCHEMA}}.agent_runs(run_id) ON DELETE CASCADE,
    step_number  INTEGER     NOT NULL,
    node_name    TEXT        NOT NULL,
    step_type    TEXT        NOT NULL DEFAULT 'node',
    status       TEXT        NOT NULL DEFAULT 'ok',
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    latency_ms   INTEGER,
    error        TEXT,
    -- Operational detail only. Redacted by the writer.
    metadata     JSONB       NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (run_id, step_number)
);

CREATE TABLE IF NOT EXISTS {{SCHEMA}}.rag_searches (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT        NOT NULL
                  REFERENCES {{SCHEMA}}.agent_runs(run_id) ON DELETE CASCADE,
    query         TEXT        NOT NULL,
    top_k         INTEGER,
    threshold     DOUBLE PRECISION,
    results_count INTEGER     NOT NULL DEFAULT 0,
    latency_ms    INTEGER,
    -- True for the retry after a query rewrite, so the dashboard can show both
    -- attempts and a developer can see which query actually worked.
    rewritten     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {{SCHEMA}}.rag_results (
    id          BIGSERIAL PRIMARY KEY,
    search_id   BIGINT      NOT NULL
                REFERENCES {{SCHEMA}}.rag_searches(id) ON DELETE CASCADE,
    run_id      TEXT        NOT NULL,
    -- Plain TEXT, NOT a foreign key onto rag_documents: deleting a document
    -- must not erase the record of a run that cited it.
    document_id TEXT,
    chunk_id    TEXT,
    score       DOUBLE PRECISION,
    source      TEXT,
    -- The retrieved passage, so "why did it answer that?" is answerable. Subject
    -- to the same auth as every other route on this service.
    content     TEXT,
    rank        INTEGER
);

CREATE TABLE IF NOT EXISTS {{SCHEMA}}.llm_calls (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT        NOT NULL
                  REFERENCES {{SCHEMA}}.agent_runs(run_id) ON DELETE CASCADE,
    node_name     TEXT        NOT NULL,
    provider      TEXT,
    -- The model actually called. Comes from the resolver, never from a literal.
    model         TEXT,
    input_tokens  INTEGER     NOT NULL DEFAULT 0,
    output_tokens INTEGER     NOT NULL DEFAULT 0,
    latency_ms    INTEGER,
    status        TEXT        NOT NULL DEFAULT 'ok',
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {{SCHEMA}}.tool_calls (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT        NOT NULL
               REFERENCES {{SCHEMA}}.agent_runs(run_id) ON DELETE CASCADE,
    tool_name  TEXT        NOT NULL,
    -- Redacted before the write - see observability.redact().
    arguments  JSONB       NOT NULL DEFAULT '{}'::jsonb,
    result     TEXT,
    status     TEXT        NOT NULL DEFAULT 'ok',
    latency_ms INTEGER,
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The dashboard's access patterns: newest runs first, filter by conversation,
-- user, status and action; then every child row for one run.
CREATE INDEX IF NOT EXISTS agent_runs_started_idx      ON {{SCHEMA}}.agent_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS agent_runs_conversation_idx ON {{SCHEMA}}.agent_runs (conversation_id);
CREATE INDEX IF NOT EXISTS agent_runs_user_idx         ON {{SCHEMA}}.agent_runs (user_id);
CREATE INDEX IF NOT EXISTS agent_runs_status_idx       ON {{SCHEMA}}.agent_runs (status);
CREATE INDEX IF NOT EXISTS agent_runs_action_idx       ON {{SCHEMA}}.agent_runs (action);
CREATE INDEX IF NOT EXISTS agent_steps_run_idx         ON {{SCHEMA}}.agent_steps (run_id, step_number);
CREATE INDEX IF NOT EXISTS rag_searches_run_idx        ON {{SCHEMA}}.rag_searches (run_id);
CREATE INDEX IF NOT EXISTS rag_results_search_idx      ON {{SCHEMA}}.rag_results (search_id);
CREATE INDEX IF NOT EXISTS llm_calls_run_idx           ON {{SCHEMA}}.llm_calls (run_id);
CREATE INDEX IF NOT EXISTS tool_calls_run_idx          ON {{SCHEMA}}.tool_calls (run_id);
