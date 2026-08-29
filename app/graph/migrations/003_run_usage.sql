-- Token totals per run.
--
-- The per-CALL numbers already live in llm_calls. These are the per-RUN sums,
-- denormalised on purpose: the runs list and the overview page would otherwise
-- aggregate llm_calls on every page load, and a run's cost is the one number a
-- developer looks at first.
--
-- ADD COLUMN IF NOT EXISTS: idempotent, and it rewrites no existing row - the
-- defaults apply to rows already there. Nothing outside agent_runs is touched.

ALTER TABLE {{SCHEMA}}.agent_runs
    ADD COLUMN IF NOT EXISTS input_tokens  INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS output_tokens INTEGER NOT NULL DEFAULT 0;

-- total_tokens is GENERATED, not written: two columns and their sum can drift
-- if three separate writes maintain them, and dm_ai_logs.total_tokens must
-- always equal prompt + completion.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{{SCHEMA}}' AND table_name = 'agent_runs'
          AND column_name = 'total_tokens'
    ) THEN
        EXECUTE 'ALTER TABLE {{SCHEMA}}.agent_runs ADD COLUMN total_tokens INTEGER '
                'GENERATED ALWAYS AS (input_tokens + output_tokens) STORED';
    END IF;
END
$$;
