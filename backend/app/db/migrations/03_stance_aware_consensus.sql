-- VeriFact AI - Migration 03: stance-aware consensus (C1) and evidence status (D2)
-- Spec: docs/superpowers/specs/2026-09-23-prd-tdd-review.md
-- Do NOT re-run after migration 04: it recreates the 2-argument match_claims next to 04's
-- 3-argument version, and PostgREST calls then fail as ambiguous (cache always misses).

-- 1. New result columns
ALTER TABLE results ADD COLUMN IF NOT EXISTS v_refute FLOAT NOT NULL DEFAULT 0;
ALTER TABLE results ADD COLUMN IF NOT EXISTS evidence_status VARCHAR(20)
    CHECK (evidence_status IN ('fact_checked', 'disputed', 'corroborated', 'unverified'));
-- Legacy rows keep evidence_status = NULL: they were scored with coverage-only consensus,
-- and lookup_cache treats them as cache misses so they are never served again.

-- 2. match_claims must return the new columns. The return type changes, so DROP first
--    (CREATE OR REPLACE cannot change a function's result columns).
DROP FUNCTION IF EXISTS match_claims(vector, double precision);

CREATE FUNCTION match_claims(
    query_embedding vector(384),
    similarity_threshold float
)
RETURNS TABLE (
    id UUID,
    claim_id UUID,
    c_total FLOAT,
    s_fact FLOAT,
    b_ling FLOAT,
    v_consensus FLOAT,
    v_refute FLOAT,
    evidence_status VARCHAR(20),
    weights JSONB,
    confidence_level VARCHAR(15),
    agent_outputs JSONB,
    source_citations JSONB,
    linguistic_profile JSONB,
    warnings JSONB,
    processing_time_ms INT,
    extracted_claim TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.claim_id,
        r.c_total,
        r.s_fact,
        r.b_ling,
        r.v_consensus,
        r.v_refute,
        r.evidence_status,
        r.weights,
        r.confidence_level,
        r.agent_outputs,
        r.source_citations,
        r.linguistic_profile,
        r.warnings,
        r.processing_time_ms,
        c.extracted_claim,
        (1 - (c.embedding <=> query_embedding))::float AS similarity
    FROM claims c
    JOIN results r ON r.claim_id = c.id
    WHERE (1 - (c.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY similarity DESC
    LIMIT 1;
END;
$$;
