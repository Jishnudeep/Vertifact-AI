-- VeriFact AI - Migration 04: top-k cache candidates + created_at for TTL (C2)
-- Spec: docs/superpowers/specs/2026-09-23-prd-tdd-review.md
-- Requires migration 03 (v_refute, evidence_status).
-- If you re-run this (or any migration that recreates match_claims) after 06, re-run 06
-- afterwards so the recreated function is not executable by anon/authenticated.

-- The signature gains match_count and the result gains created_at: drop both old shapes.
DROP FUNCTION IF EXISTS match_claims(vector, double precision);
DROP FUNCTION IF EXISTS match_claims(vector, double precision, integer);

CREATE FUNCTION match_claims(
    query_embedding vector(384),
    similarity_threshold float,
    match_count int DEFAULT 3
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
    created_at TIMESTAMPTZ,
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
        r.created_at,
        c.extracted_claim,
        (1 - (c.embedding <=> query_embedding))::float AS similarity
    FROM claims c
    JOIN results r ON r.claim_id = c.id
    WHERE (1 - (c.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;
