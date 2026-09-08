-- VeriFact AI - Migration 02: Add bias_ratings, eval_runs, and results warnings
-- Implements TDD Section 5 schemas and adds warnings support to semantic cache results.

-- 1. Static bias ratings table (pre-loaded from AllSides CSV)
CREATE TABLE IF NOT EXISTS bias_ratings (
    domain VARCHAR(255) PRIMARY KEY,
    outlet_name VARCHAR(255) NOT NULL,
    allsides_rating VARCHAR(15) NOT NULL
        CHECK (allsides_rating IN ('Left', 'Lean Left', 'Center', 'Lean Right', 'Right')),
    last_updated DATE
);

-- 2. Evaluation logs table (for tracking eval runs)
CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name VARCHAR(255),
    dataset_name VARCHAR(100),
    total_claims INT,
    mae FLOAT,
    bucketed_accuracy FLOAT,
    spearman_rho FLOAT,
    weights_used JSONB,
    detailed_results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Add warnings column to results table if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'results' AND column_name = 'warnings'
    ) THEN
        ALTER TABLE results ADD COLUMN warnings JSONB DEFAULT '[]';
    END IF;
END $$;

-- 4. Update match_claims RPC function to include warnings
CREATE OR REPLACE FUNCTION match_claims(
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
