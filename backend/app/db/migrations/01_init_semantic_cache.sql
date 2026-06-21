-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Claims table
CREATE TABLE IF NOT EXISTS claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    input_type VARCHAR(10) NOT NULL CHECK (input_type IN ('text', 'url', 'image')),
    raw_input TEXT NOT NULL,
    extracted_claim TEXT NOT NULL,
    embedding vector(384),  -- dev config: 384 dimensions (all-MiniLM-L6-v2)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_claims_embedding ON claims
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Results table
CREATE TABLE IF NOT EXISTS results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    c_total FLOAT NOT NULL,
    s_fact FLOAT NOT NULL,
    b_ling FLOAT NOT NULL,
    v_consensus FLOAT NOT NULL,
    weights JSONB NOT NULL DEFAULT '{"w1": 0.5, "w2": 0.2, "w3": 0.3}',
    confidence_level VARCHAR(15) NOT NULL
        CHECK (confidence_level IN ('high', 'medium', 'low', 'insufficient')),
    agent_outputs JSONB,  -- Raw agent responses for debugging
    source_citations JSONB DEFAULT '[]',
    linguistic_profile JSONB,
    processing_time_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_results_claim_id ON results(claim_id);

-- Match Claims RPC Function
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
