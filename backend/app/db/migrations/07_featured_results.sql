-- Migration 07: Featured results for recent claims on homepage (Plan 7)
-- Add featured flag to results table and index for fast retrieval of newest featured results.

ALTER TABLE results ADD COLUMN IF NOT EXISTS featured BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_results_featured_created_at ON results (created_at DESC) WHERE featured;
