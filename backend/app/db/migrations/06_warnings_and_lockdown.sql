-- VeriFact AI - Migration 06: restore results.warnings and lock down the public API roles
-- Requires migrations 04 and 05 (the REVOKEs name their function signatures). Safe to re-run.
--
-- 1. Migration 02 step 3 did not apply on the live project (its information_schema guard
--    had no table_schema filter), so match_claims (03/04) failed with 42703
--    "column r.warnings does not exist" and every insert_cache failed.
-- 2. The backend connects with the service-role key (app/db/connection.py), which bypasses
--    RLS and function grants. The frontend never talks to Supabase. So nothing legitimate
--    uses the anon/authenticated roles; close them off.

-- 1. results.warnings
ALTER TABLE public.results ADD COLUMN IF NOT EXISTS warnings JSONB DEFAULT '[]';

-- 2a. RLS on, no policies: anon/authenticated can read or write nothing; service role unaffected.
ALTER TABLE public.claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.bias_ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.eval_runs ENABLE ROW LEVEL SECURITY;

-- 2b. Functions default to EXECUTE for PUBLIC. purge_old_claims deletes data, so anyone with
--     the publishable key could otherwise call rpc/purge_old_claims(retention_days => 1).
REVOKE EXECUTE ON FUNCTION public.purge_old_claims(int) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.match_claims(vector, double precision, integer) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.purge_old_claims(int) TO service_role;
GRANT EXECUTE ON FUNCTION public.match_claims(vector, double precision, integer) TO service_role;

-- 2c. Supabase's default privileges grant EXECUTE on every NEW function in public to
--     anon/authenticated, so a later DROP + CREATE (e.g. the 384 -> 1024 embedding swap
--     recreating match_claims) would silently re-open it. Stop that for functions created
--     by this role (the SQL editor's). Tables created later still need RLS enabled by hand.
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC, anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO service_role;

NOTIFY pgrst, 'reload schema';
