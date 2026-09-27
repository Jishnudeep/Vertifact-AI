-- VeriFact AI - Migration 05: data retention (H4)
-- No accounts exist; inputs (including OCR text from screenshots) are kept at most
-- retention_days, then deleted with their results. GET /api/results/{id} links expire with them.

CREATE OR REPLACE FUNCTION purge_old_claims(retention_days int DEFAULT 90)
RETURNS int
LANGUAGE plpgsql
AS $$
DECLARE
    removed int;
BEGIN
    IF retention_days IS NULL OR retention_days < 1 THEN
        RAISE EXCEPTION 'retention_days must be >= 1';
    END IF;
    DELETE FROM results
    WHERE claim_id IN (SELECT id FROM claims WHERE created_at < NOW() - make_interval(days => retention_days));
    DELETE FROM claims WHERE created_at < NOW() - make_interval(days => retention_days);
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed;
END;
$$;

-- Optional: daily schedule with pg_cron (Supabase: Database -> Extensions -> enable pg_cron).
-- SELECT cron.schedule('verifact-retention', '15 3 * * *', $cron$SELECT purge_old_claims(90)$cron$);
