import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from app.services.embedding import generate_embedding
from app.services.cache import lookup_cache, insert_cache
from app.db.connection import get_supabase


def _servable_row(**overrides):
    row = {
        "claim_id": "test-uuid",
        "extracted_claim": "NASA water Mars",
        "similarity": 0.98,
        "c_total": 0.85,
        "confidence_level": "high",
        "evidence_status": "fact_checked",
        "agent_outputs": {
            "factcheck": {"degraded": False},
            "crossref": {"degraded": False},
            "linguistic": {"degraded": False},
        },
        "created_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    }
    row.update(overrides)
    return row


def _supabase_returning(rows):
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=rows)
    return mock_supabase


@pytest.mark.anyio
async def test_generate_embedding_fallback():
    """Verify generate_embedding handles empty strings and returns a zero vector."""
    result = generate_embedding("")
    assert len(result) == 384
    assert all(v == 0.0 for v in result)
    
    result_none = generate_embedding(None)
    assert len(result_none) == 384
    assert all(v == 0.0 for v in result_none)


@pytest.mark.anyio
async def test_lookup_cache_offline_miss():
    mock_supabase = _supabase_returning([])
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        result = await lookup_cache([0.1] * 384, claim_text="NASA water Mars", similarity_threshold=0.95)
    assert result is None
    mock_supabase.rpc.assert_called_once_with(
        "match_claims",
        {"query_embedding": [0.1] * 384, "similarity_threshold": 0.95, "match_count": 3},
    )


@pytest.mark.anyio
async def test_lookup_cache_rpc_error_is_miss():
    """A broken match_claims (e.g. 42703 missing column, as seen live 2026-09-27) must fail closed."""
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.side_effect = Exception("42703: column r.warnings does not exist")
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        result = await lookup_cache([0.1] * 384, claim_text="NASA water Mars")
    assert result is None


@pytest.mark.anyio
async def test_lookup_cache_offline_hit():
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning([_servable_row()])), \
         patch("app.services.cache.are_equivalent", return_value=True):
        result = await lookup_cache([0.1] * 384, claim_text="NASA water Mars")
    assert result is not None
    assert result["claim_id"] == "test-uuid"
    assert result["warnings"] == []


@pytest.mark.anyio
async def test_lookup_cache_legacy_row_without_evidence_status_is_miss():
    """Rows scored before stance-aware consensus must never be served (C1)."""
    rows = [_servable_row(evidence_status=None)]
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning(rows)), \
         patch("app.services.cache.are_equivalent", return_value=True):
        assert await lookup_cache([0.1] * 384, claim_text="NASA water Mars") is None


@pytest.mark.anyio
async def test_non_equivalent_top_match_is_skipped_for_next_candidate():
    """C2: the closest embedding may be the negation; a lower-ranked equivalent row is served."""
    rows = [
        _servable_row(claim_id="negated", extracted_claim="NASA did not find water on Mars", similarity=0.99),
        _servable_row(claim_id="same", extracted_claim="NASA found water on Mars", similarity=0.96),
    ]
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning(rows)), \
         patch("app.services.cache.are_equivalent", side_effect=[False, True]) as mock_eq:
        result = await lookup_cache([0.1] * 384, claim_text="NASA has found water on Mars")
    assert result["claim_id"] == "same"
    assert mock_eq.call_args_list[0][0] == ("NASA has found water on Mars", "NASA did not find water on Mars")


@pytest.mark.anyio
async def test_policy_runs_before_equivalence():
    """Unservable rows (e.g. partial runs) never pay for an NLI call."""
    partial = _servable_row(agent_outputs={"factcheck": {"degraded": False},
                                           "crossref": {"degraded": True},
                                           "linguistic": {"degraded": False}})
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning([partial])), \
         patch("app.services.cache.are_equivalent") as mock_eq:
        assert await lookup_cache([0.1] * 384, claim_text="NASA water Mars") is None
    mock_eq.assert_not_called()


@pytest.mark.anyio
async def test_equivalence_error_fails_closed():
    with patch("app.services.cache.get_supabase", return_value=_supabase_returning([_servable_row()])), \
         patch("app.services.cache.are_equivalent", side_effect=RuntimeError("NLI down")):
        assert await lookup_cache([0.1] * 384, claim_text="NASA water Mars") is None


@pytest.mark.anyio
async def test_insert_cache_offline_success():
    """Verify insert_cache performs claims and results inserts offline."""
    mock_claim_resp = MagicMock()
    mock_claim_resp.data = [{"id": "new-claim-uuid"}]
    
    mock_result_resp = MagicMock()
    mock_result_resp.data = [{"id": "new-result-uuid"}]
    
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.insert.return_value.execute.side_effect = [
        mock_claim_resp,
        mock_result_resp
    ]
    
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        claim_id = await insert_cache(
            input_type="text",
            raw_input="Raw text",
            extracted_claim="Extracted",
            embedding=[0.1]*384,
            c_total=0.8,
            s_fact=0.8,
            b_ling=0.1,
            v_consensus=0.7,
            weights={"w1": 0.5, "w2": 0.2, "w3": 0.3},
            confidence_level="high",
            agent_outputs={},
            source_citations=[],
            linguistic_profile={},
            processing_time_ms=100
        )
        assert claim_id == "new-claim-uuid"
        assert mock_supabase.table.call_count == 2

@pytest.mark.anyio
@pytest.mark.live
async def test_live_cache_integration():
    """
    Live Integration Test.
    Verifies embedding generation, cache insertion, exact match hits,
    similar match hits, different match misses, and database cleanup.
    """
    # 1. Generate embeddings
    text_a = "NASA scientists found liquid water on Mars today"
    text_b = "NASA has confirmed liquid water was found on Mars"
    text_c = "Stock prices fell due to inflation concerns"
    
    # Pre-clean database of orphaned test claims from prior failed runs
    supabase = get_supabase()
    old_claims = supabase.table("claims").select("id").in_("raw_input", [text_a, text_b, text_c]).execute()
    if old_claims.data:
        old_ids = [r["id"] for r in old_claims.data]
        supabase.table("results").delete().in_("claim_id", old_ids).execute()
        supabase.table("claims").delete().in_("id", old_ids).execute()
        
    emb_a = generate_embedding(text_a)
    emb_b = generate_embedding(text_b)
    emb_c = generate_embedding(text_c)
    
    # 2. Insert text_a into cache
    claim_id = await insert_cache(
        input_type="text",
        raw_input=text_a,
        extracted_claim=text_a,
        embedding=emb_a,
        c_total=0.85,
        s_fact=0.9,
        b_ling=0.1,
        v_consensus=0.8,
        weights={"w1": 0.5, "w2": 0.2, "w3": 0.3},
        confidence_level="high",
        agent_outputs={"note": "live test result"},
        source_citations=[],
        linguistic_profile={},
        processing_time_ms=1200
    )
    
    assert claim_id is not None
    
    try:
        # 3. Exact cache lookup (similarity should be >= 0.95)
        hit_exact = await lookup_cache(emb_a, similarity_threshold=0.95)
        assert hit_exact is not None
        assert hit_exact["claim_id"] == claim_id
        assert hit_exact["similarity"] >= 0.99
        
        # 4. Similar cache lookup
        # MiniLM cosine similarity for these two should be high.
        # Let's test at 0.85 threshold to ensure it hits.
        hit_similar = await lookup_cache(emb_b, similarity_threshold=0.85)
        assert hit_similar is not None
        assert hit_similar["claim_id"] == claim_id
        
        # 5. Different cache lookup (should miss at 0.80+)
        miss = await lookup_cache(emb_c, similarity_threshold=0.80)
        assert miss is None
        
    finally:
        # Clean up database: delete inserted results first (due to foreign key constraint), then claims
        supabase = get_supabase()
        supabase.table("results").delete().eq("claim_id", claim_id).execute()
        supabase.table("claims").delete().eq("id", claim_id).execute()

@pytest.mark.anyio
@pytest.mark.live
async def test_embedding_latency():
    """Verify that generating an embedding after warm-up takes less than 500ms."""
    import time
    
    # Warm-up call
    generate_embedding("warm up sentence")
    
    # Measured call
    start_time = time.time()
    generate_embedding("A representative claim for semantic caching latency verification.")
    duration_ms = (time.time() - start_time) * 1000
    
    print(f"Embedding latency: {duration_ms:.2f} ms")
    assert duration_ms < 500.0, f"Embedding generation took {duration_ms:.2f} ms, which is over 500ms"

@pytest.mark.anyio
async def test_lookup_cache_offline_hit_missing_similarity():
    """Verify lookup_cache handles database hit correctly when similarity is missing/None."""
    mock_data = [_servable_row(similarity=None)]
    mock_response = MagicMock()
    mock_response.data = mock_data
    
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = mock_response
    
    with patch("app.services.cache.get_supabase", return_value=mock_supabase), \
         patch("app.services.cache.are_equivalent", return_value=True):
        result = await lookup_cache([0.1]*384, claim_text="NASA water Mars", similarity_threshold=0.95)
        assert result is not None
        assert result["claim_id"] == "test-uuid"
        assert result["similarity"] is None

@pytest.mark.anyio
async def test_insert_cache_rejects_zero_vector():
    """Verify that attempting to cache a zero-vector is rejected."""
    result = await insert_cache(
        input_type="text",
        raw_input="Empty text",
        extracted_claim="Empty",
        embedding=[0.0] * 384,
        c_total=0.5,
        s_fact=0.5,
        b_ling=0.5,
        v_consensus=0.0,
        weights={"w1": 0.5, "w2": 0.2, "w3": 0.3},
        confidence_level="insufficient",
        agent_outputs={},
        source_citations=[],
        linguistic_profile={},
        processing_time_ms=10,
        warnings=["Test warning"]
    )
    assert result is None

@pytest.mark.anyio
async def test_insert_cache_persists_warnings():
    """Verify that warnings are forwarded to results table during insert."""
    mock_claim_resp = MagicMock()
    mock_claim_resp.data = [{"id": "claim-with-warnings"}]
    
    mock_result_resp = MagicMock()
    mock_result_resp.data = [{"id": "result-with-warnings"}]
    
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.insert.return_value.execute.side_effect = [
        mock_claim_resp,
        mock_result_resp
    ]
    
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        claim_id = await insert_cache(
            input_type="text",
            raw_input="Claim text",
            extracted_claim="Extracted",
            embedding=[0.1] * 384,
            c_total=0.55,
            s_fact=0.5,
            b_ling=0.1,
            v_consensus=0.2,
            weights={"w1": 0.5, "w2": 0.2, "w3": 0.3},
            confidence_level="low",
            agent_outputs={},
            source_citations=[],
            linguistic_profile={},
            processing_time_ms=100,
            warnings=["Low confidence: no fact-checks found"]
        )
        assert claim_id == "claim-with-warnings"
        # Check that result table insert received warnings
        results_insert_call = mock_supabase.table.return_value.insert.call_args_list[1]
        inserted_result_dict = results_insert_call[0][0]
        assert inserted_result_dict["warnings"] == ["Low confidence: no fact-checks found"]


@pytest.mark.anyio
async def test_insert_cache_persists_refute_and_evidence_status():
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.insert.return_value.execute.side_effect = [
        MagicMock(data=[{"id": "claim-1"}]),
        MagicMock(data=[{"id": "result-1"}]),
    ]
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        claim_id = await insert_cache(
            input_type="text", raw_input="c", extracted_claim="c", embedding=[0.1] * 384,
            c_total=0.33, s_fact=0.5, b_ling=0.0, v_consensus=0.0,
            weights={"w1": 0.5, "w2": 0.2, "w3": 0.3}, confidence_level="medium",
            agent_outputs={}, source_citations=[], linguistic_profile={}, processing_time_ms=10,
            warnings=[], v_refute=0.6, evidence_status="disputed",
        )
    assert claim_id == "claim-1"
    result_row = mock_supabase.table.return_value.insert.call_args_list[1][0][0]
    assert result_row["v_refute"] == 0.6
    assert result_row["evidence_status"] == "disputed"


def test_cache_hit_uses_shared_formatter():
    from app.api.result_format import format_stored_result
    row = _servable_row(
        id="results-row-999",
        claim_id="claim-uuid-111",
        c_total=0.85,
        s_fact=0.9,
        b_ling=0.1,
        v_consensus=0.8,
        confidence_level="high",
        evidence_status="fact_checked",
        warnings=[],
        source_citations=[],
        agent_outputs={
            "factcheck": {"degraded": False, "duration_ms": 100},
            "crossref": {"degraded": False, "duration_ms": 200},
            "linguistic": {"degraded": False, "duration_ms": 50},
            "scoring": {"raw_c_total": 0.85, "cap_rule": None},
        },
    )
    claim = "NASA water Mars"
    expected = format_stored_result(row, claim, cached=True)
    from app.agents.orchestrator import _format_cached_response
    actual = _format_cached_response(row, claim)
    assert actual == expected
    assert actual["id"] == "claim-uuid-111"


