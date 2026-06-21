import pytest
from unittest.mock import patch, MagicMock
from app.services.embedding import generate_embedding
from app.services.cache import lookup_cache, insert_cache
from app.db.connection import get_supabase

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
    """Verify lookup_cache handles database miss correctly offline."""
    mock_response = MagicMock()
    mock_response.data = []
    
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = mock_response
    
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        result = await lookup_cache([0.1]*384, similarity_threshold=0.95)
        assert result is None
        mock_supabase.rpc.assert_called_once_with(
            "match_claims",
            {"query_embedding": [0.1]*384, "similarity_threshold": 0.95}
        )

@pytest.mark.anyio
async def test_lookup_cache_offline_hit():
    """Verify lookup_cache handles database hit correctly offline."""
    mock_data = [{
        "claim_id": "test-uuid",
        "extracted_claim": "NASA water Mars",
        "similarity": 0.98,
        "c_total": 0.85
    }]
    mock_response = MagicMock()
    mock_response.data = mock_data
    
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = mock_response
    
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        result = await lookup_cache([0.1]*384, similarity_threshold=0.95)
        assert result is not None
        assert result["claim_id"] == "test-uuid"
        assert result["similarity"] == 0.98

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
    mock_data = [{
        "claim_id": "test-uuid",
        "extracted_claim": "NASA water Mars",
        "similarity": None,
        "c_total": 0.85
    }]
    mock_response = MagicMock()
    mock_response.data = mock_data
    
    mock_supabase = MagicMock()
    mock_supabase.rpc.return_value.execute.return_value = mock_response
    
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        result = await lookup_cache([0.1]*384, similarity_threshold=0.95)
        assert result is not None
        assert result["claim_id"] == "test-uuid"
        assert result["similarity"] is None

