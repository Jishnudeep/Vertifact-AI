import pytest
from unittest.mock import patch, MagicMock
from app.services.cache import lookup_cache, insert_cache

@pytest.mark.anyio
async def test_near_miss_similarity_misses_cache():
    """Verify that a near-miss claim (e.g. 0.92 similarity) misses the 0.95 cache threshold."""
    mock_supabase = MagicMock()
    # match_claims RPC returns empty list when below similarity_threshold
    mock_supabase.rpc.return_value.execute.return_value = MagicMock(data=[])

    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        result = await lookup_cache(
            embedding=[0.2] * 384,
            similarity_threshold=0.95
        )
        assert result is None
        mock_supabase.rpc.assert_called_once_with(
            "match_claims",
            {"query_embedding": [0.2] * 384, "similarity_threshold": 0.95}
        )

@pytest.mark.anyio
async def test_warnings_survive_cache_roundtrip():
    """Verify that confidence warnings are persisted and retrieved from cache."""
    mock_claim_resp = MagicMock(data=[{"id": "claim-uuid-warnings"}])
    mock_result_resp = MagicMock(data=[{"id": "result-uuid-warnings"}])
    
    mock_supabase = MagicMock()
    mock_supabase.table.return_value.insert.return_value.execute.side_effect = [
        mock_claim_resp,
        mock_result_resp
    ]
    
    test_warnings = [
        "Partial analysis: 2/3 agents succeeded",
        "Low confidence: no fact-checks found and insufficient corroborating sources"
    ]
    
    with patch("app.services.cache.get_supabase", return_value=mock_supabase):
        claim_id = await insert_cache(
            input_type="text",
            raw_input="Sample novel claim",
            extracted_claim="Sample novel claim",
            embedding=[0.1] * 384,
            c_total=0.50,
            s_fact=0.50,
            b_ling=0.20,
            v_consensus=0.10,
            weights={"w1": 0.5, "w2": 0.2, "w3": 0.3},
            confidence_level="low",
            agent_outputs={},
            source_citations=[],
            linguistic_profile={},
            processing_time_ms=120,
            warnings=test_warnings
        )
        assert claim_id == "claim-uuid-warnings"
        
        # Verify inserted result dict contains the warnings
        insert_call = mock_supabase.table.return_value.insert.call_args_list[1]
        assert insert_call[0][0]["warnings"] == test_warnings

@pytest.mark.anyio
async def test_cache_poisoning_zero_vector_rejected():
    """Verify that an all-zero vector is never inserted into the cache."""
    result = await insert_cache(
        input_type="text",
        raw_input="Poison attempt",
        extracted_claim="Poison attempt",
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
        processing_time_ms=50,
        warnings=[]
    )
    assert result is None
