import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.agents.orchestrator import (
    verify_claim,
    InvalidInputError,
    NonVerifiableClaimError,
    AllAgentsFailedError
)
from app.agents.factcheck_agent import FactCheckResult, make_neutral_factcheck
from app.agents.crossref_agent import CrossRefResult, make_neutral_crossref
from app.agents.linguistic_agent import LinguisticResult, make_neutral_linguistic

@pytest.mark.anyio
async def test_verify_claim_cache_hit():
    """Verify that a cache hit returns immediately without invoking worker agents."""
    mock_cached = {
        "claim_id": "cached-uuid-123",
        "extracted_claim": "The Earth orbits the Sun",
        "c_total": 0.95,
        "s_fact": 1.0,
        "b_ling": 0.1,
        "v_consensus": 0.8,
        "confidence_level": "high",
        "warnings": [],
        "weights": {"w1": 0.5, "w2": 0.2, "w3": 0.3},
        "source_citations": [{"url": "https://snopes.com", "publisher": "Snopes", "type": "fact_check"}],
        "linguistic_profile": {"clickbait_syntax": 0.0, "sensationalism": 0.0, "emotional_framing": 0.1, "informational_density": 0.9}
    }

    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=mock_cached) as mock_lookup, \
         patch("app.agents.factcheck_agent.run") as mock_fc:
         
        result = await verify_claim("The Earth orbits the Sun")
        
        assert mock_lookup.called
        assert not mock_fc.called
        assert result["cached"] is True
        assert result["id"] == "cached-uuid-123"
        assert result["trust_score"]["c_total"] == 0.95
        assert result["confidence"]["level"] == "high"

@pytest.mark.anyio
async def test_verify_claim_cache_miss_pipeline_success():
    """Verify complete verification flow on cache miss when all agents succeed."""
    mock_fc = FactCheckResult(
        matches_found=1,
        best_match={"publisher": "PolitiFact", "rating": "True", "url": "https://politifact.com/1"},
        all_matches=[{"publisher": "PolitiFact", "rating": "True", "url": "https://politifact.com/1"}],
        s_fact_score=1.0,
        degraded=False
    )
    mock_cr = CrossRefResult(
        entities_extracted=["Earth", "Sun"],
        sources_found=[{"url": "https://reuters.com/1", "publisher": "Reuters", "bias_label": "Center", "relevance_score": 0.9}],
        coverage_distribution={"Left": 0, "Lean Left": 0, "Center": 1, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.8,
        degraded=False
    )
    mock_ling = LinguisticResult(
        clickbait_syntax=0.0,
        emotional_framing=0.1,
        sensationalism=0.0,
        informational_density=0.8,
        b_ling_score=0.15,
        degraded=False
    )

    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", return_value=mock_fc), \
         patch("app.agents.crossref_agent.run", return_value=mock_cr), \
         patch("app.agents.linguistic_agent.run", return_value=mock_ling), \
         patch("app.agents.orchestrator.insert_cache", return_value="new-claim-uuid-456") as mock_insert:

        result = await verify_claim("The Earth orbits the Sun")

        assert result["cached"] is False
        assert result["id"] == "new-claim-uuid-456"
        assert result["confidence"]["level"] == "high"
        assert result["trust_score"]["s_fact"] == 1.0
        assert result["trust_score"]["c_total"] > 0.8
        assert len(result["sources"]) == 2
        assert mock_insert.called

@pytest.mark.anyio
async def test_verify_claim_all_agents_fail_raises_error():
    """Verify CLAUDE.md Section 7 guardrail: all agents fail raises an error, NEVER returns a fabricated score."""
    mock_fc = make_neutral_factcheck(degraded=True, error="API timeout")
    mock_cr = make_neutral_crossref(degraded=True, error="Network failure")
    mock_ling = make_neutral_linguistic(degraded=True, error="Model crash")

    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", return_value=mock_fc), \
         patch("app.agents.crossref_agent.run", return_value=mock_cr), \
         patch("app.agents.linguistic_agent.run", return_value=mock_ling):

        with pytest.raises(AllAgentsFailedError) as exc_info:
            await verify_claim("NASA launched a rocket yesterday")
            
        assert "All verification agents failed" in str(exc_info.value)

@pytest.mark.anyio
async def test_verify_claim_partial_agent_degradation():
    """Verify that partial agent failure triggers confidence capping <= 0.50 and warnings."""
    # 2 agents succeed, 1 agent degrades
    mock_fc = FactCheckResult(matches_found=1, s_fact_score=1.0, degraded=False)
    mock_cr = make_neutral_crossref(degraded=True, error="Timeout")
    mock_ling = LinguisticResult(0.0, 0.0, 0.0, 0.8, 0.1, degraded=False)

    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", return_value=mock_fc), \
         patch("app.agents.crossref_agent.run", return_value=mock_cr), \
         patch("app.agents.linguistic_agent.run", return_value=mock_ling), \
         patch("app.agents.orchestrator.insert_cache", return_value="partial-uuid-789"):

        result = await verify_claim("NASA launched a rocket yesterday")

        assert result["trust_score"]["c_total"] <= 0.50
        assert any("Partial analysis" in w for w in result["confidence"]["warnings"])

@pytest.mark.anyio
async def test_verify_claim_rejects_empty_input():
    with pytest.raises(InvalidInputError):
        await verify_claim("   ")

@pytest.mark.anyio
async def test_verify_claim_rejects_opinion():
    with pytest.raises(NonVerifiableClaimError):
        await verify_claim("I think coffee is the greatest drink in human history.")
