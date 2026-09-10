import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.agents.factcheck_agent import run, map_rating_to_score, FactCheckResult, make_neutral_factcheck
from app.config import settings

@pytest.mark.anyio
async def test_map_rating_to_score():
    # Test exact mapping
    assert map_rating_to_score("true") == 1.0
    assert map_rating_to_score("mostly false") == 0.25
    assert map_rating_to_score("pants on fire") == 0.0
    assert map_rating_to_score("unproven") == 0.5
    
    # Test substring/fallback mapping
    assert map_rating_to_score("This claim is false.") == 0.0
    assert map_rating_to_score("mostly incorrect info") == 0.25
    assert map_rating_to_score("entirely correct statement") == 1.0
    assert map_rating_to_score("mixture of truth and fiction") == 0.5
    assert map_rating_to_score("completely outdated") == 0.4
    assert map_rating_to_score(None) == 0.5
    assert map_rating_to_score("") == 0.5
    assert map_rating_to_score("random gibberish rating") == 0.5

@pytest.mark.anyio
async def test_agent_degrades_on_timeout():
    """Verify that the agent returns degraded neutral when the inner logic times out."""
    async def slow_analyze(claim):
        await asyncio.sleep(1.0)
        return FactCheckResult(matches_found=1, s_fact_score=1.0)
        
    with patch("app.agents.factcheck_agent._analyze", side_effect=slow_analyze), \
         patch("app.agents.factcheck_agent.TIMEOUT", 0.01):
        result = await run("Test claim")
        assert result.degraded is True
        assert result.s_fact_score == 0.5
        assert result.error is not None and "Timeout" in result.error

@pytest.mark.anyio
async def test_agent_degrades_on_exception():
    """Verify that the agent returns degraded neutral when an unexpected exception is raised."""
    with patch("app.agents.factcheck_agent._analyze", side_effect=Exception("Database crash")):
        result = await run("Test claim")
        assert result.degraded is True
        assert result.s_fact_score == 0.5
        assert result.error is not None and "Database crash" in result.error

@pytest.mark.anyio
async def test_google_factcheck_api_hit():
    """Verify Layer 1 flow when Google Fact Check API finds a direct match."""
    mock_google_response = [
        {
            "text": "The Earth is flat",
            "claimReview": [
                {
                    "url": "https://www.snopes.com/fact-check/flat-earth/",
                    "publisher": {"name": "Snopes"},
                    "textualRating": "False",
                    "reviewDate": "2026-01-01"
                }
            ]
        }
    ]
    
    with patch("app.agents.factcheck_agent.search_factchecks", return_value=mock_google_response) as mock_search:
        result = await run("The Earth is flat")
        assert mock_search.called
        assert result.degraded is False
        assert result.matches_found == 1
        assert result.s_fact_score == 0.0
        assert result.best_match["publisher"] == "Snopes"
        assert result.best_match["rating"] == "False"

@pytest.mark.anyio
async def test_tavily_fallback_with_llm():
    """Verify Layer 2 flow when Google Fact Check API misses but Tavily + LLM extracts rating."""
    mock_tavily_response = [
        {
            "url": "https://www.politifact.com/factcheck/earth-sun/",
            "title": "PolitiFact - Does the Earth orbit the Sun?",
            "content": "PolitiFact rating: True. The Earth does indeed orbit the Sun once every year.",
            "score": 0.95
        }
    ]
    
    mock_adk_response = '{"rating": "true", "explanation": "PolitiFact rates the claim as true.", "publisher": "PolitiFact"}'
    
    with patch("app.agents.factcheck_agent.search_factchecks", return_value=[]), \
         patch("app.agents.factcheck_agent.search_web", return_value=mock_tavily_response), \
         patch("app.agents.factcheck_agent.run_adk_agent", return_value=mock_adk_response):
         
        result = await run("The Earth orbits the Sun")
        assert result.degraded is False
        assert result.matches_found == 1
        assert result.s_fact_score == 1.0
        assert result.best_match["publisher"] == "Politifact"
        assert result.best_match["rating"] == "true"

@pytest.mark.anyio
async def test_tavily_fallback_invalid_enum_falls_back_to_unproven():
    """Verify that unrecognized LLM ratings fall back to unproven (0.5)."""
    mock_tavily_response = [
        {
            "url": "https://www.snopes.com/factcheck/test/",
            "title": "Snopes Check",
            "content": "Content",
            "score": 0.9
        }
    ]
    mock_adk_response = '{"rating": "inconclusive opinion", "explanation": "Not sure"}'
    
    with patch("app.agents.factcheck_agent.search_factchecks", return_value=[]), \
         patch("app.agents.factcheck_agent.search_web", return_value=mock_tavily_response), \
         patch("app.agents.factcheck_agent.run_adk_agent", return_value=mock_adk_response):
        result = await run("Some novel claim")
        assert result.s_fact_score == 0.5
        assert result.best_match["rating"] == "unproven"
        assert result.verdict_found is False

@pytest.mark.anyio
async def test_tavily_fallback_half_true_does_not_set_verdict_found():
    """Verify R2.B2 guardrail: Layer-2 'half true' rating does NOT set verdict_found to True."""
    mock_tavily_response = [
        {
            "url": "https://www.snopes.com/factcheck/test/",
            "title": "Snopes Check",
            "content": "Content",
            "score": 0.9
        }
    ]
    mock_adk_response = '{"rating": "half true", "explanation": "Mixed evidence"}'
    
    with patch("app.agents.factcheck_agent.search_factchecks", return_value=[]), \
         patch("app.agents.factcheck_agent.search_web", return_value=mock_tavily_response), \
         patch("app.agents.factcheck_agent.run_adk_agent", return_value=mock_adk_response):
        result = await run("Claim with mixed evidence")
        assert result.s_fact_score == 0.5
        assert result.verdict_found is False

@pytest.mark.anyio
async def test_tavily_fallback_non_factcheck_domain_does_not_set_verdict_found():
    """Verify R2.B2 guardrail: Definitive rating from non-factcheck domain does NOT set verdict_found to True."""
    mock_tavily_response = [
        {
            "url": "https://www.randomnewsblog.com/article/1",
            "title": "Blog Post",
            "content": "Content",
            "score": 0.9
        }
    ]
    mock_adk_response = '{"rating": "false", "explanation": "Blog says false"}'
    
    with patch("app.agents.factcheck_agent.search_factchecks", return_value=[]), \
         patch("app.agents.factcheck_agent.search_web", return_value=mock_tavily_response), \
         patch("app.agents.factcheck_agent.run_adk_agent", return_value=mock_adk_response):
        result = await run("Random claim")
        assert result.s_fact_score == 0.0
        assert result.verdict_found is False

@pytest.mark.anyio
@pytest.mark.live
async def test_live_factchecks():
    """
    Live Integration Test.
    Runs real queries on Google Fact Check Tools API and Tavily to verify end-to-end integration works.
    Only runs if API keys are configured (which they are in dev).
    """
    if not settings.google_fact_check_api_key or not settings.tavily_api_key:
        pytest.skip("Skipping live integration tests because API keys are not set.")

    # 1. Test flat Earth claim (known false claim) with retry for rate limits
    for attempt in range(3):
        flat_earth_res = await run("The Earth is flat")
        if flat_earth_res.matches_found > 0 and flat_earth_res.s_fact_score <= 0.25:
            break
        await asyncio.sleep(6.0)
    else:
        flat_earth_res = await run("The Earth is flat")
        assert flat_earth_res.matches_found > 0
        assert flat_earth_res.s_fact_score <= 0.25

    # 2. Test COVID-19 bleach cure claim (known false claim) with retry
    for attempt in range(3):
        bleach_res = await run("Drinking bleach cures COVID-19")
        if bleach_res.matches_found > 0 and bleach_res.s_fact_score <= 0.25:
            break
        await asyncio.sleep(6.0)
    else:
        bleach_res = await run("Drinking bleach cures COVID-19")
        assert bleach_res.matches_found > 0
        assert bleach_res.s_fact_score <= 0.25

    # 3. Test a generally true claim (should either hit fact-check or Tavily fallback) with retry
    for attempt in range(3):
        sun_res = await run("The Earth orbits the Sun")
        if sun_res.s_fact_score in [1.0, 0.5]:
            break
        await asyncio.sleep(6.0)
    else:
        sun_res = await run("The Earth orbits the Sun")
        assert sun_res.s_fact_score in [1.0, 0.5]
