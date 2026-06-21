import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.agents.crossref_agent import run, CrossRefResult, NEUTRAL
from app.config import settings

@pytest.mark.anyio
async def test_agent_degrades_on_timeout():
    """Verify that the agent returns NEUTRAL when it times out."""
    async def slow_search(*args, **kwargs):
        await asyncio.sleep(10.0)
        return []

    with patch("app.agents.crossref_agent.search_web", side_effect=slow_search):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await run("NASA announces Mars landing")
            assert result == NEUTRAL

@pytest.mark.anyio
async def test_agent_degrades_on_exception():
    """Verify that the agent returns NEUTRAL when search_web raises an exception."""
    with patch("app.agents.crossref_agent.search_web", side_effect=Exception("API key revoked")):
        result = await run("NASA announces Mars landing")
        assert result == NEUTRAL

@pytest.mark.anyio
async def test_factual_claim_cross_reference():
    """Verify entity extraction, bias lookup, and consensus scoring for a factual claim."""
    mock_tavily_results = [
        {
            "url": "https://www.nytimes.com/2026/06/20/us/politics/biden-climate.html",
            "title": "Biden announces new climate initiatives",
            "score": 0.92
        },
        {
            "url": "https://www.reuters.com/world/us/biden-unveils-climate-policy-2026.html",
            "title": "Reuters - President Biden targets carbon emissions",
            "score": 0.96
        },
        {
            "url": "https://www.foxnews.com/politics/biden-climate-rules-reaction.html",
            "title": "Fox News - GOP responds to Biden climate actions",
            "score": 0.82
        }
    ]

    with patch("app.agents.crossref_agent.search_web", return_value=mock_tavily_results) as mock_search:
        # Biden is a PERSON, climate represents concepts
        claim = "President Joe Biden announced new climate policies in Washington yesterday"
        result = await run(claim)
        
        assert mock_search.called
        # Check entities
        assert any("Biden" in ent for ent in result.entities_extracted)
        
        # Check sources mapping
        assert len(result.sources_found) == 3
        
        # NYT (Lean Left), Reuters (Center), Fox News (Right)
        nyt_src = next(s for s in result.sources_found if "nytimes.com" in s["url"])
        reuters_src = next(s for s in result.sources_found if "reuters.com" in s["url"])
        fox_src = next(s for s in result.sources_found if "foxnews.com" in s["url"])
        
        assert nyt_src["bias_label"] == "Lean Left"
        assert nyt_src["publisher"] == "New York Times (News)"
        
        assert reuters_src["bias_label"] == "Center"
        assert reuters_src["publisher"] == "Reuters"
        
        assert fox_src["bias_label"] == "Right"
        assert fox_src["publisher"] == "Fox News (Online News)"
        
        # Check distribution
        assert result.coverage_distribution["Lean Left"] == 1
        assert result.coverage_distribution["Center"] == 1
        assert result.coverage_distribution["Right"] == 1
        
        # Score calculation: count=3, avg_relevance = (0.92 + 0.96 + 0.82) / 3 = 0.9
        # v_consensus = min(3/5, 1.0) * 0.9 = 0.6 * 0.9 = 0.54
        assert result.v_consensus_score == 0.54

@pytest.mark.anyio
async def test_conspiracy_claim_cross_reference():
    """Verify that unrated/fringe domains result in zero consensus score."""
    mock_tavily_results = [
        {
            "url": "https://fringe-conspiracy-blog.org/2026/aliens.html",
            "title": "PROOF: Aliens are among us!",
            "score": 0.75
        }
    ]

    with patch("app.agents.crossref_agent.search_web", return_value=mock_tavily_results):
        claim = "Secret treaty signed between aliens and world leaders"
        result = await run(claim)
        
        assert len(result.sources_found) == 1
        src = result.sources_found[0]
        assert src["bias_label"] == "Unrated"
        assert src["publisher"] == "Fringe-conspiracy-blog" # Fallback capitalizing subdomain
        
        # All bias categories should be 0 because the source is Unrated
        assert all(count == 0 for count in result.coverage_distribution.values())
        
        # Consensus should be 0.0 because reputable source count is 0
        assert result.v_consensus_score == 0.0

@pytest.mark.anyio
async def test_live_cross_reference():
    """
    Live Integration Test.
    Queries Tavily live for a well-known topic and checks if it extracts sources correctly.
    """
    if not settings.tavily_api_key:
        pytest.skip("Skipping live integration tests because Tavily API key is not set.")

    # Query live with retry loop for rate limits
    for attempt in range(3):
        result = await run("NASA Artemis lunar landing mission")
        if len(result.sources_found) > 0:
            break
        await asyncio.sleep(6.0)
    else:
        result = await run("NASA Artemis lunar landing mission")
        assert len(result.sources_found) > 0

    assert len(result.entities_extracted) >= 0
    # Make sure we got at least one source
    assert len(result.sources_found) > 0
    
    # Standard check on first source
    first_src = result.sources_found[0]
    assert first_src["url"].startswith("http")
    assert len(first_src["title"]) > 0
    assert first_src["relevance_score"] >= 0.0
