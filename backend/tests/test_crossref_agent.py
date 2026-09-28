import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.agents.crossref_agent import (
    run, CrossRefResult, make_neutral_crossref, label_stance, compute_consensus, STANCE_MIN_PROB
)
from app.services.errors import EvidenceAPIError
from app.services.bias_lookup import load_bias_ratings
from app.services.nli import NLIScores
from app.config import settings

@pytest.fixture(autouse=True)
def setup_bias():
    load_bias_ratings()

SUPPORT = NLIScores(entailment=0.9, contradiction=0.05, neutral=0.05)
REFUTE = NLIScores(entailment=0.05, contradiction=0.9, neutral=0.05)
NEUTRAL = NLIScores(entailment=0.2, contradiction=0.2, neutral=0.6)


@pytest.fixture(autouse=True)
def stub_nli():
    """Default: every source supports the claim. Tests override side_effect as needed."""
    def _all_support(pairs):
        return [SUPPORT for _ in pairs]
    with patch("app.agents.crossref_agent.classify_pairs", side_effect=_all_support) as mock_nli:
        yield mock_nli

@pytest.mark.anyio
async def test_agent_degrades_on_timeout():
    """Verify that the agent returns degraded result with v_consensus=0.0 when it times out."""
    async def slow_search(*args, **kwargs):
        await asyncio.sleep(1.0)
        return []

    with patch("app.agents.crossref_agent.search_web", side_effect=slow_search), \
         patch("app.agents.crossref_agent.TIMEOUT", 0.01):
        result = await run("NASA announces Mars landing")
        assert result.degraded is True
        assert result.v_consensus_score == 0.0
        assert result.error is not None and "Timeout" in result.error

@pytest.mark.anyio
async def test_agent_degrades_on_exception():
    """Verify that the agent returns degraded result with v_consensus=0.0 on exception."""
    with patch("app.agents.crossref_agent._analyze", side_effect=Exception("API key revoked")):
        result = await run("NASA announces Mars landing")
        assert result.degraded is True
        assert result.v_consensus_score == 0.0
        assert result.error is not None and "API key revoked" in result.error

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
        assert result.degraded is False
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
async def test_agent_degrades_when_search_api_is_down():
    """R7.B4: total Tavily outage must report degraded=True, not 'no corroborating sources'."""
    with patch("app.agents.crossref_agent.search_web", side_effect=EvidenceAPIError("Tavily search returned HTTP 432")):
        result = await run("NASA announces Mars landing")
        assert result.degraded is True
        assert result.v_consensus_score == 0.0
        assert result.sources_found == []
        assert result.error is not None and "unavailable" in result.error


@pytest.mark.anyio
async def test_agent_not_degraded_when_one_query_fails():
    """R7.B4: a partial search failure still yields real evidence and is NOT degradation."""
    good_results = [
        {
            "url": "https://www.reuters.com/science/mars-landing-2026.html",
            "title": "Reuters - Mars landing",
            "score": 0.9
        }
    ]
    calls = {"n": 0}

    async def flaky_search(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise EvidenceAPIError("Tavily search returned HTTP 432")
        return good_results

    with patch("app.agents.crossref_agent.search_web", side_effect=flaky_search):
        result = await run("NASA announces Mars landing")
        assert result.degraded is False
        assert len(result.sources_found) == 1


@pytest.mark.anyio
@pytest.mark.live
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


def test_label_stance_thresholds():
    assert label_stance(SUPPORT) == ("supports", 0.9)
    assert label_stance(REFUTE) == ("refutes", 0.9)
    assert label_stance(NEUTRAL)[0] == "neutral"
    # Below STANCE_MIN_PROB the model is not confident enough to take a side
    weak = NLIScores(entailment=STANCE_MIN_PROB - 0.01, contradiction=0.0, neutral=1 - (STANCE_MIN_PROB - 0.01))
    assert label_stance(weak)[0] == "neutral"


def test_compute_consensus_splits_support_and_refute():
    sources = [
        {"relevance_score": 0.9, "stance": "supports"},
        {"relevance_score": 0.8, "stance": "supports"},
        {"relevance_score": 0.7, "stance": "refutes"},
        {"relevance_score": 1.0, "stance": "neutral"},
    ]
    assert compute_consensus(sources) == (0.34, 0.14)


def test_compute_consensus_saturates_at_one():
    sources = [{"relevance_score": 1.0, "stance": "supports"} for _ in range(7)]
    assert compute_consensus(sources) == (1.0, 0.0)


@pytest.mark.anyio
async def test_refuting_coverage_does_not_raise_consensus(stub_nli):
    """C1: reputable outlets debunking a claim must NOT count as corroboration."""
    stub_nli.side_effect = lambda pairs: [REFUTE for _ in pairs]
    mock_results = [
        {"url": "https://www.reuters.com/fact-check/a", "title": "Claim is false", "content": "No evidence.", "score": 0.9},
        {"url": "https://www.foxnews.com/health/b", "title": "Experts reject claim", "content": "Debunked.", "score": 0.8},
        {"url": "https://www.nytimes.com/c", "title": "Viral post misleads", "content": "Untrue.", "score": 0.7},
    ]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        result = await run("Drinking bleach cures the flu")
    assert result.degraded is False
    assert result.v_consensus_score == 0.0
    assert result.v_refute_score == 0.48  # (0.9 + 0.8 + 0.7) / 5
    assert all(s["stance"] == "refutes" for s in result.sources_found)


@pytest.mark.anyio
async def test_premise_is_source_text_and_hypothesis_is_claim(stub_nli):
    mock_results = [{"url": "https://www.reuters.com/x", "title": "T", "content": "Body text", "score": 0.9}]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        await run("NASA announces Mars landing")
    pairs = stub_nli.call_args[0][0]
    assert pairs == [("T. Body text", "NASA announces Mars landing")]


@pytest.mark.anyio
async def test_claimreview_domains_excluded_from_consensus(stub_nli):
    """Fact-check sites restate the claim they debunk; they belong to the Fact-Check agent."""
    mock_results = [
        {"url": "https://www.politifact.com/factchecks/x", "title": "Claim: X", "content": "Rating: False", "score": 0.95},
    ]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        result = await run("Senator X voted against the bill")
    assert len(result.sources_found) == 1          # still shown to the user
    assert result.v_consensus_score == 0.0         # but never counted as corroboration
    assert result.v_refute_score == 0.0


@pytest.mark.anyio
async def test_claimreview_domains_carry_no_stance_label(stub_nli):
    """NLI misreads fact-check snippets, so their cards must not show a Supports/Refutes chip."""
    mock_results = [
        {"url": "https://www.politifact.com/factchecks/x", "title": "Claim: X", "content": "Rating: False", "score": 0.95},
        {"url": "https://www.reuters.com/x", "title": "T", "content": "C", "score": 0.9},
    ]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        result = await run("Senator X voted against the bill")
    by_url = {s["url"]: s for s in result.sources_found}
    assert by_url["https://www.politifact.com/factchecks/x"]["stance"] is None
    assert by_url["https://www.politifact.com/factchecks/x"]["stance_confidence"] is None
    assert by_url["https://www.reuters.com/x"]["stance"] == "supports"
    # Their stance is discarded, so they must not spend NLI time inside the agent budget.
    premises = [premise for premise, _ in stub_nli.call_args[0][0]]
    assert premises == ["T. C"]


@pytest.mark.anyio
async def test_only_claimreview_sources_needs_no_nli(stub_nli):
    """Nothing to score, so an NLI outage is irrelevant: zero consensus, not a degrade."""
    stub_nli.side_effect = RuntimeError("model file missing")
    mock_results = [
        {"url": "https://www.politifact.com/factchecks/x", "title": "Claim: X", "content": "Rating: False", "score": 0.95},
        {"url": "https://www.snopes.com/fact-check/x", "title": "Claim: X", "content": "Rating: False", "score": 0.9},
    ]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        result = await run("Senator X voted against the bill")
    assert not stub_nli.called
    assert result.degraded is False
    assert result.v_consensus_score == 0.0 and result.v_refute_score == 0.0
    assert len(result.sources_found) == 2


@pytest.mark.anyio
async def test_nli_failure_degrades_instead_of_falling_back(stub_nli):
    """Without stance, coverage cannot be read as agreement: degrade, never use the old count."""
    stub_nli.side_effect = RuntimeError("model file missing")
    mock_results = [{"url": "https://www.reuters.com/x", "title": "T", "content": "C", "score": 0.9}]
    with patch("app.agents.crossref_agent.search_web", return_value=mock_results):
        result = await run("NASA announces Mars landing")
    assert result.degraded is True
    assert result.v_consensus_score == 0.0
    assert result.v_refute_score == 0.0
    assert "Stance model unavailable" in result.error


@pytest.mark.anyio
async def test_neutral_degraded_result_has_zero_refute():
    assert make_neutral_crossref(degraded=True).v_refute_score == 0.0

