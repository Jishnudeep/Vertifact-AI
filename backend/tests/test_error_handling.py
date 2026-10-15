import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
import httpx

from app.utils.errors import (
    ProviderErrorCode,
    AgentErrorDetails,
    classify_provider_error
)
from app.services.errors import EvidenceAPIError
from app.agents import factcheck_agent, crossref_agent, linguistic_agent
from app.agents.orchestrator import verify_claim

try:
    import litellm
except ImportError:
    litellm = None

def test_classify_timeout():
    exc = asyncio.TimeoutError()
    details = classify_provider_error(exc, provider="factcheck_agent")
    assert details.code == ProviderErrorCode.TIMEOUT
    assert details.provider == "factcheck_agent"
    assert "timed out" in details.message

def test_classify_quota_exhausted():
    # 1. Direct EvidenceAPIError with status 432
    ev_exc = EvidenceAPIError("Search provider returned HTTP 432", status_code=432, provider="search_provider")
    details = classify_provider_error(ev_exc)
    assert details.code == ProviderErrorCode.QUOTA_EXHAUSTED
    assert details.provider == "search_provider"
    assert "quota exhausted" in details.message

    # 2. HTTPStatusError 432
    request = httpx.Request("POST", "https://api.example.com/search")
    response = httpx.Response(432, request=request)
    http_exc = httpx.HTTPStatusError("Client error '432'", request=request, response=response)
    details2 = classify_provider_error(http_exc, provider="search_provider")
    assert details2.code == ProviderErrorCode.QUOTA_EXHAUSTED
    assert details2.provider == "search_provider"

def test_classify_rate_limit_with_header():
    request = httpx.Request("GET", "https://api.example.com")
    response = httpx.Response(429, headers={"retry-after": "4.5"}, request=request)
    http_exc = httpx.HTTPStatusError("Client error '429'", request=request, response=response)
    details = classify_provider_error(http_exc, provider="google_factcheck")
    assert details.code == ProviderErrorCode.RATE_LIMIT
    assert details.provider == "google_factcheck"
    assert details.retry_after == 4.5

def test_classify_unconfigured_api_key():
    exc = EvidenceAPIError("Google Fact Check API key not configured", provider="google_factcheck")
    details = classify_provider_error(exc)
    assert details.code == ProviderErrorCode.AUTH_ERROR
    assert "not configured" in details.message

@pytest.mark.skipif(litellm is None, reason="litellm not installed")
def test_classify_litellm_errors():
    # RateLimitError with text retry
    rl_exc = litellm.RateLimitError(
        message="Rate limit reached. Please try again in 3.45s",
        model="groq/openai/gpt-oss-20b",
        llm_provider="groq"
    )
    details = classify_provider_error(rl_exc, provider="groq")
    assert details.code == ProviderErrorCode.RATE_LIMIT
    assert details.provider == "groq"
    assert details.retry_after == 3.45

    # NotFoundError
    nf_exc = litellm.NotFoundError(
        message="Model llama-3.1-8b-instant was not found",
        model="llama-3.1-8b-instant",
        llm_provider="groq"
    )
    details_nf = classify_provider_error(nf_exc, provider="groq")
    assert details_nf.code == ProviderErrorCode.MODEL_NOT_FOUND

@pytest.mark.anyio
async def test_run_adk_agent_rate_limit_retry():
    """Verify that run_adk_agent retries on RateLimitError and succeeds."""
    if litellm is None:
        pytest.skip("litellm not installed")

    from app.utils.adk_helpers import run_adk_agent

    mock_runner = MagicMock()
    
    # First attempt raises RateLimitError, second attempt yields a valid response
    rl_exc = litellm.RateLimitError(
        message="Rate limit reached. Please try again in 0.1s",
        model="groq/openai/gpt-oss-20b",
        llm_provider="groq"
    )
    rl_exc.retry_after = 0.05

    event_success = MagicMock()
    event_success.is_final_response.return_value = True
    part_mock = MagicMock()
    part_mock.text = '{"rating": "true", "explanation": "test"}'
    part_mock.thought = False
    event_success.content.parts = [part_mock]

    attempts = 0
    async def fake_run_async(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise rl_exc
        yield event_success

    mock_runner.run_async = fake_run_async

    with patch("app.utils.adk_helpers.get_adk_runner", return_value=mock_runner):
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            res = await run_adk_agent("test_agent", "instruction", "prompt", "test-app")
            assert '{"rating": "true"' in res
            assert attempts == 2
            mock_sleep.assert_awaited_once_with(0.05)

@pytest.mark.anyio
async def test_orchestrator_enriches_warnings_with_provider_error():
    """Verify that orchestrator surfaces specific provider degradation warnings."""
    # Mock factcheck to succeed
    fc_ok = factcheck_agent.FactCheckResult(
        matches_found=1,
        best_match={"publisher": "Snopes", "rating": "false"},
        all_matches=[],
        s_fact_score=0.0,
        search_queries_used=["test"],
        matched_review=True,
        verdict_found=True,
        degraded=False
    )
    # Mock crossref to degrade with search provider quota exhausted
    cr_degraded = crossref_agent.make_neutral_crossref(
        degraded=True,
        error="Evidence retrieval API unavailable: Search provider returned HTTP 432",
        error_details=AgentErrorDetails(
            code=ProviderErrorCode.QUOTA_EXHAUSTED,
            provider="gemini_search",
            message="Provider monthly search quota exhausted"
        )
    )
    # Mock linguistic to succeed
    ling_ok = linguistic_agent.LinguisticResult(
        clickbait_syntax=0.1,
        emotional_framing=0.1,
        sensationalism=0.1,
        informational_density=0.8,
        b_ling_score=0.15,
        flagged_phrases=[],
        degraded=False
    )

    with patch("app.agents.factcheck_agent.run", new_callable=AsyncMock, return_value=fc_ok), \
         patch("app.agents.crossref_agent.run", new_callable=AsyncMock, return_value=cr_degraded), \
         patch("app.agents.linguistic_agent.run", new_callable=AsyncMock, return_value=ling_ok), \
         patch("app.agents.orchestrator.lookup_cache", new_callable=AsyncMock, return_value=None), \
         patch("app.agents.orchestrator.insert_cache", new_callable=AsyncMock, return_value="mock-id-123"):
        
        result = await verify_claim("The Moon is made of green cheese", bypass_cache=True)
        
        warnings = result["confidence"]["warnings"]
        # Must have standard partial analysis warning
        assert any("Partial analysis" in w for w in warnings)
        # Must have specific provider degradation warning
        assert any("Cross-reference agent degraded: Provider monthly search quota exhausted (gemini_search)" in w for w in warnings)
