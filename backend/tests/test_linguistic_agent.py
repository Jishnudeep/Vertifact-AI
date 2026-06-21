import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.agents.linguistic_agent import run, LinguisticResult, NEUTRAL

@pytest.mark.anyio
async def test_agent_degrades_on_timeout():
    """Verify that the agent returns NEUTRAL when it times out."""
    async def slow_analyze(claim):
        await asyncio.sleep(10.0)
        return LinguisticResult(0.0, 0.0, 0.0, 1.0, 0.0)
        
    with patch("app.agents.linguistic_agent._analyze", side_effect=slow_analyze):
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await run("Test claim")
            assert result == NEUTRAL

@pytest.mark.anyio
async def test_agent_degrades_on_exception():
    """Verify that the agent returns NEUTRAL when an unexpected exception occurs."""
    with patch("app.agents.linguistic_agent._analyze", side_effect=Exception("spaCy model corrupted")):
        result = await run("Test claim")
        assert result == NEUTRAL

@pytest.mark.anyio
async def test_factual_text_analysis():
    """Verify that objective/factual claims score low b_ling."""
    mock_adk_response = '{"sensationalism": 0.05}'
    
    with patch("app.agents.linguistic_agent.run_adk_agent", return_value=mock_adk_response):
        claim = "GDP grew 2.3% in Q4 according to Bureau of Labor Statistics"
        result = await run(claim)
        
        assert result.clickbait_syntax == 0.0
        assert result.emotional_framing < 0.2
        assert result.sensationalism == 0.05
        assert result.informational_density > 0.3
        # Density inverted makes b_ling_score low
        assert result.b_ling_score < 0.3

@pytest.mark.anyio
async def test_sensational_text_analysis():
    """Verify that sensational claims score high b_ling."""
    mock_adk_response = '{"sensationalism": 0.85}'
    
    with patch("app.agents.linguistic_agent.run_adk_agent", return_value=mock_adk_response):
        claim = "SHOCKING: You Won't BELIEVE What Scientists Just Found!!!"
        result = await run(claim)
        
        assert result.clickbait_syntax >= 0.5
        assert result.sensationalism == 0.85
        assert result.informational_density < 0.3
        assert result.b_ling_score > 0.6
        assert any("Clickbait" in phrase or "CAPS" in phrase for phrase in result.flagged_phrases)

@pytest.mark.anyio
async def test_live_linguistic_analysis():
    """
    Live Integration Test.
    Runs the agent end-to-end against live Groq APIs to verify ADK sensationalism agent works.
    """
    # 1. Factual test with retry
    for attempt in range(3):
        factual_res = await run("The unemployment rate fell to 3.5% in March, the Labor Department reported Friday.")
        if factual_res.b_ling_score < 0.4:
            break
        await asyncio.sleep(6.0)
    else:
        factual_res = await run("The unemployment rate fell to 3.5% in March, the Labor Department reported Friday.")
        assert factual_res.b_ling_score < 0.4
    
    # 2. Sensational test with retry
    for attempt in range(3):
        sensational_res = await run("SHOCKING SECRET EXPOSED: This One Trick Will BLOW YOUR MIND!!! Must Watch Now!")
        if sensational_res.b_ling_score > 0.6 and len(sensational_res.flagged_phrases) > 0:
            break
        await asyncio.sleep(6.0)
    else:
        sensational_res = await run("SHOCKING SECRET EXPOSED: This One Trick Will BLOW YOUR MIND!!! Must Watch Now!")
        assert sensational_res.b_ling_score > 0.6
        assert len(sensational_res.flagged_phrases) > 0
