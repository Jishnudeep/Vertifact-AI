import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.agents.linguistic_agent import run, LinguisticResult, make_neutral_linguistic
from app.config import settings

@pytest.mark.anyio
async def test_agent_degrades_on_timeout():
    """Verify that the agent returns degraded result when it times out."""
    async def slow_analyze(claim):
        await asyncio.sleep(1.0)
        return LinguisticResult(0.0, 0.0, 0.0, 1.0, 0.0)
        
    with patch("app.agents.linguistic_agent._analyze", side_effect=slow_analyze), \
         patch("app.agents.linguistic_agent.TIMEOUT", 0.01):
        result = await run("Test claim")
        assert result.degraded is True
        assert result.b_ling_score == 0.5
        assert result.error is not None and "Timeout" in result.error

@pytest.mark.anyio
async def test_agent_degrades_on_exception():
    """Verify that the agent returns degraded result when an unexpected exception occurs."""
    with patch("app.agents.linguistic_agent._analyze", side_effect=Exception("spaCy model corrupted")):
        result = await run("Test claim")
        assert result.degraded is True
        assert result.b_ling_score == 0.5
        assert result.error is not None and "spaCy model corrupted" in result.error

@pytest.mark.anyio
async def test_factual_text_analysis():
    """Verify that objective/factual claims score low b_ling."""
    mock_adk_response = '{"sensationalism": 0.05}'
    
    with patch("app.config.settings.enable_llm_sensationalism", True), \
         patch("app.agents.linguistic_agent.run_adk_agent", return_value=mock_adk_response):
        claim = "GDP grew 2.3% in Q4 according to Bureau of Labor Statistics"
        result = await run(claim)
        
        assert result.degraded is False
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
    
    with patch("app.config.settings.enable_llm_sensationalism", True), \
         patch("app.agents.linguistic_agent.run_adk_agent", return_value=mock_adk_response):
        claim = "SHOCKING: You Won't BELIEVE What Scientists Just Found!!!"
        result = await run(claim)
        
        assert result.degraded is False
        assert result.clickbait_syntax >= 0.5
        assert result.sensationalism == 0.85
        assert result.informational_density < 0.5
        assert result.b_ling_score > 0.6
        assert any("Clickbait" in phrase or "CAPS" in phrase for phrase in result.flagged_phrases)

@pytest.mark.anyio
async def test_deterministic_heuristics_mode():
    """Verify that default heuristics mode is deterministic and does not call LLM."""
    with patch("app.config.settings.enable_llm_sensationalism", False), \
         patch("app.agents.linguistic_agent.run_adk_agent") as mock_adk:
        claim = "SHOCKING disaster and catastrophe hits the city!"
        res1 = await run(claim)
        res2 = await run(claim)
        
        assert not mock_adk.called
        assert res1.sensationalism_source == "heuristics"
        assert res1.sensationalism == res2.sensationalism
        assert res1.b_ling_score == res2.b_ling_score
        assert res1.sensationalism > 0.0

@pytest.mark.anyio
@pytest.mark.live
async def test_live_linguistic_analysis():
    """
    Live Integration Test.
    Runs the agent end-to-end against live Groq APIs to verify ADK sensationalism agent works.
    """
    with patch("app.config.settings.enable_llm_sensationalism", True):
        for attempt in range(3):
            factual_res = await run("The unemployment rate fell to 3.5% in March, the Labor Department reported Friday.")
            if factual_res.sensationalism < 0.3:
                break
            await asyncio.sleep(4.0)
        else:
            factual_res = await run("The unemployment rate fell to 3.5% in March, the Labor Department reported Friday.")
            assert factual_res.sensationalism < 0.3

        sensational_res = await run("SHOCKING: Aliens INVADE Washington! Scientists in TOTAL PANIC!")
        assert sensational_res.sensationalism > 0.6
        assert sensational_res.b_ling_score > factual_res.b_ling_score
