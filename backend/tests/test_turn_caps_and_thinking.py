"""
Tests for thinking configuration and model turn capping across ADK worker agents.
Spec: docs/superpowers/specs/2026-10-20-latency-thinking-budgets.md
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from app.agents import common
from app.agents import factcheck_agent as fc
from app.agents import crossref_agent as cr
from app.config import get_settings
from app.services.grounded_search import GroundedSearch, GroundedSource
from app.services.page_fetch import PageText
from app.utils import adk_helpers
from app.utils.errors import AgentTurnLimitExceeded, ProviderErrorCode, classify_provider_error
from tests.scripted_llm import call, final, install_script, run_node

pytestmark = pytest.mark.anyio


def test_turn_cap_lookup_by_agent_name():
    s = get_settings()
    assert common.get_agent_max_turns("factcheck_judge") == s.factcheck_max_turns
    assert common.get_agent_max_turns("crossref_searcher") == s.crossref_max_turns
    assert common.get_agent_max_turns("linguistic_judge") is None
    assert common.get_agent_max_turns(None) is None


def test_enforce_turn_cap_raises_on_exceeded():
    # Context for factcheck_judge (cap = 5)
    ctx = SimpleNamespace(agent_name="factcheck_judge")
    
    # 4 prior model messages -> turn 5 (allowed)
    contents_turn5 = [types.Content(role="model")] * 4
    req_turn5 = SimpleNamespace(contents=contents_turn5)
    assert common.enforce_turn_cap(ctx, req_turn5) is None

    # 5 prior model messages -> turn 6 (exceeds cap of 5)
    contents_turn6 = [types.Content(role="model")] * 5
    req_turn6 = SimpleNamespace(contents=contents_turn6)
    with pytest.raises(AgentTurnLimitExceeded) as exc_info:
        common.enforce_turn_cap(ctx, req_turn6)
    assert "factcheck_judge" in str(exc_info.value)
    assert "5" in str(exc_info.value)


async def test_factcheck_agent_enforces_turn_cap(monkeypatch):
    """Factcheck agent taking 6 turns raises AgentTurnLimitExceeded."""
    s = SimpleNamespace(
        db=AsyncMock(return_value=[]),
        search=AsyncMock(return_value=GroundedSearch([], [], None)),
        fetch=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(fc, "search_factchecks", s.db)
    monkeypatch.setattr(fc, "grounded_search", s.search)
    monkeypatch.setattr(fc, "fetch_page_text", s.fetch)

    # 5 tool calls then attempt a 6th turn
    steps = [
        call("search_fact_check_database", query=f"q{i}")
        for i in range(1, 6)
    ] + [
        final(evidence="none", source_id=None, relevant=False, rating="unproven", explanation="none")
    ]
    install_script(monkeypatch, *steps)

    with pytest.raises(AgentTurnLimitExceeded):
        await run_node(fc.factcheck_node, "Some claim")


async def test_crossref_agent_enforces_turn_cap(monkeypatch):
    """Cross-ref agent taking 5 turns raises AgentTurnLimitExceeded (cap=4)."""
    search_mock = AsyncMock(return_value=GroundedSearch([], [], None))
    monkeypatch.setattr(cr, "grounded_search", search_mock)
    monkeypatch.setattr(cr, "fetch_page_text", AsyncMock(return_value=None))

    # 4 tool calls then attempt a 5th turn
    steps = [
        call("search_news_coverage", query=f"q{i}")
        for i in range(1, 5)
    ] + [
        final(queries_rationale="done")
    ]
    install_script(monkeypatch, *steps)

    with pytest.raises(AgentTurnLimitExceeded):
        await run_node(cr.crossref_node, "Some claim")


async def test_supervisor_degrades_gracefully_on_turn_cap_exceeded(monkeypatch):
    """When a worker exceeds turn limit, orchestrator supervisor marks degraded=True, no crash."""
    from app.agents.orchestrator import dispatch_agents_node
    from app.agents.request import VerifyRequest

    # Scripted LLM that causes factcheck to exceed turn limit
    steps_fc = [
        call("search_fact_check_database", query=f"q{i}")
        for i in range(1, 6)
    ] + [
        final(evidence="none", source_id=None, relevant=False, rating="unproven", explanation="none")
    ]
    # Normal 2-turn step for crossref
    steps_cr = [
        call("search_news_coverage", query="q1"),
        final(queries_rationale="done"),
    ]

    routes = {
        "fact-check": steps_fc,
        "news": steps_cr,
    }
    install_script(monkeypatch, routes=routes)

    # Patch services
    monkeypatch.setattr(fc, "search_factchecks", AsyncMock(return_value=[]))
    monkeypatch.setattr(fc, "grounded_search", AsyncMock(return_value=GroundedSearch([], [], None)))
    monkeypatch.setattr(fc, "fetch_page_text", AsyncMock(return_value=None))
    monkeypatch.setattr(cr, "grounded_search", AsyncMock(return_value=GroundedSearch([], [], None)))
    monkeypatch.setattr(cr, "fetch_page_text", AsyncMock(return_value=None))

    from tests.scripted_llm import run_node
    # We can test via orchestrator's verify_claim
    from app.agents.orchestrator import verify_claim

    with patch("app.agents.orchestrator.lookup_cache", new_callable=AsyncMock, return_value=None), \
         patch("app.agents.orchestrator.insert_cache", new_callable=AsyncMock, return_value="mock-id-456"):
        outcome = await verify_claim("The Moon is made of cheese", bypass_cache=True)
        assert "trust_score" in outcome
        assert outcome["confidence"]["verdict_found"] is False
        warnings = outcome["confidence"]["warnings"]
        assert any("Partial analysis" in w for w in warnings)
        assert any("Fact-check hit its model-turn limit" in w for w in warnings)
        assert not any("Fact-check check timed out" in w for w in warnings)
        fc_agent = next(a for a in outcome["agents"] if a["name"] == "factcheck")
        assert fc_agent["status"] == "degraded"
        assert fc_agent["error_code"] == "TURN_LIMIT"


def test_agents_thinking_config():
    """Verify thinking config on LlmAgents matches settings."""
    s = get_settings()

    # Fact-check agent
    ev_fc = fc.Evidence()
    tools_fc = fc.build_tools("claim", ev_fc)
    agent_fc = fc.LlmAgent(
        name="factcheck_judge",
        model=common.get_model(),
        instruction=fc.get_instruction(),
        tools=tools_fc,
        output_schema=fc.FactCheckDecision,
        generate_content_config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=s.factcheck_thinking_level
            )
        ),
        **common.agent_kwargs(),
    )
    assert agent_fc.generate_content_config.thinking_config.thinking_level == types.ThinkingLevel.LOW

    # Cross-ref agent
    cov_cr = cr.Coverage()
    tools_cr = cr.build_tools("claim", cov_cr)
    agent_cr = cr.LlmAgent(
        name="crossref_searcher",
        model=common.get_model(),
        instruction=cr.get_instruction(),
        tools=tools_cr,
        output_schema=cr.CoverageSummary,
        generate_content_config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=s.crossref_thinking_level
            )
        ),
        **common.agent_kwargs(),
    )
    assert agent_cr.generate_content_config.thinking_config.thinking_level == types.ThinkingLevel.LOW

    # Grounded search runner
    runner = adk_helpers.get_grounded_runner("coverage", "search instruction")
    assert runner.agent.generate_content_config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
