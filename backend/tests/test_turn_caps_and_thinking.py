"""
Tests for thinking configuration and model turn capping across ADK worker agents.
Spec: docs/superpowers/specs/2026-10-20-latency-thinking-budgets.md
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
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


def test_grounded_runner_thinking_config_matches_settings():
    adk_helpers._grounded_runner_cache.clear()
    try:
        # Pin a Gemini 3 model: a local .env may select a 2.5 search model.
        with patch.object(adk_helpers.settings, "search_model_id", "gemini-3.1-flash-lite"),                 patch.object(adk_helpers.settings, "grounded_search_thinking_level", "MINIMAL"):
            runner = adk_helpers.get_grounded_runner("coverage", "search instruction")
        assert runner.agent.generate_content_config.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    finally:
        adk_helpers._grounded_runner_cache.clear()


async def _agent_built_by(analyze, ctx_agent_output, monkeypatch, *args):
    """Run the real analyze() with a fake ctx.run_node and return the LlmAgent it built."""
    built = {}

    async def fake_run_node(agent, input_text, *a, **kw):
        built["agent"] = agent
        return ctx_agent_output

    ctx = AsyncMock(spec=Context)
    ctx.run_node = AsyncMock(side_effect=fake_run_node)
    await analyze(ctx, "Test claim", *args)
    return built["agent"]


@pytest.mark.parametrize(
    "model_id,expected",
    [("gemini-3.1-flash-lite", types.ThinkingLevel.LOW), ("gemini-2.5-flash-lite", None)],
)
async def test_factcheck_node_wires_thinking_config(monkeypatch, model_id, expected):
    """analyze() must pass the guarded thinking config to the real LlmAgent it builds."""
    monkeypatch.setattr(fc, "finalize", lambda claim, decision, ev: fc.make_neutral_factcheck())
    decision = fc.FactCheckDecision(evidence="none", relevant=False, rating="unproven", explanation="t")
    with patch.object(fc.settings, "model_id", model_id),             patch.object(fc.settings, "factcheck_thinking_level", "LOW"):
        agent = await _agent_built_by(fc.analyze, decision, monkeypatch)
    cfg = agent.generate_content_config
    level = cfg.thinking_config.thinking_level if cfg and cfg.thinking_config else None
    assert level == expected


@pytest.mark.parametrize(
    "model_id,expected",
    [("gemini-3.1-flash-lite", types.ThinkingLevel.LOW), ("gemini-2.5-flash-lite", None)],
)
async def test_crossref_node_wires_thinking_config(monkeypatch, model_id, expected):
    monkeypatch.setattr(cr, "_extract_entities", lambda claim: [])
    monkeypatch.setattr(cr, "finalize", lambda entities, cov: cr.make_neutral_crossref())
    with patch.object(cr.settings, "model_id", model_id),             patch.object(cr.settings, "crossref_thinking_level", "LOW"):
        agent = await _agent_built_by(cr.analyze, cr.CoverageSummary(queries_rationale="t"), monkeypatch)
    cfg = agent.generate_content_config
    level = cfg.thinking_config.thinking_level if cfg and cfg.thinking_config else None
    assert level == expected


@pytest.mark.parametrize("model_id", ["gemini-3.1-flash-lite", "gemini-3-pro-preview", "models/gemini-3.1-flash-lite",
                                      "projects/p/locations/l/publishers/google/models/gemini-3.1-flash-lite",
                                      "  Gemini-3.1-Flash-Lite "])
def test_thinking_config_sent_to_gemini_3_models(model_id):
    cfg = adk_helpers.thinking_content_config(model_id, "LOW")
    assert cfg is not None
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW


@pytest.mark.parametrize("model_id", ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash", "", None])
def test_thinking_level_never_sent_to_models_that_reject_it(model_id):
    """Gemini 2.5 rejects thinking_level with HTTP 400 (seen in the 2026-10-21 A.4 bench)."""
    assert adk_helpers.thinking_content_config(model_id, "MINIMAL") is None


def test_grounded_runner_omits_thinking_config_for_gemini_25_search_model():
    adk_helpers._grounded_runner_cache.clear()
    try:
        with patch.object(adk_helpers.settings, "search_model_id", "gemini-2.5-flash-lite"):
            runner = adk_helpers.get_grounded_runner("coverage", "search instruction")
        cfg = runner.agent.generate_content_config  # ADK normalizes None to an empty config
        assert cfg is None or cfg.thinking_config is None
    finally:
        adk_helpers._grounded_runner_cache.clear()


def test_dropped_thinking_level_is_logged_once(caplog):
    adk_helpers._warned_thinking_dropped.clear()
    with caplog.at_level("WARNING", logger=adk_helpers.logger.name):
        adk_helpers.thinking_content_config("gemini-2.5-flash-lite", "LOW")
        adk_helpers.thinking_content_config("gemini-2.5-flash-lite", "LOW")
    assert len([r for r in caplog.records if "not sent" in r.getMessage()]) == 1


def test_invalid_thinking_level_fails_at_settings_load():
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(factcheck_thinking_level="TURBO")
