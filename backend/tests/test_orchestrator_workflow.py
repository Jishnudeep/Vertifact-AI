"""ADK verification workflow: graph shape, cache routing and the per-agent degradation contract.

The agents run as dynamic children of the `dispatch_agents` supervisor, so one agent failing
or timing out must degrade only that agent: the healthy siblings finish and the run is scored
(CLAUDE.md §4.3). These tests pin that, because ADK's static fan-out would cancel the siblings.
"""
import asyncio
import time
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest
from app.agents import factcheck_agent, orchestrator
from app.agents.crossref_agent import CrossRefResult, make_neutral_crossref
from app.agents.factcheck_agent import make_neutral_factcheck
from app.agents.linguistic_agent import LinguisticResult, make_neutral_linguistic
from app.agents.orchestrator import AllAgentsFailedError, verify_claim
from app.config import settings
from app.utils.errors import ProviderErrorCode
from tests.agent_patches import patched_agents

CLAIM = "The Earth orbits the Sun"


def _ok_results():
    fc = make_neutral_factcheck(degraded=False)
    cr = CrossRefResult(entities_extracted=[], sources_found=[], coverage_distribution={
        "Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.0, degraded=False)
    ling = LinguisticResult(clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0,
                            informational_density=1.0, b_ling_score=0.0, degraded=False)
    return fc, cr, ling


def _patches(stack: ExitStack, fc_run, cr_run, ling_run, cached=None):
    stack.enter_context(patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384))
    stack.enter_context(patch("app.agents.orchestrator.lookup_cache", new=AsyncMock(return_value=cached)))
    stack.enter_context(patch("app.agents.orchestrator.insert_cache", new=AsyncMock(return_value="id-1")))
    # The agents are ADK nodes: their bodies are replaced, the node machinery around them is real.
    stack.enter_context(patched_agents(fc_run, cr_run, ling_run))


def test_graph_builds_with_expected_nodes():
    wf = orchestrator.build_verification_workflow()
    names = {n.name for n in wf.graph.nodes}
    assert {"cache_lookup", "format_cached", "dispatch_agents", "score"} <= names
    # The agents are dynamic children of the supervisor, never static graph nodes.
    assert not names & {"factcheck", "crossref", "linguistic"}


@pytest.mark.anyio
async def test_one_agent_raising_does_not_cancel_its_siblings():
    fc, cr, ling = _ok_results()
    finished = []

    async def slow_ok(result, name):
        await asyncio.sleep(0.2)          # still running when the fact-check agent fails
        finished.append(name)
        return result

    async def cr_run(claim):
        return await slow_ok(cr, "crossref")

    async def ling_run(claim):
        return await slow_ok(ling, "linguistic")

    with ExitStack() as stack:
        _patches(stack, AsyncMock(side_effect=RuntimeError("boom")), cr_run, ling_run)
        result = await verify_claim(CLAIM)

    assert sorted(finished) == ["crossref", "linguistic"]
    warnings = result["confidence"]["warnings"]
    assert any("Partial analysis: 2/3" in w for w in warnings)
    assert any(w == "Fact-check check failed" for w in warnings)
    assert result["cached"] is False


@pytest.mark.anyio
async def test_agent_exceeding_its_node_timeout_degrades_as_a_timeout(monkeypatch):
    """The agent node's `timeout=` cuts off a hung agent; ADK's NodeTimeoutError still classifies as TIMEOUT."""
    fc, cr, ling = _ok_results()

    async def hung(claim):
        await asyncio.sleep(30)

    monkeypatch.setattr(factcheck_agent.factcheck_node, "timeout", 0.1)
    with ExitStack() as stack:
        _patches(stack, hung, AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        result = await asyncio.wait_for(verify_claim(CLAIM), timeout=5)
        insert = orchestrator.insert_cache

    warnings = result["confidence"]["warnings"]
    assert any("Partial analysis: 2/3" in w for w in warnings)
    assert any(w == "Fact-check check timed out" for w in warnings)
    fc_out = insert.call_args.kwargs["agent_outputs"]["factcheck"]
    assert fc_out["error_details"]["code"] == ProviderErrorCode.TIMEOUT.value


@pytest.mark.anyio
async def test_client_disconnect_cancels_agents_and_leaks_no_session():
    started, cancelled = asyncio.Event(), []

    def tracking(name):
        async def run(claim):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
        return run

    with ExitStack() as stack:
        _patches(stack, tracking("factcheck"), tracking("crossref"), tracking("linguistic"))
        request = asyncio.create_task(verify_claim(CLAIM))
        await started.wait()
        await asyncio.sleep(0.05)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request

    assert sorted(cancelled) == ["crossref", "factcheck", "linguistic"]
    runner = orchestrator.get_verification_runner()
    sessions = await runner.session_service.list_sessions(
        app_name=orchestrator.WORKFLOW_APP_NAME, user_id=orchestrator._WORKFLOW_USER_ID)
    assert sessions.sessions == []


def test_missing_a2a_import_is_short_circuited():
    """ADK's build_node retries the optional A2A import on every ctx.run_node; it must fail fast."""
    import importlib.util
    import sys
    if importlib.util.find_spec("a2a") is None:
        assert "google.adk.agents.remote_a2a_agent" in sys.modules
        assert sys.modules["google.adk.agents.remote_a2a_agent"] is None


@pytest.mark.anyio
async def test_workflow_overhead_stays_small():
    """Regression guard: the graph must not add hundreds of ms of event-loop work per request."""
    fc, cr, ling = _ok_results()
    runs = 10
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        await verify_claim(CLAIM)                      # warm-up (first-run imports)
        start = time.perf_counter()
        for i in range(runs):
            await verify_claim(f"{CLAIM} run {i}")
        mean_ms = (time.perf_counter() - start) / runs * 1000
    # ~19 ms with the A2A import guard, ~70-580 ms without it (dev machine, 2026-10). The 80 ms
    # ceiling leaves room for slow CI runners while still catching the cold-import regression.
    assert mean_ms < 80, f"mean verify_claim overhead {mean_ms:.0f} ms with instant agents"


@pytest.mark.anyio
async def test_wrong_result_type_degrades_that_agent_only():
    fc, cr, ling = _ok_results()
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value={"not": "a result"}),
                 AsyncMock(return_value=ling))
        result = await verify_claim(CLAIM)
    assert any(w == "Cross-reference check failed" for w in result["confidence"]["warnings"])
    assert result["trust_score"]["v_consensus"] == 0.0


@pytest.mark.anyio
async def test_all_agents_failing_raises_never_scores():
    with ExitStack() as stack:
        _patches(stack,
                 AsyncMock(return_value=make_neutral_factcheck(degraded=True, error="x")),
                 AsyncMock(side_effect=RuntimeError("down")),
                 AsyncMock(return_value=make_neutral_linguistic(degraded=True, error="y")))
        with pytest.raises(AllAgentsFailedError):
            await verify_claim(CLAIM)


@pytest.mark.anyio
async def test_cache_hit_routes_around_the_agents():
    cached_row = {
        "claim_id": "cached-1", "extracted_claim": CLAIM, "c_total": 0.8, "s_fact": 1.0,
        "b_ling": 0.1, "v_consensus": 0.6, "v_refute": 0.0, "confidence_level": "high",
        "evidence_status": "fact_checked", "warnings": [], "source_citations": [],
    }
    fc_run = AsyncMock()
    with ExitStack() as stack:
        _patches(stack, fc_run, AsyncMock(), AsyncMock(), cached=cached_row)
        result = await verify_claim(CLAIM)
    assert result["cached"] is True and result["id"] == "cached-1"
    assert not fc_run.called


@pytest.mark.anyio
async def test_cache_lookup_error_is_a_miss():
    fc, cr, ling = _ok_results()
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        stack.enter_context(patch("app.agents.orchestrator.lookup_cache",
                                  new=AsyncMock(side_effect=RuntimeError("pgvector down"))))
        result = await verify_claim(CLAIM)
    assert result["cached"] is False


@pytest.mark.anyio
async def test_provider_error_classification_survives_the_node_wrapper():
    """The agent's own exception (not ADK's wrapper) is what gets classified."""
    from app.services.errors import EvidenceAPIError
    fc, cr, ling = _ok_results()
    quota = EvidenceAPIError("Gemini search returned HTTP 432", status_code=432, provider="gemini_search")
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(side_effect=quota), AsyncMock(return_value=ling))
        result = await verify_claim(CLAIM)
    assert any(w == "Cross-reference check quota exhausted" for w in result["confidence"]["warnings"])
    cr_status = next(a for a in result["agents"] if a["name"] == "crossref")
    assert cr_status["status"] == "degraded"
    assert cr_status["error_code"] == "QUOTA_EXHAUSTED"


@pytest.mark.anyio
async def test_concurrent_requests_use_isolated_sessions():
    fc, cr, ling = _ok_results()
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        results = await asyncio.gather(*(verify_claim(f"{CLAIM} number {i}") for i in range(5)))
    assert [r["claim"] for r in results] == [f"{CLAIM} number {i}" for i in range(5)]
    runner = orchestrator.get_verification_runner()
    sessions = await runner.session_service.list_sessions(
        app_name=orchestrator.WORKFLOW_APP_NAME, user_id=orchestrator._WORKFLOW_USER_ID)
    assert sessions.sessions == []   # every per-request session was deleted


@pytest.mark.anyio
async def test_search_suggestions_in_response():
    fc, cr, ling = _ok_results()
    fc.search_suggestions_html = "<div>Google Search Suggestions</div>"
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        insert_mock = AsyncMock(return_value="id-sugg")
        stack.enter_context(patch("app.agents.orchestrator.insert_cache", new=insert_mock))
        result = await verify_claim(CLAIM)
    assert result["search_suggestions_html"] == "<div>Google Search Suggestions</div>"
    assert insert_mock.called
    kwargs = insert_mock.call_args.kwargs
    assert kwargs["agent_outputs"]["search_suggestions_html"] == "<div>Google Search Suggestions</div>"


@pytest.mark.anyio
async def test_search_suggestions_survive_cache_hit():
    cached_row = {
        "claim_id": "cached-sugg-1",
        "extracted_claim": CLAIM,
        "c_total": 0.8,
        "s_fact": 1.0,
        "b_ling": 0.1,
        "v_consensus": 0.6,
        "v_refute": 0.0,
        "confidence_level": "high",
        "evidence_status": "fact_checked",
        "warnings": [],
        "source_citations": [],
        "agent_outputs": {
            "search_suggestions_html": "<div>Cached Suggestions</div>"
        },
    }
    with ExitStack() as stack:
        _patches(stack, AsyncMock(), AsyncMock(), AsyncMock(), cached=cached_row)
        result = await verify_claim(CLAIM)
    assert result["cached"] is True
    assert result["search_suggestions_html"] == "<div>Cached Suggestions</div>"


@pytest.mark.anyio
async def test_response_has_agent_status_and_durations():
    fc, cr, ling = _ok_results()
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        result = await verify_claim(CLAIM)
    agents = result.get("agents")
    assert agents is not None
    assert len(agents) == 3
    assert [a["name"] for a in agents] == ["factcheck", "crossref", "linguistic"]
    for a in agents:
        assert a["status"] == "ok"
        assert a["error_code"] is None
        assert isinstance(a["duration_ms"], int) and a["duration_ms"] >= 0


@pytest.mark.anyio
async def test_degraded_agent_status_and_code():
    fc, cr, ling = _ok_results()
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(side_effect=TimeoutError("timed out")), AsyncMock(return_value=ling))
        result = await verify_claim(CLAIM)
    agents = result["agents"]
    cr_agent = next(a for a in agents if a["name"] == "crossref")
    assert cr_agent["status"] == "degraded"
    assert cr_agent["error_code"] == "TIMEOUT"
    assert isinstance(cr_agent["duration_ms"], int) and cr_agent["duration_ms"] >= 0


@pytest.mark.anyio
async def test_unknown_error_text_never_in_warnings():
    fc, cr, ling = _ok_results()
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(side_effect=RuntimeError("secret-path /etc/x")))
        result = await verify_claim(CLAIM)
    warnings = result["confidence"]["warnings"]
    assert not any("secret-path" in w for w in warnings)
    assert any(w == "Linguistic check failed" for w in warnings)


@pytest.mark.anyio
async def test_found_by_marks_layer2_results_as_factcheck():
    fc, cr, ling = _ok_results()
    fc.all_matches = [{
        "url": "https://example.com/article",
        "claim_reviewed": "Article title",
        "publisher": "Example News",
        "rating": "Web Search Result",
        "domain": "example.com",
    }]
    cr.sources_found = [{
        "url": "https://reuters.com/news",
        "title": "Reuters Article",
        "publisher": "Reuters",
        "bias_label": "Center",
        "stance": "supports",
    }]
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        result = await verify_claim(CLAIM)
    sources = result["sources"]
    fc_src = next(s for s in sources if s["url"] == "https://example.com/article")
    cr_src = next(s for s in sources if s["url"] == "https://reuters.com/news")
    assert fc_src["type"] == "corroborating"
    assert fc_src["found_by"] == "factcheck"
    assert cr_src["found_by"] == "crossref"


@pytest.mark.anyio
async def test_cap_fields_in_response():
    fc, cr, ling = _ok_results()
    ling.degraded = True
    fc.s_fact_score = 0.9
    cr.v_consensus_score = 0.8
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(side_effect=RuntimeError("ling fail")))
        result = await verify_claim(CLAIM)
    ts = result["trust_score"]
    assert ts["cap_rule"] == "partial_analysis"
    assert ts["raw_c_total"] is not None
    assert ts["raw_c_total"] > ts["c_total"]
    assert ts["c_total"] == 0.50


@pytest.mark.anyio
async def test_agent_outputs_persist_durations_and_scoring():
    fc, cr, ling = _ok_results()
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        insert_mock = AsyncMock(return_value="id-persist")
        stack.enter_context(patch("app.agents.orchestrator.insert_cache", new=insert_mock))
        await verify_claim(CLAIM)
    assert insert_mock.called
    kwargs = insert_mock.call_args.kwargs
    agent_outputs = kwargs["agent_outputs"]
    assert "crossref" in agent_outputs
    assert isinstance(agent_outputs["crossref"]["duration_ms"], int)
    assert "scoring" in agent_outputs
    assert "cap_rule" in agent_outputs["scoring"]
    assert "raw_c_total" in agent_outputs["scoring"]


