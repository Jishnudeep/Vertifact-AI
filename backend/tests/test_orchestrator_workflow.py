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
from google.adk.workflow import FunctionNode

from app.agents import orchestrator
from app.agents.crossref_agent import CrossRefResult, make_neutral_crossref
from app.agents.factcheck_agent import make_neutral_factcheck
from app.agents.linguistic_agent import LinguisticResult, make_neutral_linguistic
from app.agents.orchestrator import AllAgentsFailedError, verify_claim
from app.config import settings
from app.utils.errors import ProviderErrorCode

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
    stack.enter_context(patch("app.agents.factcheck_agent.run", new=fc_run))
    stack.enter_context(patch("app.agents.crossref_agent.run", new=cr_run))
    stack.enter_context(patch("app.agents.linguistic_agent.run", new=ling_run))


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

    async def cr_run(claim, timeout=None):
        return await slow_ok(cr, "crossref")

    async def ling_run(claim, timeout=None):
        return await slow_ok(ling, "linguistic")

    with ExitStack() as stack:
        _patches(stack, AsyncMock(side_effect=RuntimeError("boom")), cr_run, ling_run)
        result = await verify_claim(CLAIM)

    assert sorted(finished) == ["crossref", "linguistic"]
    warnings = result["confidence"]["warnings"]
    assert any("Partial analysis: 2/3" in w for w in warnings)
    assert any(w.startswith("Fact-check agent degraded") for w in warnings)
    assert result["cached"] is False


@pytest.mark.anyio
async def test_agent_ignoring_its_budget_hits_the_adk_node_backstop(monkeypatch):
    """An agent that never honours its own timeout is cut off by the ADK node timeout."""
    fc, cr, ling = _ok_results()

    async def hung(claim, timeout=None):
        await asyncio.sleep(30)

    nodes = list(orchestrator.AGENT_NODES)
    _, provider, make_neutral = nodes[0]
    nodes[0] = (FunctionNode(func=orchestrator._factcheck, name="factcheck", timeout=0.1), provider, make_neutral)
    monkeypatch.setattr(orchestrator, "AGENT_NODES", tuple(nodes))

    with ExitStack() as stack:
        _patches(stack, hung, AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        result = await asyncio.wait_for(verify_claim(CLAIM), timeout=5)
        insert = orchestrator.insert_cache

    warnings = result["confidence"]["warnings"]
    assert any("Partial analysis: 2/3" in w for w in warnings)
    assert any("Fact-check agent degraded" in w and "timed out" in w for w in warnings)
    # ADK's NodeTimeoutError is not a TimeoutError subclass; it must still classify as TIMEOUT.
    fc_out = insert.call_args.kwargs["agent_outputs"]["factcheck"]
    assert fc_out["error_details"]["code"] == ProviderErrorCode.TIMEOUT.value


@pytest.mark.anyio
async def test_agent_ignoring_its_budget_is_cut_off_at_the_request_budget(monkeypatch):
    """The per-request backstop follows the deadline, not the fixed 9 s node guard."""
    monkeypatch.setattr(settings, "request_deadline_s", 2.0)   # -> agent budget 1.0 s
    fc, cr, ling = _ok_results()

    async def hung(claim, timeout=None):
        await asyncio.sleep(30)

    start = time.perf_counter()
    with ExitStack() as stack:
        _patches(stack, hung, AsyncMock(return_value=cr), AsyncMock(return_value=ling))
        result = await verify_claim(CLAIM)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0 + orchestrator.AGENT_GRACE_S + 1.0
    assert any("Fact-check agent degraded" in w and "timed out" in w
               for w in result["confidence"]["warnings"])


@pytest.mark.anyio
async def test_client_disconnect_cancels_agents_and_leaks_no_session():
    started, cancelled = asyncio.Event(), []

    def tracking(name):
        async def run(claim, timeout=None):
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
    assert any("Cross-reference agent degraded" in w for w in result["confidence"]["warnings"])
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
    quota = EvidenceAPIError("Tavily search returned HTTP 432", status_code=432, provider="tavily")
    with ExitStack() as stack:
        _patches(stack, AsyncMock(return_value=fc), AsyncMock(side_effect=quota), AsyncMock(return_value=ling))
        result = await verify_claim(CLAIM)
    assert any("Cross-reference agent degraded" in w and "(tavily)" in w
               for w in result["confidence"]["warnings"])


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
