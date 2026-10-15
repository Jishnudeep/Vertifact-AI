import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.agents import orchestrator
from app.agents.orchestrator import agent_budget, verify_claim
from app.agents.crossref_agent import CrossRefResult, run as crossref_run
from app.agents.factcheck_agent import make_neutral_factcheck
from app.agents.linguistic_agent import LinguisticResult
from app.config import settings
from app.services.grounded_search import GroundedSearch


@pytest.fixture(autouse=True)
def _fresh_background_tasks(monkeypatch):
    """Late writes from one test must not leak into another's assertions."""
    monkeypatch.setattr(orchestrator, "_background_tasks", set())


def test_agent_budget_bounds():
    assert agent_budget(9.5) == 8.0          # capped at the 8 s agent contract
    assert agent_budget(5.0) == 4.0          # leaves PERSIST_RESERVE_S for the DB write
    assert agent_budget(0.3) == 1.0          # never below MIN_AGENT_TIMEOUT


def _ok_results():
    fc = make_neutral_factcheck(degraded=False)
    cr = CrossRefResult(entities_extracted=[], sources_found=[], coverage_distribution={
        "Left": 0, "Lean Left": 0, "Center": 0, "Lean Right": 0, "Right": 0},
        v_consensus_score=0.0, degraded=False)
    ling = LinguisticResult(clickbait_syntax=0.0, emotional_framing=0.0, sensationalism=0.0,
                            informational_density=1.0, b_ling_score=0.0, degraded=False)
    return fc, cr, ling


@pytest.mark.anyio
async def test_agents_get_the_remaining_budget(monkeypatch):
    monkeypatch.setattr(settings, "request_deadline_s", 4.0)
    fc, cr, ling = _ok_results()
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)) as fc_run, \
         patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)), \
         patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)), \
         patch("app.agents.orchestrator.insert_cache", new=AsyncMock(return_value="id-1")):
        await verify_claim("The Earth orbits the Sun")
    timeout = fc_run.call_args.kwargs["timeout"]
    assert 1.0 <= timeout <= 3.0


@pytest.mark.anyio
async def test_slow_cache_lookup_counts_as_miss(monkeypatch):
    async def slow_lookup(*args, **kwargs):
        await asyncio.sleep(5)
    monkeypatch.setattr(orchestrator, "CACHE_LOOKUP_TIMEOUT_S", 0.05)
    fc, cr, ling = _ok_results()
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", side_effect=slow_lookup), \
         patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)) as fc_run, \
         patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)), \
         patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)), \
         patch("app.agents.orchestrator.insert_cache", new=AsyncMock(return_value="id-1")):
        result = await verify_claim("The Earth orbits the Sun")
    assert fc_run.called
    assert result["cached"] is False


@pytest.mark.anyio
async def test_slow_persistence_does_not_block_the_response(monkeypatch):
    monkeypatch.setattr(settings, "request_deadline_s", 1.0)
    async def slow_insert(**kwargs):
        await asyncio.sleep(3)
        return "late-id"
    fc, cr, ling = _ok_results()
    start = time.perf_counter()
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)), \
         patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)), \
         patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)), \
         patch("app.agents.orchestrator.insert_cache", side_effect=slow_insert):
        result = await verify_claim("The Earth orbits the Sun")
        assert result["id"] is None
        assert time.perf_counter() - start < 2.0
        await orchestrator.drain_background_tasks(timeout=5.0)  # no pending task at teardown


@pytest.mark.anyio
async def test_raw_input_is_persisted_sanitized():
    """H4: never store the unsanitized input (scripts, tracking params)."""
    fc, cr, ling = _ok_results()
    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)), \
         patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)), \
         patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)), \
         patch("app.agents.orchestrator.insert_cache", new=AsyncMock(return_value="id-1")) as insert:
        await verify_claim("NASA confirms <script>alert(1)</script>water on Mars")
    kwargs = insert.call_args.kwargs
    assert "<script>" not in kwargs["raw_input"]
    assert kwargs["raw_input"] == kwargs["extracted_claim"]


@pytest.mark.anyio
async def test_agent_honours_explicit_timeout():
    async def slow_search(*args, **kwargs):
        await asyncio.sleep(1.0)
        return GroundedSearch(sources=[], queries=[], search_suggestions_html=None)
    with patch("app.agents.crossref_agent.grounded_search", side_effect=slow_search):
        result = await crossref_run("NASA announces Mars landing", timeout=0.01)
    assert result.degraded is True and "Timeout" in result.error


def test_process_time_header():
    from app.main import app
    response = TestClient(app).get("/health")
    assert int(response.headers["X-Process-Time-Ms"]) >= 0


def test_process_time_header_is_readable_cross_origin():
    """The frontend runs on another origin; browsers hide non-exposed headers from JS."""
    from app.main import app
    response = TestClient(app).get("/health", headers={"Origin": settings.cors_origins[0]})
    exposed = response.headers.get("access-control-expose-headers", "")
    assert "x-process-time-ms" in exposed.lower()


def _slow_insert_patches(slow_insert):
    fc, cr, ling = _ok_results()
    return (
        patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384),
        patch("app.agents.orchestrator.lookup_cache", return_value=None),
        patch("app.agents.factcheck_agent.run", new=AsyncMock(return_value=fc)),
        patch("app.agents.crossref_agent.run", new=AsyncMock(return_value=cr)),
        patch("app.agents.linguistic_agent.run", new=AsyncMock(return_value=ling)),
        patch("app.agents.orchestrator.insert_cache", side_effect=slow_insert),
    )


@pytest.mark.anyio
async def test_late_write_is_tracked_then_released(monkeypatch):
    monkeypatch.setattr(settings, "request_deadline_s", 1.0)
    done = asyncio.Event()
    async def slow_insert(**kwargs):
        await asyncio.sleep(1.5)   # outlives the ~0.95 s left of the 1.0 s deadline
        done.set()
        return "late-id"
    p = _slow_insert_patches(slow_insert)
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        result = await verify_claim("The Earth orbits the Sun")
        assert result["id"] is None
        assert len(orchestrator._background_tasks) == 1
        await orchestrator.drain_background_tasks(timeout=3.0)
    assert done.is_set()
    assert len(orchestrator._background_tasks) == 0


@pytest.mark.anyio
async def test_cancelled_request_keeps_its_write_referenced(monkeypatch):
    """A client disconnect cancels the request; the shielded write must stay referenced."""
    monkeypatch.setattr(settings, "request_deadline_s", 9.5)
    started = asyncio.Event()
    async def slow_insert(**kwargs):
        started.set()
        await asyncio.sleep(0.3)
        return "id"
    p = _slow_insert_patches(slow_insert)
    with p[0], p[1], p[2], p[3], p[4], p[5]:
        request = asyncio.create_task(verify_claim("The Earth orbits the Sun"))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert len(orchestrator._background_tasks) == 1
        await orchestrator.drain_background_tasks(timeout=2.0)
    assert len(orchestrator._background_tasks) == 0


@pytest.mark.anyio
async def test_drain_gives_up_after_timeout():
    async def never_finishes():
        await asyncio.sleep(10)
    task = asyncio.create_task(never_finishes())
    orchestrator._background_tasks.add(task)
    task.add_done_callback(orchestrator._background_tasks.discard)
    start = time.perf_counter()
    await orchestrator.drain_background_tasks(timeout=0.05)
    assert time.perf_counter() - start < 1.0
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
