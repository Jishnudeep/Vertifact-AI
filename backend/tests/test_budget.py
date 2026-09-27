from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from app.services import budget as budget_mod
from app.services.budget import DailyBudget, LLMBudgetExceeded
from app.services.errors import EvidenceAPIError
from app.services.search_api import search_web


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def test_budget_charges_until_limit_then_refuses():
    b = DailyBudget("t", limit=3, clock=Clock())
    assert b.try_charge(2) and b.used == 2
    assert not b.try_charge(2)          # would exceed; nothing charged
    assert b.used == 2
    assert b.try_charge(1) and b.used == 3


def test_budget_resets_on_new_utc_day():
    clock = Clock()
    b = DailyBudget("t", limit=1, clock=clock)
    assert b.try_charge()
    assert not b.try_charge()
    clock.now += timedelta(days=1)
    assert b.try_charge()


def test_zero_limit_disables_guard():
    b = DailyBudget("t", limit=0, clock=Clock())
    assert all(b.try_charge(100) for _ in range(5))


@pytest.mark.anyio
async def test_exhausted_tavily_budget_is_an_outage_without_http(monkeypatch):
    monkeypatch.setattr(budget_mod, "get_tavily_budget", lambda: DailyBudget("tavily", limit=1, clock=Clock()))
    with patch("app.services.search_api.httpx.AsyncClient") as client:
        with pytest.raises(EvidenceAPIError) as err:
            await search_web("q", api_key="k", search_depth="advanced")  # costs 2 > 1
    assert err.value.provider == "budget"
    client.assert_not_called()


@pytest.mark.anyio
async def test_basic_depth_costs_one_credit(monkeypatch):
    b = DailyBudget("tavily", limit=5, clock=Clock())
    monkeypatch.setattr(budget_mod, "get_tavily_budget", lambda: b)
    fake_resp = AsyncMock()
    fake_resp.status_code = 200
    fake_resp.json = lambda: {"results": []}
    with patch("app.services.search_api.httpx.AsyncClient") as client:
        client.return_value.__aenter__.return_value.post = AsyncMock(return_value=fake_resp)
        await search_web("q", api_key="k", search_depth="basic")
        payload = client.return_value.__aenter__.return_value.post.call_args.kwargs["json"]
    assert payload["search_depth"] == "basic"
    assert b.used == 1


@pytest.mark.anyio
async def test_exhausted_llm_budget_raises_before_calling_the_model(monkeypatch):
    from app.utils import adk_helpers
    exhausted = DailyBudget("llm", limit=1, clock=Clock())
    exhausted.try_charge()
    monkeypatch.setattr(budget_mod, "get_llm_budget", lambda: exhausted)
    with patch.object(adk_helpers, "get_adk_runner") as runner:
        with pytest.raises(LLMBudgetExceeded):
            await adk_helpers.run_adk_agent("n", "i", "p", "app")
    runner.return_value.run_async.assert_not_called()


@pytest.mark.anyio
async def test_crossref_uses_basic_depth():
    from app.agents.crossref_agent import run
    with patch("app.agents.crossref_agent.search_web", new=AsyncMock(return_value=[])) as sw:
        await run("NASA announces Mars landing")
    assert all(c.kwargs.get("search_depth") == "basic" for c in sw.call_args_list)
