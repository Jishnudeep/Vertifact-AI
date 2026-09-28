import pytest

import app.agents.crossref_agent as crossref_agent
import app.agents.factcheck_agent as factcheck_agent
from eval.replay import (
    EvidenceReplay, ReplayMissError, excluding_domains,
    filter_factcheck_claims, filter_web_results,
)
from app.services.errors import EvidenceAPIError


def test_filter_web_results_drops_excluded_domains():
    results = [{"url": "https://www.politifact.com/a"}, {"url": "https://www.reuters.com/b"}]
    assert filter_web_results(results, frozenset({"politifact.com"})) == [{"url": "https://www.reuters.com/b"}]


def test_filter_factcheck_claims_drops_reviews_and_empty_claims():
    claims = [
        {"text": "c1", "claimReview": [{"url": "https://www.politifact.com/x"}]},
        {"text": "c2", "claimReview": [{"url": "https://www.politifact.com/y"},
                                       {"url": "https://www.snopes.com/z"}]},
    ]
    kept = filter_factcheck_claims(claims, frozenset({"politifact.com"}))
    assert kept == [{"text": "c2", "claimReview": [{"url": "https://www.snopes.com/z"}]}]


def test_replay_miss_is_an_evidence_outage():
    assert issubclass(ReplayMissError, EvidenceAPIError)


@pytest.mark.anyio
async def test_record_then_replay_round_trip(tmp_path, monkeypatch):
    calls = {"n": 0}

    async def fake_search_web(query, api_key, max_results=5, include_domains=None, **kwargs):
        calls["n"] += 1
        return [{"url": "https://www.reuters.com/x", "title": query}]

    monkeypatch.setattr(crossref_agent, "search_web", fake_search_web)
    monkeypatch.setattr(factcheck_agent, "search_web", fake_search_web)

    recorder = EvidenceReplay(tmp_path, "record")
    recorder.install()
    try:
        first = await crossref_agent.search_web(query="q1", api_key="k", max_results=5)
    finally:
        recorder.uninstall()
    assert recorder.recorded == 1 and calls["n"] == 1

    player = EvidenceReplay(tmp_path, "replay")
    player.install()
    try:
        again = await crossref_agent.search_web(query="q1", api_key="DIFFERENT-KEY", max_results=5)
        with pytest.raises(ReplayMissError):
            await crossref_agent.search_web(query="never-recorded", api_key="k", max_results=5)
    finally:
        player.uninstall()
    assert again == first
    assert calls["n"] == 1          # replay never touched the live function
    assert player.misses == 1
    assert crossref_agent.search_web is fake_search_web  # uninstall restored the original


@pytest.mark.anyio
async def test_exclusion_applies_on_replayed_results(tmp_path, monkeypatch):
    async def fake_search_web(query, api_key, max_results=5, include_domains=None, **kwargs):
        return [{"url": "https://www.politifact.com/x"}, {"url": "https://www.reuters.com/y"}]

    monkeypatch.setattr(crossref_agent, "search_web", fake_search_web)
    monkeypatch.setattr(factcheck_agent, "search_web", fake_search_web)
    replay = EvidenceReplay(tmp_path, "record")
    replay.install()
    try:
        with excluding_domains(["politifact.com"]):
            inside = await crossref_agent.search_web(query="q", api_key="k")
        outside = await crossref_agent.search_web(query="q", api_key="k")
    finally:
        replay.uninstall()
    assert [r["url"] for r in inside] == ["https://www.reuters.com/y"]
    assert len(outside) == 2        # the cache stores raw results; exclusion is only a view


@pytest.mark.anyio
async def test_live_errors_are_not_recorded(tmp_path, monkeypatch):
    async def failing(query, api_key, max_results=5, include_domains=None, **kwargs):
        raise EvidenceAPIError("HTTP 432", provider="tavily")

    monkeypatch.setattr(crossref_agent, "search_web", failing)
    monkeypatch.setattr(factcheck_agent, "search_web", failing)
    replay = EvidenceReplay(tmp_path, "record")
    replay.install()
    try:
        with pytest.raises(EvidenceAPIError):
            await crossref_agent.search_web(query="q", api_key="k")
    finally:
        replay.uninstall()
    assert replay.recorded == 0
    assert list(tmp_path.iterdir()) == []
