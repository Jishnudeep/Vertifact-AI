import pytest

import app.agents.crossref_agent as crossref_tools
import app.agents.factcheck_agent as factcheck_tools
from eval.replay import (
    EvidenceReplay, ReplayMissError, excluding_domains,
    filter_factcheck_claims, filter_web_results, request_key,
)
from google.adk.models.llm_request import LlmRequest
from google.genai import types
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
async def test_live_errors_are_not_recorded(tmp_path, monkeypatch):
    async def failing(kind, claim):
        raise EvidenceAPIError("Gemini search failed", provider="gemini_search")

    monkeypatch.setattr(crossref_tools, "grounded_search", failing)
    monkeypatch.setattr(factcheck_tools, "grounded_search", failing)
    replay = EvidenceReplay(tmp_path, "record")
    replay.install()
    try:
        with pytest.raises(EvidenceAPIError):
            await crossref_tools.grounded_search("coverage", "claim")
    finally:
        replay.uninstall()
    assert replay.recorded == 0
    assert list(tmp_path.iterdir()) == []



@pytest.mark.anyio
async def test_grounded_search_replayed_and_filtered(tmp_path, monkeypatch):
    from app.services.grounded_search import GroundedSearch, GroundedSource
    calls = {"n": 0}
    sample_gs = GroundedSearch(
        sources=[
            GroundedSource(url="https://www.politifact.com/x", title="PolitiFact X", domain="politifact.com"),
            GroundedSource(url="https://www.reuters.com/y", title="Reuters Y", domain="reuters.com"),
        ],
        queries=["q1"],
        search_suggestions_html="<div>sugg</div>",
    )

    async def fake_grounded_search(kind, claim):
        calls["n"] += 1
        return sample_gs

    monkeypatch.setattr(crossref_tools, "grounded_search", fake_grounded_search)
    monkeypatch.setattr(factcheck_tools, "grounded_search", fake_grounded_search)

    recorder = EvidenceReplay(tmp_path, "record")
    recorder.install()
    try:
        first = await crossref_tools.grounded_search("coverage", "claim 1")
    finally:
        recorder.uninstall()
    assert recorder.recorded == 1 and calls["n"] == 1
    assert first == sample_gs

    player = EvidenceReplay(tmp_path, "replay")
    player.install()
    try:
        with excluding_domains(["politifact.com"]):
            filtered = await crossref_tools.grounded_search("coverage", "claim 1")
        unfiltered = await crossref_tools.grounded_search("coverage", "claim 1")
    finally:
        player.uninstall()

    assert calls["n"] == 1  # replay did not call live
    assert len(filtered.sources) == 1
    assert filtered.sources[0].domain == "reuters.com"
    assert len(unfiltered.sources) == 2


@pytest.mark.anyio
async def test_page_fetch_replay_roundtrip_none(tmp_path, monkeypatch):
    from app.services.page_fetch import PageText
    calls = {"n": 0}

    async def fake_fetch(url, expected_domain=None):
        calls["n"] += 1
        if "reuters" in url:
            return PageText(final_url=url, domain="reuters.com", title="Title", text="Text " * 50)
        return None

    monkeypatch.setattr(crossref_tools, "fetch_page_text", fake_fetch)
    monkeypatch.setattr(factcheck_tools, "fetch_page_text", fake_fetch)

    recorder = EvidenceReplay(tmp_path, "record")
    recorder.install()
    try:
        res_ok = await crossref_tools.fetch_page_text("https://reuters.com/news", "reuters.com")
        res_none = await crossref_tools.fetch_page_text("https://broken.com", "broken.com")
    finally:
        recorder.uninstall()
    assert recorder.recorded == 2 and calls["n"] == 2
    assert res_ok is not None and res_none is None

    player = EvidenceReplay(tmp_path, "replay")
    player.install()
    try:
        rep_ok = await crossref_tools.fetch_page_text("https://reuters.com/news", "reuters.com")
        rep_none = await crossref_tools.fetch_page_text("https://broken.com", "broken.com")
    finally:
        player.uninstall()

    assert calls["n"] == 2
    assert rep_ok == res_ok
    assert rep_none is None


@pytest.mark.anyio
async def test_replay_miss_raises_for_grounded_search(tmp_path, monkeypatch):
    player = EvidenceReplay(tmp_path, "replay")
    player.install()
    try:
        with pytest.raises(ReplayMissError) as exc_info:
            await factcheck_tools.grounded_search("factcheck", "unrecorded claim")
        assert exc_info.value.provider == "gemini_search"
    finally:
        player.uninstall()



def test_replay_request_key_stable_across_dates():
    """Different injected dates must not break the replay cache key (Task A.2)."""
    req1 = LlmRequest(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="Today's date is 2026-10-20 (UTC). Base instruction text.",
        ),
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="test")])],
    )
    req2 = LlmRequest(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction="Today's date is 2026-10-21 (UTC). Base instruction text.",
        ),
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="test")])],
    )

    key1 = request_key(req1)
    key2 = request_key(req2)
    assert key1 == key2, "Date differences in system instruction must be normalized in replay request_key"


@pytest.mark.parametrize("mode", ["record", "record-missing"])
def test_replay_record_modes_pass_current_date_and_cache_key_stable(tmp_path, monkeypatch, mode):
    """
    EvidenceReplay.install() in record / record-missing modes must NOT pin get_today_utc
    to a stale date; the live model must receive the real current date, while request_key
    normalizes the date so cache keys remain identical across two different dates.
    """
    import app.agents.common as common
    import app.agents.factcheck_agent as fc
    import app.agents.crossref_agent as cr

    real_date_1 = "2026-10-25"
    monkeypatch.setattr(common, "get_today_utc", lambda: real_date_1)

    replay = EvidenceReplay(tmp_path, mode=mode)
    replay.install()
    try:
        assert common.get_today_utc() == real_date_1
        assert f"Today's date is {real_date_1} (UTC)" in fc.get_instruction()
        assert f"Today's date is {real_date_1} (UTC)" in cr.get_instruction()
    finally:
        replay.uninstall()

    real_date_2 = "2026-10-26"
    req1 = LlmRequest(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=fc.get_instruction(real_date_1),
        ),
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="claim 1")])],
    )
    req2 = LlmRequest(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=fc.get_instruction(real_date_2),
        ),
        contents=[types.Content(role="user", parts=[types.Part.from_text(text="claim 1")])],
    )

    assert request_key(req1) == request_key(req2), "Cache key must be identical across different dates"
