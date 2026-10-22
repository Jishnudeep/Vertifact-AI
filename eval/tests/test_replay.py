import pytest
from types import SimpleNamespace

import app.agents.crossref_agent as crossref_tools
import app.agents.factcheck_agent as factcheck_tools
from eval.replay import (
    EvidenceReplay, ReplayMissError, _note_agent, excluding_domains,
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


@pytest.mark.anyio
async def test_provider_calls_tallies(tmp_path, monkeypatch):
    from app.services.grounded_search import GroundedSearch, GroundedSource
    from app.services.page_fetch import PageText

    async def fake_search(kind, claim):
        return GroundedSearch([], [], None)

    async def fake_fetch(url, expected_domain=None):
        return PageText(final_url=url, domain="example.com", title="T", text="Content")

    async def fake_factchecks(query, api_key):
        return []

    monkeypatch.setattr(crossref_tools, "grounded_search", fake_search)
    monkeypatch.setattr(crossref_tools, "fetch_page_text", fake_fetch)
    monkeypatch.setattr(factcheck_tools, "search_factchecks", fake_factchecks)

    replay = EvidenceReplay(tmp_path, "record")
    replay.install()
    try:
        assert replay.calls_by_provider == {"search": 0, "fetch": 0, "factcheck_api": 0, "llm": 0}
        assert replay.claim_counts() == {"search": 0, "fetch": 0, "factcheck_api": 0, "llm": 0}

        await crossref_tools.grounded_search("coverage", "claim 1")
        await crossref_tools.fetch_page_text("https://example.com/1", "example.com")
        await factcheck_tools.search_factchecks("claim 1", "test-key")

        assert replay.calls_by_provider == {"search": 1, "fetch": 1, "factcheck_api": 1, "llm": 0}
        assert replay.claim_counts() == {"search": 1, "fetch": 1, "factcheck_api": 1, "llm": 0}

        replay.reset_claim_counts()
        assert replay.claim_counts() == {"search": 0, "fetch": 0, "factcheck_api": 0, "llm": 0}
        assert replay.calls_by_provider == {"search": 1, "fetch": 1, "factcheck_api": 1, "llm": 0}
    finally:
        replay.uninstall()


@pytest.mark.anyio
async def test_failed_calls_do_not_increment_tallies(tmp_path, monkeypatch):
    async def failing(kind, claim):
        raise EvidenceAPIError("Search failure", provider="gemini_search")

    monkeypatch.setattr(crossref_tools, "grounded_search", failing)
    replay = EvidenceReplay(tmp_path, "record")
    replay.install()
    try:
        with pytest.raises(EvidenceAPIError):
            await crossref_tools.grounded_search("coverage", "claim")
        assert replay.calls_by_provider["search"] == 0
        assert replay.claim_counts()["search"] == 0
    finally:
        replay.uninstall()


@pytest.mark.anyio
async def test_leaf_transcript_recorded_in_replay_mode(tmp_path, monkeypatch):
    import app.agents.common as common
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class DummyLlm(BaseLlm):
        model: str = "dummy"
        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text='{"rating": "false", "explanation": "untrue"}')]))

    monkeypatch.setattr(common, "get_model", lambda: DummyLlm())

    recorder = EvidenceReplay(tmp_path, "record")
    recorder.install()
    try:
        model = common.get_model()
        req = LlmRequest(
            model="dummy",
            contents=[types.Content(role="user", parts=[types.Part(text="Check this claim")])],
            config=types.GenerateContentConfig(system_instruction="You check claims."),
        )
        _note_agent(SimpleNamespace(agent_name="factcheck_judge"))     # what ADK's callback context provides
        responses = [r async for r in model.generate_content_async(req)]
        assert len(responses) == 1
        assert len(recorder.leaf_transcript) == 1
        entry = recorder.leaf_transcript[0]
        assert entry["name"] == "factcheck_judge"
        assert "Check this claim" in entry["prompt"]
        assert entry["response"] == {"rating": "false", "explanation": "untrue"}
        assert recorder.calls_by_provider["llm"] == 1
    finally:
        recorder.uninstall()




@pytest.mark.anyio
async def test_record_missing_uses_cache_when_present_and_records_when_missing(tmp_path, monkeypatch):
    from app.services.grounded_search import GroundedSearch, GroundedSource
    calls = {"n": 0}
    sample_gs = GroundedSearch(
        sources=[GroundedSource(url="https://www.reuters.com/y", title="Reuters Y", domain="reuters.com")],
        queries=["q1"],
        search_suggestions_html="<div>sugg</div>",
    )

    async def fake_grounded_search(kind, claim):
        calls["n"] += 1
        return sample_gs

    monkeypatch.setattr(crossref_tools, "grounded_search", fake_grounded_search)
    monkeypatch.setattr(factcheck_tools, "grounded_search", fake_grounded_search)

    # First run in record-missing mode with empty cache -> live call made and cached
    rm_replay = EvidenceReplay(tmp_path, "record-missing")
    rm_replay.install()
    try:
        first = await crossref_tools.grounded_search("coverage", "claim 1")
    finally:
        rm_replay.uninstall()

    assert calls["n"] == 1
    assert rm_replay.recorded == 1
    assert first == sample_gs

    # Second run in record-missing mode for same claim -> served from cache, live call NOT made
    rm_replay2 = EvidenceReplay(tmp_path, "record-missing")
    rm_replay2.install()
    try:
        second = await crossref_tools.grounded_search("coverage", "claim 1")
        assert calls["n"] == 1  # No new live call!
        assert rm_replay2.recorded == 0

        # Query a new claim that is not cached -> calls live and records
        third = await crossref_tools.grounded_search("coverage", "claim 2")
        assert calls["n"] == 2  # Live call made for uncached claim!
        assert rm_replay2.recorded == 1
    finally:
        rm_replay2.uninstall()

    assert second == sample_gs
    assert third == sample_gs


@pytest.mark.anyio
async def test_record_missing_failed_calls_stay_unrecorded(tmp_path, monkeypatch):
    async def failing(kind, claim):
        raise EvidenceAPIError("Search failure", provider="gemini_search")

    monkeypatch.setattr(crossref_tools, "grounded_search", failing)
    replay = EvidenceReplay(tmp_path, "record-missing")
    replay.install()
    try:
        with pytest.raises(EvidenceAPIError):
            await crossref_tools.grounded_search("coverage", "claim")
        assert replay.calls_by_provider["search"] == 0
        assert replay.recorded == 0
        assert list(tmp_path.iterdir()) == []
    finally:
        replay.uninstall()


@pytest.mark.anyio
async def test_record_missing_llm_uses_cache_and_records_missing(tmp_path, monkeypatch):
    import app.agents.common as common
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    llm_calls = {"n": 0}

    class CountingLlm(BaseLlm):
        model: str = "dummy"
        async def generate_content_async(self, llm_request, stream=False):
            llm_calls["n"] += 1
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text='{"rating": "mostly_true"}')]))

    monkeypatch.setattr(common, "get_model", lambda: CountingLlm())

    rm_replay = EvidenceReplay(tmp_path, "record-missing")
    rm_replay.install()
    try:
        model = common.get_model()
        req1 = LlmRequest(
            model="dummy",
            contents=[types.Content(role="user", parts=[types.Part(text="Prompt 1")])],
            config=types.GenerateContentConfig(system_instruction="You are a judge."),
        )
        responses1 = [r async for r in model.generate_content_async(req1)]
        assert len(responses1) == 1
        assert llm_calls["n"] == 1
        assert rm_replay.recorded == 1

        # Second call with same request in record-missing -> should hit cache, NOT call inner model
        responses2 = [r async for r in model.generate_content_async(req1)]
        assert len(responses2) == 1
        assert llm_calls["n"] == 1  # Unchanged!

        # Call with different prompt -> should invoke inner model
        req2 = LlmRequest(
            model="dummy",
            contents=[types.Content(role="user", parts=[types.Part(text="Prompt 2")])],
            config=types.GenerateContentConfig(system_instruction="You are a judge."),
        )
        responses3 = [r async for r in model.generate_content_async(req2)]
        assert len(responses3) == 1
        assert llm_calls["n"] == 2
        assert rm_replay.recorded == 2
    finally:
        rm_replay.uninstall()


def test_run_eval_cli_args_parsing():
    from eval.run_eval import _parse_args
    args = _parse_args(["--replay", "record-missing", "--limit", "10", "--offset", "5"])
    assert args.replay == "record-missing"
    assert args.limit == 10
    assert args.offset == 5



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
