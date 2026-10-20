import pytest

import app.agents.crossref_agent as crossref_tools
import app.agents.factcheck_agent as factcheck_tools
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
            config=types.GenerateContentConfig(system_instruction="You are a Fact-Check judge agent."),
        )
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


