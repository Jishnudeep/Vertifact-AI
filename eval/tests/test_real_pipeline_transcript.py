"""
The leaf transcript and the tool-cap audit on what the REAL pipeline produces (Task C re-review R1).

Workers run through ctx.run_node, whose requests carry no "internal name is ..." line, so the agent
name must come from ADK's callback context. These tests run the real `verify_claim` (scripted model,
patched services) under EvidenceReplay and never inject an identity string into a prompt.
"""
from unittest.mock import patch

import pytest

from app.agents import crossref_agent, factcheck_agent
from app.agents.orchestrator import verify_claim
from app.services.grounded_search import GroundedSearch, GroundedSource
from app.services.nli import NLIScores
from app.services.page_fetch import PageText
from eval.deepeval_runner import items_from_leaf_transcript
from eval.replay import EvidenceReplay
from eval.trace_eval import audit_trace, tool_call_counts
from tests.scripted_llm import call, final, install_script

pytestmark = pytest.mark.anyio

EXCERPT = "Apollo moon rock samples are basalt and anorthosite, not cheese."
EXPLANATION = "Snopes reports that moon rock samples are rock, not cheese."


async def _run_pipeline(tmp_path, monkeypatch, news_calls=2):
    async def fake_factcheck_db(query, api_key):
        return []

    async def fake_grounded(kind, query):
        domain = "snopes.com" if kind == "factcheck" else "reuters.com"
        return GroundedSearch(
            sources=[GroundedSource(url=f"https://vertexaisearch.cloud.google.com/redirect/{domain}",
                                    title=domain, domain=domain)],
            queries=[query], search_suggestions_html=None)

    async def fake_fetch(url, expected_domain=None):
        return PageText(final_url=f"https://www.{expected_domain}/article", domain=expected_domain,
                        title="Moon cheese", text=EXCERPT + " " * 3 + "More text. " * 30)

    decision = final(evidence="web", source_id="web-1", relevant=True, opposite_polarity=False,
                     rating="false", explanation=EXPLANATION)
    routes = {
        "fact-check lookup agent": [call("search_fact_check_database", query="moon cheese"),
                                    call("search_fact_check_sites", query="moon cheese"), decision],
        "cross-reference agent": [*[call("search_news_coverage", query=f"moon cheese {i}") for i in range(news_calls)],
                                  final(queries_rationale="searched twice")],
    }
    support = lambda pairs: [NLIScores(entailment=0.9, contradiction=0.05, neutral=0.05) for _ in pairs]  # noqa: E731

    with patch("app.agents.orchestrator.generate_embedding", return_value=[0.1] * 384), \
         patch("app.agents.orchestrator.lookup_cache", return_value=None), \
         patch("app.agents.orchestrator.insert_cache", return_value=None), \
         patch.object(factcheck_agent, "search_factchecks", fake_factcheck_db), \
         patch.object(factcheck_agent, "grounded_search", fake_grounded), \
         patch.object(factcheck_agent, "fetch_page_text", fake_fetch), \
         patch.object(crossref_agent, "grounded_search", fake_grounded), \
         patch.object(crossref_agent, "fetch_page_text", fake_fetch), \
         patch.object(crossref_agent, "classify_pairs", support), \
         pytest.MonkeyPatch.context() as mp:
        install_script(mp, routes=routes)
        replay = EvidenceReplay(tmp_path / "cache", "record")
        replay.install()
        try:
            replay.reset_claim_counts("claim-001")
            response = await verify_claim("The Moon is made of cheese", bypass_cache=True)
        finally:
            replay.uninstall()
    return replay, response


async def test_the_real_pipeline_names_each_worker_turn_and_counts_its_tools(tmp_path, monkeypatch):
    replay, response = await _run_pipeline(tmp_path, monkeypatch)

    names = {e["name"] for e in replay.leaf_transcript}
    assert names == {"factcheck_judge", "crossref_searcher"}, names
    assert all(e["claim_id"] == "claim-001" for e in replay.leaf_transcript)
    assert tool_call_counts(replay.leaf_transcript, "claim-001") == {
        "search_fact_check_database": 1, "search_fact_check_sites": 1, "search_news_coverage": 2}

    report = audit_trace(response, tool_calls=tool_call_counts(replay.leaf_transcript, "claim-001"))
    assert report.checks["tool_caps"] is True


async def test_leaf_items_are_built_from_the_real_pipelines_factcheck_turns(tmp_path, monkeypatch):
    replay, _ = await _run_pipeline(tmp_path, monkeypatch)
    path = tmp_path / "leaf.jsonl"
    replay.write_leaf_transcript(path)

    items, skipped = items_from_leaf_transcript(path)
    assert len(items) == 1, skipped
    assert items[0]["actual_output"] == EXPLANATION
    assert any(EXCERPT in context for context in items[0]["retrieval_context"])
    assert items[0]["claim"] == "The Moon is made of cheese"


async def test_a_third_news_search_is_refused_by_the_tool_and_not_counted_as_executed(tmp_path, monkeypatch):
    replay, _ = await _run_pipeline(tmp_path, monkeypatch, news_calls=3)
    assert tool_call_counts(replay.leaf_transcript, "claim-001")["search_news_coverage"] == 2
