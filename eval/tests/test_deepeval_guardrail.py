"""
Tests for DeepEval faithfulness guardrail and leaf groundedness (Plan 9, Task 5).
Includes offline dataset validation, scoring logic tests, transcript parsing tests,
and an opt-in live test for Gate D6.
"""
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from app.config import settings
from eval.deepeval_runner import (
    DEFAULT_DATASET,
    get_judge_model,
    items_from_leaf_transcript,
    score_guardrail,
    score_leaf_groundedness,
)


def _load_dataset():
    assert DEFAULT_DATASET.exists(), f"Dataset not found at {DEFAULT_DATASET}"
    with open(DEFAULT_DATASET, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def test_dataset_structure_and_balance():
    items = _load_dataset()
    assert len(items) == 30, f"Expected 30 items, got {len(items)}"

    ids = [x["id"] for x in items]
    assert len(ids) == len(set(ids)), "IDs must be unique"

    clean = [x for x in items if not x["is_hallucinated"]]
    poisoned = [x for x in items if x["is_hallucinated"]]

    assert len(clean) == 15, f"Expected 15 clean items, got {len(clean)}"
    assert len(poisoned) == 15, f"Expected 15 poisoned items, got {len(poisoned)}"

    clean_ids = {x["id"] for x in clean}
    valid_types = {"fabricated_statistic", "invented_entity", "wrong_date", "inverted_verdict"}

    for item in clean:
        assert item["hallucination_type"] is None, f"{item['id']} has hallucination_type but is clean"
        assert item["twin_of"] is None, f"{item['id']} should have twin_of=None"
        assert len(item["source_snippets"]) > 0, f"{item['id']} has empty source_snippets"
        assert "explanation" in item["output"], f"{item['id']} output missing explanation"

    twins_seen = set()
    for item in poisoned:
        assert item["hallucination_type"] in valid_types, (
            f"{item['id']} has invalid hallucination_type: {item['hallucination_type']}"
        )
        assert item["twin_of"] in clean_ids, f"{item['id']} twin_of {item['twin_of']} not in clean_ids"
        twins_seen.add(item["twin_of"])
        assert len(item["source_snippets"]) > 0, f"{item['id']} has empty source_snippets"
        assert "explanation" in item["output"], f"{item['id']} output missing explanation"

    assert twins_seen == clean_ids, "Every clean item must have exactly one poisoned twin"


def test_pairs_differ_only_in_injection():
    items = _load_dataset()
    by_id = {x["id"]: x for x in items}
    poisoned = [x for x in items if x["is_hallucinated"]]

    for p in poisoned:
        c = by_id[p["twin_of"]]
        assert p["claim"] == c["claim"], f"Claim mismatch between {p['id']} and {c['id']}"
        assert p["source_snippets"] == c["source_snippets"], f"Snippets mismatch between {p['id']} and {c['id']}"
        # Output explanation must differ (the injection)
        assert p["output"]["explanation"] != c["output"]["explanation"], (
            f"Explanation should differ between {p['id']} and {c['id']}"
        )


def test_scoring_logic_with_fake_judge():
    items = _load_dataset()

    def fake_judge(item: Dict[str, Any]):
        # Flag all poisoned, flag 1 clean
        if item["is_hallucinated"]:
            return {"faithfulness": 0.4, "hallucination": 0.8, "reason": "unsupported fact"}
        if item["id"] == "hallu-001":
            return {"faithfulness": 0.75, "hallucination": 0.6, "reason": "borderline"}
        return {"faithfulness": 0.95, "hallucination": 0.0, "reason": "supported"}

    results = score_guardrail(items, judge=fake_judge)

    assert results["total_items"] == 30
    assert results["clean_items"] == 15
    assert results["poisoned_items"] == 15
    assert results["detected_poisoned"] == 15
    assert results["hallucination_catch_rate"] == 1.0
    assert results["flagged_clean"] == 1
    assert round(results["false_positive_rate"], 4) == round(1 / 15, 4)


def test_flagging_threshold_boundary():
    # Exactly 0.8 is PASS (not flagged); strictly < 0.8 is flagged
    item_borderline = {
        "id": "t-1",
        "claim": "Claim",
        "output": {"explanation": "Explanation"},
        "source_snippets": ["Context"],
        "is_hallucinated": False,
    }
    item_failing = {
        "id": "t-2",
        "claim": "Claim",
        "output": {"explanation": "Explanation"},
        "source_snippets": ["Context"],
        "is_hallucinated": True,
    }

    judge_80 = lambda x: {"faithfulness": 0.80, "hallucination": 0.0}
    res_80 = score_guardrail([item_borderline], judge=judge_80)
    assert not res_80["items"][0]["flagged"]
    assert res_80["flagged_clean"] == 0

    judge_79 = lambda x: {"faithfulness": 0.799, "hallucination": 0.0}
    res_79 = score_guardrail([item_failing], judge=judge_79)
    assert res_79["items"][0]["flagged"]
    assert res_79["detected_poisoned"] == 1


def test_refusal_when_judge_model_is_none(monkeypatch):
    monkeypatch.setattr(settings, "eval_judge_model", None)
    with pytest.raises(ValueError, match="D6"):
        get_judge_model()


@pytest.mark.live_api
def test_live_guardrail_gate_d6():
    """
    Opt-in live evaluation against the real judge model (Gate D6).
    Asserts >= 12/15 poisoned caught (>=80%) and <= 3/15 clean false alarms (<=20%).
    """
    judge = get_judge_model()
    items = _load_dataset()
    results = score_guardrail(items, judge=judge, delay_s=5.0)
    assert results["status"] == "OK", f"{results['judge_errors']} judge error(s)"

    assert results["detected_poisoned"] >= 12, (
        f"Judge model caught only {results['detected_poisoned']}/15 poisoned items (target >= 12)"
    )
    assert results["flagged_clean"] <= 3, (
        f"Judge model falsely flagged {results['flagged_clean']}/15 clean items (target <= 3)"
    )


# --- Task C review: fail-closed guardrail, quota guards, runnable CLI ---------------------------

import subprocess
import sys

from eval import deepeval_runner
from eval.deepeval_runner import is_flagged
from eval.replay import _note_agent
from types import SimpleNamespace


@pytest.fixture
def fake_deepeval(monkeypatch):
    """A stand-in `deepeval` so these tests do not need the optional `eval` dependency group."""
    import types

    pkg, metrics, test_case = types.ModuleType("deepeval"), types.ModuleType("deepeval.metrics"), types.ModuleType("deepeval.test_case")
    test_case.LLMTestCase = lambda **kwargs: kwargs
    pkg.metrics, pkg.test_case = metrics, test_case
    for name, module in (("deepeval", pkg), ("deepeval.metrics", metrics), ("deepeval.test_case", test_case)):
        monkeypatch.setitem(sys.modules, name, module)
    return metrics


def _poisoned_item():
    return {"id": "p1", "claim": "c", "source_snippets": ["s"], "is_hallucinated": True,
            "output": {"rating": "false", "explanation": "e"}}


@pytest.mark.parametrize("score,expected", [
    (None, True), (float("nan"), True), (0.799, True), (0.0, True), (0.8, False), (1.0, False),
])
def test_is_flagged_fails_closed(score, expected):
    assert is_flagged(score) is expected


def test_missing_faithfulness_from_the_judge_is_flagged_not_passed():
    res = score_guardrail([_poisoned_item()], judge=lambda item: {"hallucination": 0.0})
    assert res["items"][0]["flagged"] is True
    assert res["detected_poisoned"] == 1


def test_a_judge_exception_is_recorded_flagged_and_invalidates_the_run():
    def judge(item):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    res = score_guardrail([_poisoned_item()], judge=judge)
    assert res["items"][0]["judge_error"].startswith("RuntimeError")
    assert res["items"][0]["flagged"] is True
    assert res["judge_errors"] == 1 and res["status"] == "INVALID"


def test_partial_results_are_reported_after_every_item():
    seen = []

    def judge(item):
        if len(seen) == 1:
            raise RuntimeError("boom")
        return {"faithfulness": 1.0, "hallucination": 0.0}

    items = [{**_poisoned_item(), "id": f"p{i}"} for i in range(3)]
    score_guardrail(items, judge=judge, on_item=lambda scored: seen.append(len(scored)))
    assert seen == [1, 2, 3]


def test_fake_judge_path_does_not_import_deepeval(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepeval", None)
    monkeypatch.setitem(sys.modules, "deepeval.metrics", None)
    res = score_guardrail([_poisoned_item()], judge=lambda item: {"faithfulness": 0.5, "hallucination": 0.5})
    assert res["detected_poisoned"] == 1


def test_empty_input_reports_no_rate_rather_than_a_perfect_one():
    res = score_guardrail([], judge=lambda item: {})
    assert res["hallucination_catch_rate"] is None and res["false_positive_rate"] is None
    leaf = score_leaf_groundedness([], judge=lambda item: {})
    assert leaf["groundedness_rate"] is None and leaf["mean_faithfulness"] is None


def test_faithfulness_metric_penalizes_ambiguous_claims_and_hallucination_skips_reasons(monkeypatch, fake_deepeval):
    dm = fake_deepeval

    built = {}

    class Recorder:
        def __init__(self, name, **kwargs):
            built[name] = kwargs
            self.score, self.reason = 1.0, ""

        def measure(self, test_case):
            return self.score

    monkeypatch.setattr(dm, "FaithfulnessMetric", lambda **kw: Recorder("faithfulness", **kw), raising=False)
    monkeypatch.setattr(dm, "HallucinationMetric", lambda **kw: Recorder("hallucination", **kw), raising=False)
    score_guardrail([_poisoned_item()], judge=type("J", (), {"name": "gemini/gemini-3.1-flash-lite"})())
    assert built["faithfulness"]["penalize_ambiguous_claims"] is True
    assert built["hallucination"]["include_reason"] is False


@pytest.mark.parametrize("model", ["gemini/gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini/gemini-2.5-flash-lite"])
def test_quota_starved_judge_models_are_refused(monkeypatch, model):
    monkeypatch.setattr(settings, "eval_judge_model", model)
    with pytest.raises(ValueError, match="RPD"):
        get_judge_model()


def test_result_discloses_pipeline_model_and_self_judging(monkeypatch, fake_deepeval):
    monkeypatch.setattr(settings, "model_id", "gemini-3.1-flash-lite")

    class Judge:
        model_name = "gemini/gemini-3.1-flash-lite"

    class FakeMetric:
        def __init__(self, **kwargs):
            self.score, self.reason = 1.0, ""

        def measure(self, test_case):
            return 1.0

    dm = fake_deepeval
    monkeypatch.setattr(dm, "FaithfulnessMetric", FakeMetric, raising=False)
    monkeypatch.setattr(dm, "HallucinationMetric", FakeMetric, raising=False)
    res = score_guardrail([_poisoned_item()], judge=Judge())
    assert res["pipeline_model"] == "gemini-3.1-flash-lite"
    assert res["self_judging"] is True

    res = score_guardrail([_poisoned_item()], judge=lambda item: {"faithfulness": 1.0})
    assert res["self_judging"] is False


def test_main_refuses_to_spend_quota_without_live_flag(monkeypatch):
    monkeypatch.setattr(settings, "eval_judge_model", "gemini/gemini-3.1-flash-lite")
    monkeypatch.setattr(deepeval_runner, "get_judge_model",
                        lambda: pytest.fail("a judge must not be built without --live"))
    with pytest.raises(SystemExit) as exc:
        deepeval_runner.main(["--mode", "guardrail"])
    assert exc.value.code != 0


@pytest.mark.parametrize("script", ["deepeval_runner.py", "cache_bench.py"])
def test_cli_scripts_start_from_the_repo_root(script, tmp_path):
    root = Path(__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, str(root / "eval" / script), "--help"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-800:]


# --- leaf mode on transcripts produced by the real ReplayLlm -------------------------------------

DECISION = {"evidence": "web", "source_id": "web-1", "relevant": True, "opposite_polarity": False,
            "rating": "false", "explanation": "Snopes found that moon rocks are basalt, not cheese."}
EXCERPT = "Moon rock samples returned by Apollo are basalt and anorthosite."


def _factcheck_request():
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    web = {"query": "moon cheese", "count": 1, "sources": [
        {"source_id": "web-1", "domain": "snopes.com", "title": "t", "is_claimreview_authority": True,
         "excerpt": EXCERPT}]}
    db = {"query": "moon cheese", "count": 1, "reviews": [
        {"source_id": "db-1", "claim_reviewed": "The Moon is cheese", "publisher": "PolitiFact",
         "rating": "False", "normalized_rating": 0.0, "similarity": 0.9}]}
    fr = lambda name, response: types.Part(function_response=types.FunctionResponse(name=name, response=response))
    return LlmRequest(
        model="dummy",
        contents=[
            types.Content(role="user", parts=[types.Part(text="<claim>The Moon is made of cheese</claim>")]),
            types.Content(role="user", parts=[fr("search_fact_check_database", db),
                                              fr("search_fact_check_sites", web)]),
        ],
        config=types.GenerateContentConfig(
            system_instruction="You check whether a fact-check already answers the claim."),
    )


async def _record_final_turn(tmp_path, monkeypatch, final_part, mode="record"):
    import app.agents.common as common
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
    from eval.replay import EvidenceReplay

    class Inner(BaseLlm):
        model: str = "dummy"

        async def generate_content_async(self, llm_request, stream=False):
            yield LlmResponse(content=types.Content(role="model", parts=[final_part]))

    monkeypatch.setattr(common, "get_model", lambda: Inner())
    replay = EvidenceReplay(tmp_path / "cache", mode)
    replay.install()
    try:
        replay.reset_claim_counts("claim-001")
        _note_agent(SimpleNamespace(agent_name="factcheck_judge"))     # what ADK's callback context provides
        _ = [r async for r in common.get_model().generate_content_async(_factcheck_request())]
    finally:
        replay.uninstall()
    path = tmp_path / f"leaf-{mode}.jsonl"
    replay.write_leaf_transcript(path)
    return replay, path


@pytest.mark.anyio
@pytest.mark.parametrize("final_part_factory", [
    lambda t: t.Part(function_call=t.FunctionCall(name="set_model_response", args=DECISION)),
    lambda t: t.Part(text=json.dumps(DECISION)),
], ids=["set_model_response_call", "json_text_answer"])
async def test_leaf_items_come_from_a_real_factcheck_turn(tmp_path, monkeypatch, final_part_factory):
    from google.genai import types

    _, path = await _record_final_turn(tmp_path, monkeypatch, final_part_factory(types))
    items, skipped = items_from_leaf_transcript(path)
    assert len(items) == 1, skipped
    item = items[0]
    assert item["claim"] == "The Moon is made of cheese"
    assert item["actual_output"] == DECISION["explanation"]
    assert EXCERPT in item["retrieval_context"]
    assert any("PolitiFact" in c and "False" in c for c in item["retrieval_context"])
    assert item["id"].startswith("claim-001")


@pytest.mark.anyio
async def test_leaf_items_also_come_from_replayed_turns(tmp_path, monkeypatch):
    from google.genai import types
    from eval.replay import EvidenceReplay
    import app.agents.common as common
    from google.adk.models.base_llm import BaseLlm

    await _record_final_turn(tmp_path, monkeypatch,
                             types.Part(function_call=types.FunctionCall(name="set_model_response", args=DECISION)))

    class Boom(BaseLlm):
        model: str = "dummy"

        async def generate_content_async(self, llm_request, stream=False):
            raise AssertionError("replay must not call the model")
            yield

    monkeypatch.setattr(common, "get_model", lambda: Boom())
    replay = EvidenceReplay(tmp_path / "cache", "replay")
    replay.install()
    try:
        _note_agent(SimpleNamespace(agent_name="factcheck_judge"))
        _ = [r async for r in common.get_model().generate_content_async(_factcheck_request())]
    finally:
        replay.uninstall()
    assert replay.calls_by_provider["llm"] == 1 and replay.misses == 0
    path = tmp_path / "leaf-replay.jsonl"
    replay.write_leaf_transcript(path)
    items, _ = items_from_leaf_transcript(path)
    assert len(items) == 1 and items[0]["actual_output"] == DECISION["explanation"]


@pytest.mark.anyio
async def test_an_unrecorded_llm_turn_in_replay_counts_a_miss(tmp_path, monkeypatch):
    import app.agents.common as common
    from google.adk.models.base_llm import BaseLlm
    from eval.replay import EvidenceReplay, ReplayMissError

    class Boom(BaseLlm):
        model: str = "dummy"

        async def generate_content_async(self, llm_request, stream=False):
            raise AssertionError("replay must not call the model")
            yield

    monkeypatch.setattr(common, "get_model", lambda: Boom())
    replay = EvidenceReplay(tmp_path / "empty", "replay")
    replay.install()
    try:
        with pytest.raises(ReplayMissError):
            _ = [r async for r in common.get_model().generate_content_async(_factcheck_request())]
    finally:
        replay.uninstall()
    assert replay.misses == 1


def test_leaf_skips_other_agents_and_non_final_turns(tmp_path):
    path = tmp_path / "t.jsonl"
    rows = [
        {"name": "crossref_searcher", "decision": {"explanation": "x"}, "tool_responses": []},
        {"name": LEAF_NAME, "decision": None, "tool_responses": []},
        {"name": LEAF_NAME, "decision": {"explanation": "e"}, "tool_responses": []},
        {"name": LEAF_NAME, "decision": {"rating": "false"}, "tool_responses": []},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    items, skipped = items_from_leaf_transcript(path)
    assert items == []
    assert skipped == {"non_target_agent": 1, "not_a_final_turn": 1, "missing_context": 1,
                       "missing_explanation": 1, "invalid_json": 1}


LEAF_NAME = "factcheck_judge"


# --- re-gate: disclosure for the real judge class, missing scores count as judge errors ----------

def test_judge_label_reads_the_name_deepeval_models_expose(monkeypatch):
    from eval.deepeval_runner import _judge_label

    class DeepEvalStyleModel:               # deepeval's LiteLLMModel keeps the id in `.name`; `.model` is None
        name = "gemini/gemini-3.1-flash-lite"
        model = None

    assert _judge_label(DeepEvalStyleModel()) == "gemini/gemini-3.1-flash-lite"


def test_judge_label_never_falls_back_to_an_object_repr():
    from eval.deepeval_runner import _judge_label

    with pytest.raises(ValueError, match="judge model"):
        _judge_label(object())


def test_self_judging_is_true_for_a_deepeval_style_judge(monkeypatch, fake_deepeval):
    monkeypatch.setattr(settings, "model_id", "gemini-3.1-flash-lite")

    class FakeMetric:
        def __init__(self, **kwargs):
            self.score, self.reason = 1.0, ""

        def measure(self, test_case):
            return 1.0

    monkeypatch.setattr(fake_deepeval, "FaithfulnessMetric", FakeMetric, raising=False)
    monkeypatch.setattr(fake_deepeval, "HallucinationMetric", FakeMetric, raising=False)

    class Judge:
        name = "gemini/gemini-3.1-flash-lite"
        model = None

    res = score_guardrail([_poisoned_item()], judge=Judge())
    assert res["self_judging"] is True and res["judge_model"] == "gemini/gemini-3.1-flash-lite"


def test_a_missing_score_without_an_exception_is_a_judge_error():
    res = score_guardrail([_poisoned_item()], judge=lambda item: {"hallucination": 0.0})
    assert res["items"][0]["flagged"] is True
    assert res["judge_errors"] == 1 and res["status"] == "INVALID"


def test_a_successful_cli_run_leaves_no_partial_file(tmp_path, monkeypatch):
    dataset = tmp_path / "set.json"
    dataset.write_text(json.dumps([_poisoned_item()]), encoding="utf-8")
    monkeypatch.setattr(deepeval_runner, "get_judge_model",
                        lambda: (lambda item: {"faithfulness": 0.1, "hallucination": 0.9}))
    out = tmp_path / "res.json"
    deepeval_runner.main(["--live", "--dataset", str(dataset), "--output", str(out), "--delay", "0"])
    assert out.exists() and not (tmp_path / "res.partial.json").exists()
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == "OK"
