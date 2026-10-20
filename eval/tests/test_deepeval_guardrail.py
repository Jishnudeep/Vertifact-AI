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


def test_items_from_leaf_transcript(tmp_path):
    transcript_file = tmp_path / "test_leaf_transcript.jsonl"
    entries = [
        # Valid entry
        {
            "name": "factcheck_judge",
            "prompt": "<claim>Moon is made of cheese</claim>\n<search_snippets><snippet>Moon is made of rock and dust.</snippet></search_snippets>",
            "response": {"rating": "false", "explanation": "The moon consists of rock and dust, not cheese."},
        },
        # Non-target agent entry
        {
            "name": "other_agent",
            "prompt": "some prompt",
            "response": {"rating": "true", "explanation": "ok"},
        },
        # Missing search snippets
        {
            "name": "verdict_extractor",
            "prompt": "<claim>Water is wet</claim>",
            "response": {"rating": "true", "explanation": "Water is liquid."},
        },
        # Missing explanation
        {
            "name": "factcheck_judge",
            "prompt": "<claim>Earth is round</claim>\n<search_snippets>Earth is spherical.</search_snippets>",
            "response": {"rating": "true"},
        },
    ]

    with open(transcript_file, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    items, skipped = items_from_leaf_transcript(transcript_file)
    assert len(items) == 1
    assert items[0]["claim"] == "Moon is made of cheese"
    assert items[0]["actual_output"] == "The moon consists of rock and dust, not cheese."
    assert items[0]["retrieval_context"] == ["Moon is made of rock and dust."]
    assert skipped["non_target_agent"] == 1
    assert skipped["missing_search_snippets"] == 1
    assert skipped["missing_explanation"] == 1


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
    results = score_guardrail(items, judge=judge, delay_s=1.0)

    assert results["detected_poisoned"] >= 12, (
        f"Judge model caught only {results['detected_poisoned']}/15 poisoned items (target >= 12)"
    )
    assert results["flagged_clean"] <= 3, (
        f"Judge model falsely flagged {results['flagged_clean']}/15 clean items (target <= 3)"
    )
