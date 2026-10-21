"""
Run-summary integrity from the Task C review (B8, B9, N3, N7): what the summary records must be what
was applied, and a run must not leave or read stale artifacts.
"""
import hashlib
import json

import pytest

import eval.run_eval as re_mod
from app.agents.orchestrator import AGENT_NODES
from eval.run_eval import STATUS_GUARDRAIL_FAIL, assess_run_validity, run_evaluation

pytestmark = pytest.mark.anyio


def _claim(cid, text="The sky is blue."):
    return {"id": cid, "claim_text": text, "category": "science", "claim_type": "verifiable",
            "ground_truth_score": 0.9, "expected_confidence": "high", "origin_domain": None}


def _response(timeouts=None):
    if timeouts is not None:
        timeouts.append({n.name: n.timeout for n, _, _, _ in AGENT_NODES})
    return {
        "trust_score": {"c_total": 0.85},
        "confidence": {"level": "high", "evidence_status": "corroborated", "verdict_found": True, "warnings": []},
        "sources": [],
        "processing_time_ms": 100.0,
        "agents": [{"name": n, "status": "ok", "duration_ms": 10, "error_code": None}
                   for n in ("factcheck", "crossref", "linguistic")],
    }


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps([_claim("c1"), _claim("c2")]), encoding="utf-8")
    return path


async def test_the_pinned_agent_timeout_reaches_the_agent_nodes_and_is_restored(tmp_path, dataset, monkeypatch):
    seen = []
    before = {n.name: n.timeout for n, _, _, _ in AGENT_NODES}

    async def fake_verify(text, bypass_cache=True):
        return _response(seen)

    monkeypatch.setattr(re_mod, "verify_claim", fake_verify)
    report = await run_evaluation(dataset_paths=[dataset], output_path=tmp_path / "out.json",
                                  replay_mode="replay", agent_timeout_s=25.0, request_deadline_s=30.0)
    assert all(t == {"factcheck": 25.0, "crossref": 25.0, "linguistic": 25.0} for t in seen)
    assert report["summary"]["settings"]["node_timeouts"] == {"factcheck": 25.0, "crossref": 25.0, "linguistic": 25.0}
    assert {n.name: n.timeout for n, _, _, _ in AGENT_NODES} == before


async def test_summary_records_the_pipeline_settings_and_dataset_hash(tmp_path, dataset, monkeypatch):
    async def fake_verify(text, bypass_cache=True):
        return _response()

    monkeypatch.setattr(re_mod, "verify_claim", fake_verify)
    report = await run_evaluation(dataset_paths=[dataset], output_path=tmp_path / "out.json", replay_mode="replay")
    recorded = report["summary"]["settings"]
    assert recorded["model_id"] == re_mod.settings.model_id
    assert recorded["enable_llm_sensationalism"] == re_mod.settings.enable_llm_sensationalism
    assert recorded["timeouts_pinned"] is False
    assert report["summary"]["dataset_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()


async def test_traces_and_transcript_follow_the_output_path_and_stale_traces_are_cleared(tmp_path, dataset, monkeypatch):
    async def fake_verify(text, bypass_cache=True):
        return _response()

    monkeypatch.setattr(re_mod, "verify_claim", fake_verify)
    out = tmp_path / "run" / "out.json"
    stale = out.parent / "traces" / "stale-claim.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")

    await run_evaluation(dataset_paths=[dataset], output_path=out, replay_mode="replay")
    assert not stale.exists()
    assert sorted(p.name for p in (out.parent / "traces").glob("*.json")) == ["c1.json", "c2.json"]
    assert (out.parent / "leaf_transcript.jsonl").exists()


async def test_scored_claims_are_counted_separately_from_attempted_ones(tmp_path, dataset, monkeypatch):
    calls = {"n": 0}

    async def fake_verify(text, bypass_cache=True):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("provider exploded")
        return _response()

    monkeypatch.setattr(re_mod, "verify_claim", fake_verify)
    report = await run_evaluation(dataset_paths=[dataset], output_path=tmp_path / "out.json", replay_mode="replay")
    assert report["summary"]["verifiable_claims"] == 2
    assert report["summary"]["scored_verifiable_claims"] == 1


@pytest.mark.parametrize("check", ["tool_caps", "degradation_flagged"])
def test_trajectory_and_degradation_trace_failures_are_guardrail_failures(check):
    status, reasons = assess_run_validity(
        verifiable_preds=[0.2, 0.8], verifiable_targets=[0.0, 1.0],
        zero_source_rate=0.0, error_rate=0.0,
        bias_metrics={"passed": True}, bridging_violations=[], adversarial_results=[],
        replay_misses=0, trace_failures_by_check={check: 1},
    )
    assert status == STATUS_GUARDRAIL_FAIL
    assert any(check in r for r in reasons)


# --- rate-limit counter and live-mode honesty (N5, N6) ---------------------------------------------

import logging


@pytest.mark.parametrize("message,counted", [
    ("HTTP 429 Too Many Requests", True),
    ("google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED", True),
    ("Rate limit reached for model", True),
    ("done, timings_ms={'crossref': 14290}", False),
    ("DEGRADED after 4290 ms", False),
    ("fetched https://example.com/a-4291", False),
])
def test_rate_limit_counter_only_counts_rate_limit_lines(message, counted):
    counter = re_mod.RateLimitCounter()
    counter.emit(logging.LogRecord("x", logging.WARNING, __file__, 1, message, (), None))
    assert counter.count == (1 if counted else 0)


async def test_live_mode_does_not_report_a_made_up_call_tally(tmp_path, dataset, monkeypatch):
    async def fake_verify(text, bypass_cache=True):
        return _response()

    monkeypatch.setattr(re_mod, "verify_claim", fake_verify)
    report = await run_evaluation(dataset_paths=[dataset], output_path=tmp_path / "out.json", replay_mode="live")
    calls = report["summary"]["external_calls_per_claim"]
    assert calls["mean"] is None and calls["p50"] is None and "not counted" in calls["note"]
