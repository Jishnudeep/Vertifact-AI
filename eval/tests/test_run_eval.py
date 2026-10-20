import json
from pathlib import Path
import pytest
from app.config import settings
from eval.run_eval import run_evaluation, _parse_args


def test_cli_args_parsing_timeouts():
    args = _parse_args(["--agent-timeout", "12.5", "--request-deadline", "18.0"])
    assert args.agent_timeout == 12.5
    assert args.request_deadline == 18.0


@pytest.mark.anyio
async def test_run_evaluation_records_timeouts_in_summary(tmp_path, monkeypatch):
    import eval.run_eval as re_mod

    # Create dummy claim dataset
    dataset_file = tmp_path / "test_dataset.json"
    claim = {
        "id": "c_timeout_test",
        "claim_text": "The sky is blue.",
        "category": "science",
        "claim_type": "verifiable",
        "ground_truth_score": 0.9,
        "expected_confidence": "high",
        "origin_domain": None,
    }
    dataset_file.write_text(json.dumps([claim]), encoding="utf-8")

    async def fake_verify_claim(text, bypass_cache=True):
        return {
            "trust_score": {"c_total": 0.85},
            "confidence": {
                "level": "high",
                "evidence_status": "corroborated",
                "verdict_found": True,
                "warnings": [],
            },
            "sources": [],
            "processing_time_ms": 100.0,
            "agent_executions": {
                "factcheck": {"status": "ok", "duration_ms": 40, "error_code": None},
                "crossref": {"status": "ok", "duration_ms": 40, "error_code": None},
                "linguistic": {"status": "ok", "duration_ms": 20, "error_code": None},
            },
        }

    monkeypatch.setattr(re_mod, "verify_claim", fake_verify_claim)

    out_file = tmp_path / "out.json"
    report = await run_evaluation(
        dataset_paths=[dataset_file],
        output_path=out_file,
        replay_mode="replay",
        agent_timeout_s=25.0,
        request_deadline_s=30.0,
    )

    summary = report["summary"]
    assert "agent_timeout_s" in summary
    assert "request_deadline_s" in summary
    assert summary["agent_timeout_s"] == 25.0
    assert summary["request_deadline_s"] == 30.0
