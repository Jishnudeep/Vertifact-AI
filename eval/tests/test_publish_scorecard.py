import json
from pathlib import Path
import pytest
from eval.publish_scorecard import generate_scorecard_markdown, publish_scorecard, ScorecardRefusalError


def test_refuses_when_run_status_is_invalid(tmp_path):
    run_file = tmp_path / "invalid_run.json"
    run_file.write_text(json.dumps({
        "summary": {
            "run_status": "INVALID_RUN",
            "run_status_reasons": ["zero source rate too high"],
            "timestamp": "2026-10-20T12:00:00Z",
        }
    }), encoding="utf-8")

    readme = tmp_path / "README.md"
    original_text = "# VeriFact AI\n\n<!-- eval-scorecard:start -->\n<!-- eval-scorecard:end -->\n"
    readme.write_text(original_text, encoding="utf-8")

    with pytest.raises(ScorecardRefusalError) as exc_info:
        publish_scorecard(baseline_path=run_file, readme_path=readme)

    assert "INVALID_RUN" in str(exc_info.value)
    # README must be completely untouched
    assert readme.read_text(encoding="utf-8") == original_text


def test_refuses_when_run_status_is_guardrail_fail(tmp_path):
    run_file = tmp_path / "guardrail_fail.json"
    run_file.write_text(json.dumps({
        "summary": {
            "run_status": "GUARDRAIL_FAIL",
            "run_status_reasons": ["bias leakage observed"],
            "timestamp": "2026-10-20T12:00:00Z",
        }
    }), encoding="utf-8")

    readme = tmp_path / "README.md"
    original_text = "# Header\nSome content\n"
    readme.write_text(original_text, encoding="utf-8")

    with pytest.raises(ScorecardRefusalError):
        publish_scorecard(baseline_path=run_file, readme_path=readme)

    assert readme.read_text(encoding="utf-8") == original_text


def test_renders_scorecard_when_run_status_ok(tmp_path):
    run_file = tmp_path / "valid_run.json"
    run_file.write_text(json.dumps({
        "summary": {
            "run_status": "OK",
            "run_status_reasons": [],
            "timestamp": "2026-10-20T12:00:00Z",
            "total_claims": 64,
            "verifiable_claims": 50,
            "mae": 0.15,
            "bucketed_accuracy": 0.76,
            "calibration_monotonic": True,
            "bias_leakage": {"passed": True, "source_leakage_rate": 0.0},
        }
    }), encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("# VeriFact AI\n\n<!-- eval-scorecard:start -->\nold content\n<!-- eval-scorecard:end -->\n", encoding="utf-8")

    publish_scorecard(baseline_path=run_file, readme_path=readme)
    updated = readme.read_text(encoding="utf-8")

    assert "<!-- eval-scorecard:start -->" in updated
    assert "<!-- eval-scorecard:end -->" in updated
    assert "old content" not in updated
    assert "2026-10-20T12:00:00Z" in updated
    assert "50" in updated
    assert "0.15" in updated
    assert "0.76" in updated
    # Optional files not provided -> renders not measured
    assert "not measured" in updated


def test_missed_target_rendered_truthfully(tmp_path):
    run_file = tmp_path / "valid_with_miss.json"
    run_file.write_text(json.dumps({
        "summary": {
            "run_status": "OK",
            "timestamp": "2026-10-20T12:00:00Z",
            "total_claims": 64,
            "verifiable_claims": 50,
            "mae": 0.25,  # target <= 0.20 (missed)
            "bucketed_accuracy": 0.65,  # target >= 0.70 (missed)
            "calibration_monotonic": False,
            "bias_leakage": {"passed": True, "source_leakage_rate": 0.0},
        }
    }), encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("<!-- eval-scorecard:start -->\n<!-- eval-scorecard:end -->", encoding="utf-8")

    publish_scorecard(baseline_path=run_file, readme_path=readme)
    updated = readme.read_text(encoding="utf-8")

    assert "missed" in updated
    assert "0.25" in updated
    assert "0.65" in updated


def test_idempotent_rendering(tmp_path):
    run_file = tmp_path / "valid_run.json"
    run_file.write_text(json.dumps({
        "summary": {
            "run_status": "OK",
            "timestamp": "2026-10-20T12:00:00Z",
            "total_claims": 64,
            "verifiable_claims": 50,
            "mae": 0.18,
            "bucketed_accuracy": 0.72,
            "calibration_monotonic": True,
            "bias_leakage": {"passed": True, "source_leakage_rate": 0.0},
        }
    }), encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("<!-- eval-scorecard:start -->\n<!-- eval-scorecard:end -->", encoding="utf-8")

    publish_scorecard(baseline_path=run_file, readme_path=readme)
    first_render = readme.read_text(encoding="utf-8")

    publish_scorecard(baseline_path=run_file, readme_path=readme)
    second_render = readme.read_text(encoding="utf-8")

    assert first_render == second_render
