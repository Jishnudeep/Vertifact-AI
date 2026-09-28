import pytest
from eval.metrics.mae import compute_mae
from eval.metrics.calibration import compute_calibration_curve
from eval.metrics.bias_leakage import evaluate_bias_leakage
from eval.metrics.bucketed_accuracy import compute_bucketed_accuracy, to_bucket
from eval.metrics.retrieval_recall import compute_recall_at_k

def test_compute_mae():
    preds = [0.1, 0.9, 0.5]
    targets = [0.0, 1.0, 0.5]
    mae = compute_mae(preds, targets)
    assert mae == round((0.1 + 0.1 + 0.0) / 3, 4)

def test_compute_mae_length_mismatch():
    with pytest.raises(ValueError):
        compute_mae([0.1], [0.1, 0.2])

def test_compute_calibration_curve_reports_mse():
    preds = [0.1, 0.3, 0.5, 0.7, 0.9]
    targets = [0.0, 0.2, 0.5, 0.8, 1.0]
    result = compute_calibration_curve(preds, targets, n_bins=5)
    assert result["sample_count"] == 5
    assert result["mse"] >= 0.0
    assert "brier_score" not in result

def test_calibration_monotonic_is_none_when_bins_are_too_sparse():
    """R7.P5: monotonicity over 1 populated bin is vacuous; report 'not evaluable'."""
    preds = [0.25, 0.26, 0.27]
    targets = [0.0, 0.5, 1.0]
    result = compute_calibration_curve(preds, targets, n_bins=5)
    assert result["monotonic"] is None

def test_calibration_monotonic_evaluated_with_enough_populated_bins():
    preds = [0.05, 0.06, 0.07, 0.45, 0.46, 0.47, 0.85, 0.86, 0.87]
    targets = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    result = compute_calibration_curve(preds, targets, n_bins=5)
    assert result["monotonic"] is True

def test_calibration_empty_input_is_not_evaluable():
    result = compute_calibration_curve([], [])
    assert result["monotonic"] is None
    assert result["mse"] is None
    assert result["sample_count"] == 0

def test_evaluate_bias_leakage():
    sources = [
        {"url": "https://snopes.com/check", "domain": "snopes.com", "bias_label": "Unrated"},
        {"url": "https://thelincolnproject.org/post", "domain": "thelincolnproject.org", "bias_label": "Unrated"},
        {"url": "https://reuters.com/news", "domain": "reuters.com", "bias_label": "Center"},
    ]
    res = evaluate_bias_leakage(sources)
    assert res["passed"] is True
    assert res["status"] == "evaluated"
    assert res["leakage_rate"] == 0.0

def test_evaluate_bias_leakage_catches_bridging_violation():
    sources = [
        {"url": "https://thelincolnproject.org/post", "domain": "thelincolnproject.org", "bias_label": "Center"},
    ]
    res = evaluate_bias_leakage(sources)
    assert res["passed"] is False
    assert res["leakage_rate"] == 1.0
    assert len(res["violations"]) == 1

def test_evaluate_bias_leakage_empty_input_is_not_evaluated():
    """R7.B3: a zero-tolerance guardrail that inspected nothing did NOT pass."""
    res = evaluate_bias_leakage([])
    assert res["passed"] is None
    assert res["status"] == "not_evaluated"
    assert res["leakage_rate"] is None
    assert res["total_sources"] == 0

def test_to_bucket_boundaries():
    # Eval Framework §2.3: false 0-0.33, mixed 0.34-0.66, true 0.67-1.0
    assert to_bucket(0.0) == "false"
    assert to_bucket(0.33) == "false"
    assert to_bucket(0.34) == "mixed"
    assert to_bucket(0.66) == "mixed"
    assert to_bucket(0.67) == "true"
    assert to_bucket(1.0) == "true"

def test_compute_bucketed_accuracy():
    preds = [0.1, 0.5, 0.9, 0.2]
    targets = [0.0, 0.5, 1.0, 0.9]  # last one is a miss (false predicted, true actual)
    res = compute_bucketed_accuracy(preds, targets)
    assert res["n"] == 4
    assert res["correct"] == 3
    assert res["accuracy"] == 0.75
    assert res["confusion"]["true"]["false"] == 1

def test_compute_bucketed_accuracy_empty_is_none():
    res = compute_bucketed_accuracy([], [])
    assert res["accuracy"] is None
    assert res["n"] == 0

def test_compute_bucketed_accuracy_length_mismatch():
    with pytest.raises(ValueError):
        compute_bucketed_accuracy([0.1], [0.1, 0.2])

def test_compute_recall_at_k():
    """retrieval_recall is NOT wired into the harness (needs dataset labels — Track 3.1)."""
    retrieved = ["snopes.com", "reuters.com", "politifact.com"]
    ground_truth = {"snopes.com", "politifact.com"}
    recall = compute_recall_at_k(retrieved, ground_truth, k=3)
    assert recall == 1.0


def test_replay_miss_invalidates_run():
    from eval.run_eval import assess_run_validity, STATUS_INVALID_RUN
    status, reasons = assess_run_validity(
        verifiable_preds=[0.2, 0.8], verifiable_targets=[0.0, 1.0],
        zero_source_rate=0.0, error_rate=0.0,
        bias_metrics={"passed": True}, bridging_violations=[], adversarial_results=[],
        replay_misses=3,
    )
    assert status == STATUS_INVALID_RUN
    assert any("replay" in r for r in reasons)

