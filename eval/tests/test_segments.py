from eval.metrics.segments import compute_segment_metrics


def _r(ds, status, pred, gt):
    return {"source_dataset": ds, "evidence_status": status, "predicted_c_total": pred, "ground_truth": gt}


def test_groups_by_dataset_and_status():
    records = [
        _r("curated", "fact_checked", 0.1, 0.0),
        _r("curated", "unverified", 0.5, 1.0),
        _r("liar", "fact_checked", 0.9, 1.0),
    ]
    out = compute_segment_metrics(records)
    assert out["by_dataset"]["curated"]["n"] == 2
    assert out["by_dataset"]["liar"]["mae"] == 0.1
    assert out["by_evidence_status"]["unverified"]["n"] == 1


def test_selective_metrics_exclude_unverified_and_report_coverage():
    records = [
        _r("curated", "fact_checked", 0.1, 0.0),   # answered, correct bucket
        _r("curated", "disputed", 0.3, 0.0),       # answered, correct bucket
        _r("curated", "unverified", 0.5, 1.0),     # abstained
        _r("curated", "unverified", 0.5, 0.0),     # abstained
    ]
    sel = compute_segment_metrics(records)["selective"]
    assert sel["coverage"] == 0.5
    assert sel["n_answered"] == 2
    assert sel["bucketed_accuracy"] == 1.0


def test_empty_input_reports_none_not_perfect():
    out = compute_segment_metrics([])
    assert out["selective"] == {"coverage": None, "n_answered": 0, "mae": None, "bucketed_accuracy": None}
    assert out["by_dataset"] == {} and out["by_evidence_status"] == {}


def test_missing_evidence_status_is_its_own_segment():
    out = compute_segment_metrics([_r("curated", None, 0.5, 0.5)])
    assert out["by_evidence_status"]["missing"]["n"] == 1
    assert out["selective"]["n_answered"] == 0   # missing status never counts as an answer
