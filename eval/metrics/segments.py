"""
Segmented metrics (C3/C4). Blended metrics hide two different systems: claims with a
matched fact-check (S_fact carries the verdict) and claims without one (score bounded to
roughly 0.25-0.75 by construction). Report them apart, and treat evidence_status
"unverified" as an abstention: selective accuracy over answered claims, plus coverage.
"""
from collections import defaultdict
from typing import Any, Dict, List

from eval.metrics.bucketed_accuracy import compute_bucketed_accuracy
from eval.metrics.mae import compute_mae

ANSWERED_STATUSES = frozenset({"fact_checked", "disputed", "corroborated"})


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    preds = [float(r["predicted_c_total"]) for r in records]
    gts = [float(r["ground_truth"]) for r in records]
    return {
        "n": len(records),
        "mae": compute_mae(preds, gts),
        "bucketed_accuracy": compute_bucketed_accuracy(preds, gts)["accuracy"],
    }


def compute_segment_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_status: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_dataset[r.get("source_dataset") or "curated"].append(r)
        by_status[r.get("evidence_status") or "missing"].append(r)

    answered = [r for r in records if r.get("evidence_status") in ANSWERED_STATUSES]
    selective_summary = _summarize(answered)
    return {
        "by_dataset": {k: _summarize(v) for k, v in sorted(by_dataset.items())},
        "by_evidence_status": {k: _summarize(v) for k, v in sorted(by_status.items())},
        "selective": {
            "coverage": round(len(answered) / len(records), 4) if records else None,
            "n_answered": len(answered),
            "mae": selective_summary["mae"],
            "bucketed_accuracy": selective_summary["bucketed_accuracy"],
        },
    }
