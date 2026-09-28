"""
Calibrate the provisional NLI thresholds from Plans 1-2 on labeled pair sets.
  stance:       STANCE_MIN_PROB (crossref_agent) — maximize accuracy, ties -> stricter
  equivalence:  EQUIVALENCE_MIN_ENTAILMENT (claim_equivalence) — lowest threshold with ZERO
                false positives; a missed cache hit is cheap, a wrong one is a safety failure.
Usage: uv run python eval/calibrate_nli.py   (loads the real model; prints a JSON report)
Changing a threshold is a code change in the backend, made by hand and reviewed.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (BASE_DIR, BASE_DIR / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.services.nli import NLIScores, classify_pairs  # noqa: E402

DATASETS = BASE_DIR / "eval" / "datasets"
THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95


def _stance_label(s: NLIScores, t: float) -> str:
    if s.entailment >= t and s.entailment > s.contradiction:
        return "supports"
    if s.contradiction >= t and s.contradiction > s.entailment:
        return "refutes"
    return "neutral"


def sweep_stance(rows: List[Dict[str, Any]], scores: Sequence[NLIScores], thresholds: Sequence[float]) -> List[Dict[str, Any]]:
    out = []
    for t in thresholds:
        correct = sum(_stance_label(s, t) == r["label"] for r, s in zip(rows, scores))
        out.append({"threshold": t, "accuracy": round(correct / len(rows), 4)})
    return out


def sweep_equivalence(rows: List[Dict[str, Any]], scores: Sequence[Tuple[NLIScores, NLIScores]], thresholds: Sequence[float]) -> List[Dict[str, Any]]:
    out = []
    for t in thresholds:
        tp = fp = 0
        for r, (fwd, bwd) in zip(rows, scores):
            predicted = fwd.entailment >= t and bwd.entailment >= t
            if predicted and r["equivalent"]:
                tp += 1
            elif predicted and not r["equivalent"]:
                fp += 1
        out.append({"threshold": t, "true_positives": tp, "false_positives": fp})
    return out


def recommend(stance_sweep: List[Dict[str, Any]], equivalence_sweep: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    best_stance = max(stance_sweep, key=lambda r: (r["accuracy"], r["threshold"]))["threshold"]
    safe = [r for r in equivalence_sweep if r["false_positives"] == 0]
    best_equiv = min(safe, key=lambda r: r["threshold"])["threshold"] if safe else None
    return {"stance_min_prob": best_stance, "equivalence_min_entailment": best_equiv}


def run_calibration() -> Dict[str, Any]:
    from app.agents.crossref_agent import STANCE_MIN_PROB
    from app.services.claim_equivalence import EQUIVALENCE_MIN_ENTAILMENT, numbers_match

    stance_rows = json.loads((DATASETS / "stance_pairs.json").read_text(encoding="utf-8"))
    equiv_rows = json.loads((DATASETS / "equivalence_pairs.json").read_text(encoding="utf-8"))

    stance_scores = classify_pairs([(r["premise"], r["claim"]) for r in stance_rows])
    # The number guard runs before NLI in production; mirror that here.
    nli_rows = [r for r in equiv_rows if numbers_match(r["a"], r["b"])]
    flat = classify_pairs([p for r in nli_rows for p in ((r["b"], r["a"]), (r["a"], r["b"]))])
    equiv_scores = list(zip(flat[0::2], flat[1::2]))

    stance_sweep = sweep_stance(stance_rows, stance_scores, THRESHOLDS)
    equiv_sweep = sweep_equivalence(nli_rows, equiv_scores, THRESHOLDS)
    current_equiv = sweep_equivalence(nli_rows, equiv_scores, [EQUIVALENCE_MIN_ENTAILMENT])[0]
    return {
        "stance_sweep": stance_sweep,
        "equivalence_sweep": equiv_sweep,
        "recommendation": recommend(stance_sweep, equiv_sweep),
        "current": {
            "stance_min_prob": STANCE_MIN_PROB,
            "equivalence_min_entailment": EQUIVALENCE_MIN_ENTAILMENT,
            "equivalence_false_positives": current_equiv["false_positives"],
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_calibration(), indent=2))
