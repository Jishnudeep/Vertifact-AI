"""
Bucketed accuracy per Eval Framework §2.3.

Both the predicted C_total and the ground-truth score are binned into three buckets:
    false  : 0.00 - 0.33
    mixed  : 0.34 - 0.66
    true    : 0.67 - 1.00
Accuracy is the fraction of claims whose predicted bucket equals the ground-truth bucket.
Target: >= 0.70 (informational only until a real, live-key baseline run exists).
"""
from typing import List, Dict, Any, Optional

BUCKET_EDGES = ((0.34, "false"), (0.67, "mixed"), (1.01, "true"))
BUCKET_LABELS = ("false", "mixed", "true")


def to_bucket(score: float) -> str:
    """Classify a 0.0-1.0 score into one of the three Eval Framework §2.3 buckets."""
    value = max(0.0, min(1.0, float(score)))
    for upper, label in BUCKET_EDGES:
        if value < upper:
            return label
    return "true"


def compute_bucketed_accuracy(
    predictions: List[float],
    ground_truths: List[float],
) -> Dict[str, Any]:
    """
    Compute 3-bucket classification accuracy plus a confusion matrix.

    Returns accuracy=None (not 0.0) when there is nothing to score, so a caller can
    distinguish "did not run" from "scored zero".
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")

    confusion: Dict[str, Dict[str, int]] = {
        actual: {predicted: 0 for predicted in BUCKET_LABELS} for actual in BUCKET_LABELS
    }

    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        p_bucket = to_bucket(pred)
        t_bucket = to_bucket(truth)
        confusion[t_bucket][p_bucket] += 1
        if p_bucket == t_bucket:
            correct += 1

    n = len(predictions)
    accuracy: Optional[float] = round(correct / n, 4) if n > 0 else None

    return {
        "accuracy": accuracy,
        "correct": correct,
        "n": n,
        "confusion": confusion,
        "target": 0.70,
    }
