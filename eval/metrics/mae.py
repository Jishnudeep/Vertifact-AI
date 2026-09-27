from typing import List, Optional

def compute_mae(predictions: List[float], ground_truths: List[float]) -> Optional[float]:
    """
    Compute Mean Absolute Error (MAE) between predictions and ground truths.

    Returns None when there is nothing to measure. R8.3: returning 0.0 on empty input
    reported a PERFECT score for a run that scored zero claims — the most misleading
    possible output for a metric whose target is "MAE <= 0.20".
    """
    if not predictions or not ground_truths:
        return None
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")

    total_err = sum(abs(p - gt) for p, gt in zip(predictions, ground_truths))
    return round(total_err / len(predictions), 4)
