from typing import List, Dict, Any, Optional

# A bin needs at least this many samples before it can support a monotonicity claim.
MIN_BIN_COUNT = 3
# Monotonicity needs at least this many sufficiently-populated bins to mean anything.
MIN_POPULATED_BINS = 3


def compute_calibration_curve(
    predictions: List[float],
    ground_truths: List[float],
    n_bins: int = 5
) -> Dict[str, Any]:
    """
    Compute a calibration curve across n_bins equal-width bins in [0, 1].

    `mse` is the mean squared error of the predicted C_total against the continuous
    ground-truth score. It is deliberately NOT called a Brier score: the Brier score is
    defined against binary outcomes, whereas our ground truth is a continuous credibility
    score, so the two are not comparable.

    `monotonic` is None ("not evaluable") unless at least MIN_POPULATED_BINS bins hold
    MIN_BIN_COUNT samples each. Note that the curated dataset uses 3-level ground truth
    (0.0 / 0.5 / 1.0), which structurally limits how informative this metric can be —
    a finer-grained ground truth is a Track 3.1 deliverable.
    """
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return {"bins": [], "monotonic": None, "mse": None, "sample_count": 0}

    bin_width = 1.0 / n_bins
    bins = []

    for i in range(n_bins):
        low = i * bin_width
        high = (i + 1) * bin_width
        bin_preds = []
        bin_gts = []
        for p, gt in zip(predictions, ground_truths):
            if i == n_bins - 1:
                in_bin = (low <= p <= high)
            else:
                in_bin = (low <= p < high)
            if in_bin:
                bin_preds.append(p)
                bin_gts.append(gt)

        count = len(bin_preds)
        avg_pred = sum(bin_preds) / count if count > 0 else (low + high) / 2
        avg_actual = sum(bin_gts) / count if count > 0 else 0.0

        bins.append({
            "bin": i,
            "range": [round(low, 2), round(high, 2)],
            "count": count,
            "mean_predicted": round(avg_pred, 4),
            "mean_actual": round(avg_actual, 4),
        })

    # Monotonicity is only asserted over sufficiently-populated bins
    populated = [b for b in bins if b["count"] >= MIN_BIN_COUNT]
    monotonic: Optional[bool]
    if len(populated) < MIN_POPULATED_BINS:
        monotonic = None
    else:
        actuals = [b["mean_actual"] for b in populated]
        monotonic = all(actuals[j] <= actuals[j + 1] for j in range(len(actuals) - 1))

    mse = round(sum((p - gt) ** 2 for p, gt in zip(predictions, ground_truths)) / len(predictions), 4)

    return {
        "bins": bins,
        "monotonic": monotonic,
        "mse": mse,
        "sample_count": len(predictions)
    }
