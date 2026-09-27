"""NOT WIRED — requires known_factcheck_url labels in the dataset. Track 3.1."""
from typing import List, Optional, Set

def compute_recall_at_k(
    retrieved_domains: List[str],
    ground_truth_domains: Set[str],
    k: int = 5,
) -> Optional[float]:
    """
    Compute Recall@k for retrieved source domains against expected authoritative domains.

    Returns None when there are no ground-truth domains to recall. R8.3: returning 1.0
    there declared perfect retrieval on an unlabelled dataset — the same vacuous-green
    trap as MAE-on-empty. This metric is still not wired into the harness, but the trap
    is removed so wiring it later cannot silently publish a fake 1.0.
    """
    if not ground_truth_domains:
        return None
    top_k = set(retrieved_domains[:k])
    matched = top_k.intersection(ground_truth_domains)
    return round(len(matched) / len(ground_truth_domains), 4)
