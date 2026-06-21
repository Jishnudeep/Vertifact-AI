# Default weights
DEFAULT_W1 = 0.50  # Fact-check (dominant)
DEFAULT_W2 = 0.20  # Linguistic Analysis (1 - b_ling)
DEFAULT_W3 = 0.30  # Cross-Reference & Source Consensus

def validate_weights(w1: float, w2: float, w3: float) -> bool:
    """
    Validates that:
    1. Sum of weights is exactly 1.0 (with floating-point tolerance).
    2. Fact-check weight w1 is dominant (w1 >= 0.4).
    """
    return abs(w1 + w2 + w3 - 1.0) < 1e-6 and w1 >= 0.4
