from dataclasses import dataclass
from typing import List, Dict

from app.scoring.weights import DEFAULT_W1, DEFAULT_W2, DEFAULT_W3, validate_weights
from app.scoring.confidence import determine_confidence_level, apply_confidence_capping

@dataclass
class ScoreComponents:
    s_fact: float
    b_ling: float
    v_consensus: float

@dataclass
class ScoringResult:
    c_total: float
    components: ScoreComponents
    weights: Dict[str, float]
    confidence_level: str
    warnings: List[str]

class ScoringEngine:
    """
    Pure and deterministic scoring engine that computes the composite trust score (c_total).
    Applies confidence capping and overall confidence level determination.
    """
    def __init__(self, w1: float = DEFAULT_W1, w2: float = DEFAULT_W2, w3: float = DEFAULT_W3):
        assert validate_weights(w1, w2, w3), (
            f"Invalid weights w1={w1}, w2={w2}, w3={w3}: "
            f"must sum to 1.0 (got {w1+w2+w3}) and w1 must be dominant (>= 0.4)"
        )
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def compute(
        self,
        s_fact: float,
        b_ling: float,
        v_consensus: float,
        agents_succeeded: int,
        total_agents: int = 3,
    ) -> ScoringResult:
        """
        Compute trust score, apply capping rules, and classify confidence level.
        """
        # Clamp inputs to [0.0, 1.0] to safeguard against anomalous agent values
        s_fact_clamped = max(0.0, min(1.0, s_fact))
        b_ling_clamped = max(0.0, min(1.0, b_ling))
        v_consensus_clamped = max(0.0, min(1.0, v_consensus))
        
        # Raw composite: w1*s_fact + w2*(1 - b_ling) + w3*v_consensus
        c_total_raw = (
            self.w1 * s_fact_clamped
            + self.w2 * (1.0 - b_ling_clamped)
            + self.w3 * v_consensus_clamped
        )
        
        # Apply confidence capping rules
        c_total_capped, warnings = apply_confidence_capping(
            c_total_raw, s_fact_clamped, v_consensus_clamped, agents_succeeded, total_agents
        )
        
        # Determine confidence level
        confidence = determine_confidence_level(
            agents_succeeded, s_fact_clamped, v_consensus_clamped, total_agents
        )
        
        # Final round and clamp to [0.0, 1.0]
        c_total_final = round(max(0.0, min(1.0, c_total_capped)), 3)
        
        return ScoringResult(
            c_total=c_total_final,
            components=ScoreComponents(
                s_fact=s_fact_clamped,
                b_ling=b_ling_clamped,
                v_consensus=v_consensus_clamped
            ),
            weights={"w1": self.w1, "w2": self.w2, "w3": self.w3},
            confidence_level=confidence,
            warnings=warnings
        )
