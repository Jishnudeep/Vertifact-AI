from typing import List, Tuple

# Stance-aware thresholds (C1 / D2). Track 3.1: calibrate against the eval set (Plan 3).
CORROBORATED_MIN = 0.3      # v_consensus at/above this counts as real corroboration
DISPUTED_REFUTE_MIN = 0.3   # v_refute at/above this, AND above v_consensus, marks a claim disputed
DISPUTED_CAP = 0.33         # top of the red gauge band (TDD §9.1: 0-33 = likely false)

EVIDENCE_STATUSES = ("fact_checked", "disputed", "corroborated", "unverified")


def is_disputed(v_consensus: float, v_refute: float, verdict_found: bool) -> bool:
    """Reputable coverage predominantly refutes the claim and no definitive verdict overrides it."""
    return (not verdict_found) and v_refute >= DISPUTED_REFUTE_MIN and v_refute > v_consensus


def determine_evidence_status(
    fact_check_found: bool,
    verdict_found: bool,
    v_consensus: float,
    v_refute: float,
) -> str:
    """
    Classify what kind of evidence backs the score (D2). The UI shows "Not enough evidence"
    instead of a gauge value when the status is "unverified".
    - fact_checked: a definitive verdict, or any matched review not contradicted by coverage
    - disputed:     reputable coverage refutes the claim and no definitive verdict exists
    - corroborated: no fact-check, but supporting reputable coverage >= CORROBORATED_MIN
    - unverified:   none of the above
    """
    if verdict_found:
        return "fact_checked"
    if is_disputed(v_consensus, v_refute, verdict_found):
        return "disputed"
    if fact_check_found:
        return "fact_checked"
    if v_consensus >= CORROBORATED_MIN:
        return "corroborated"
    return "unverified"


def determine_confidence_level(
    agents_succeeded: int,
    s_fact: float,
    v_consensus: float,
    total_agents: int = 3,
    fact_check_found: bool = False,
    verdict_found: bool = False,
    v_refute: float = 0.0,
) -> str:
    """
    Determine the confidence level of the scoring based on TDD Section 7.
    - high: all agents succeeded AND an explicit definitive fact-check verdict was found (verdict_found=True)
    - medium: at least 2 agents succeeded and there is real evidence either way
      (v_consensus >= 0.3, v_refute >= 0.3, or a matched fact-check)
    - low: at least 1 agent succeeded
    - insufficient: no agents succeeded
    """
    # High confidence requires all agents succeeded AND an explicit definitive fact-check verdict
    if agents_succeeded == total_agents and verdict_found:
        return "high"
    elif agents_succeeded >= 2 and (
        v_consensus >= CORROBORATED_MIN
        or v_refute >= DISPUTED_REFUTE_MIN
        or fact_check_found
        or verdict_found
    ):
        return "medium"
    elif agents_succeeded >= 1:
        return "low"
    else:
        return "insufficient"


def apply_confidence_capping(
    c_total: float,
    s_fact: float,
    v_consensus: float,
    agents_succeeded: int,
    total_agents: int = 3,
    fact_check_found: bool = False,
    verdict_found: bool = False,
    v_refute: float = 0.0,
) -> Tuple[float, List[str]]:
    """
    Applies the confidence capping rules from TDD Section 7.
    - Rule 1: If agents_succeeded < total_agents: cap c_total at 0.50 and warn
    - Rule 2: If not fact_check_found and v_consensus < 0.3 (and not disputed): cap c_total at 0.55 and warn
    - Rule 3: If verdict_found and s_fact <= 0.25: cap c_total at s_fact + 0.15 (max 0.25)
      so topical news coverage cannot inflate confirmed false claims.
    - Rule 4: If disputed (v_refute >= 0.3, v_refute > v_consensus, no definitive verdict):
      cap c_total at 0.33 and warn, so refuting coverage lands the claim in the red band.

    Returns:
        Tuple of (capped_c_total, list of warning strings)
    """
    warnings = []
    disputed = is_disputed(v_consensus, v_refute, verdict_found)

    # Rule 1: Partial analysis cap
    if agents_succeeded < total_agents:
        c_total = min(c_total, 0.50)
        warnings.append(f"Partial analysis: {agents_succeeded}/{total_agents} agents succeeded")

    # Rule 2: Low confidence cap (no fact check found and low consensus). Skipped when
    # disputed: Rule 4's cap is stricter and its warning is the accurate one.
    if not fact_check_found and v_consensus < CORROBORATED_MIN and not disputed:
        c_total = min(c_total, 0.55)
        warnings.append("Low confidence: no fact-checks found and insufficient corroborating sources")

    # Rule 3: Confirmed False Verdict Dominance Cap (R2.B1 / R4.A)
    # Uses verdict_found so dominance capping fires when a definitive false verdict exists
    if verdict_found and s_fact <= 0.25:
        c_total = min(c_total, s_fact + 0.15)
        c_total = min(c_total, 0.25)

    # Rule 4: Disputed cap (C1). Refuting reputable coverage must pull the claim into the red
    # band, never leave it amber; a definitive verdict (Rule 3 territory) overrides it.
    if disputed:
        c_total = min(c_total, DISPUTED_CAP)
        warnings.append("Disputed: reputable sources contradict this claim")

    return c_total, warnings
