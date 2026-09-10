from typing import List, Tuple

def determine_confidence_level(
    agents_succeeded: int,
    s_fact: float,
    v_consensus: float,
    total_agents: int = 3,
    fact_check_found: bool = False,
) -> str:
    """
    Determine the confidence level of the scoring based on TDD Section 7.
    - high: all agents succeeded AND an explicit fact-check verdict was found (fact_check_found=True)
    - medium: at least 2 agents succeeded and (v_consensus >= 0.3 or fact_check_found)
    - low: at least 1 agent succeeded
    - insufficient: no agents succeeded
    """
    # High confidence requires all agents succeeded AND an explicit fact-check verdict match
    if agents_succeeded == total_agents and fact_check_found:
        return "high"
    elif agents_succeeded >= 2 and (v_consensus >= 0.3 or fact_check_found):
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
) -> Tuple[float, List[str]]:
    """
    Applies the confidence capping rules from TDD Section 7.
    - If agents_succeeded < total_agents: cap c_total at 0.50 and warn
    - If not fact_check_found and v_consensus < 0.3: cap c_total at 0.55 and warn
    - If fact_check_found and s_fact <= 0.25: cap c_total at s_fact + 0.15 (max 0.25)
      so topical news coverage cannot inflate confirmed false claims.
    
    Returns:
        Tuple of (capped_c_total, list of warning strings)
    """
    warnings = []
    
    # Rule 1: Partial analysis cap
    if agents_succeeded < total_agents:
        c_total = min(c_total, 0.50)
        warnings.append(f"Partial analysis: {agents_succeeded}/{total_agents} agents succeeded")
        
    # Rule 2: Low confidence cap (no fact check found and low consensus)
    if not fact_check_found and v_consensus < 0.3:
        c_total = min(c_total, 0.55)
        warnings.append("Low confidence: no fact-checks found and insufficient corroborating sources")
        
    # Rule 3: Confirmed False Verdict Dominance Cap (R2.B1)
    if fact_check_found and s_fact <= 0.25:
        c_total = min(c_total, s_fact + 0.15)
        c_total = min(c_total, 0.25)

    return c_total, warnings
