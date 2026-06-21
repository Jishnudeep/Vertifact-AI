from typing import List, Tuple

def determine_confidence_level(
    agents_succeeded: int,
    s_fact: float,
    v_consensus: float,
    total_agents: int = 3
) -> str:
    """
    Determine the confidence level of the scoring based on TDD Section 7.
    - high: all agents succeeded and (s_fact != 0.5 or v_consensus >= 0.5)
    - medium: at least 2 agents succeeded and v_consensus >= 0.3
    - low: at least 1 agent succeeded
    - insufficient: no agents succeeded
    """
    if agents_succeeded == total_agents and (s_fact != 0.5 or v_consensus >= 0.5):
        return "high"
    elif agents_succeeded >= 2 and v_consensus >= 0.3:
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
    total_agents: int = 3
) -> Tuple[float, List[str]]:
    """
    Applies the confidence capping rules from TDD Section 7.
    - If agents_succeeded < total_agents: cap c_total at 0.50 and warn
    - If s_fact == 0.5 and v_consensus < 0.3: cap c_total at 0.55 and warn
    
    Returns:
        Tuple of (capped_c_total, list of warning strings)
    """
    warnings = []
    
    # Rule 1: Partial analysis cap
    if agents_succeeded < total_agents:
        c_total = min(c_total, 0.50)
        warnings.append(f"Partial analysis: {agents_succeeded}/{total_agents} agents succeeded")
        
    # Rule 2: Low confidence cap (no fact check and low consensus)
    if s_fact == 0.5 and v_consensus < 0.3:
        c_total = min(c_total, 0.55)
        warnings.append("Low confidence: no fact-checks found and insufficient corroborating sources")
        
    return c_total, warnings
