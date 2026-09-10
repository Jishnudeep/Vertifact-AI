import pytest
from app.scoring import ScoringEngine, ScoreComponents, validate_weights

def test_weight_validation():
    # Valid weights
    assert validate_weights(0.50, 0.20, 0.30) is True
    assert validate_weights(0.40, 0.30, 0.30) is True
    assert validate_weights(0.60, 0.10, 0.30) is True
    
    # Invalid sum
    assert validate_weights(0.50, 0.20, 0.20) is False
    # w1 not dominant (<0.4)
    assert validate_weights(0.35, 0.35, 0.30) is False

def test_engine_initialization_asserts():
    # Valid initialization
    engine = ScoringEngine(0.50, 0.20, 0.30)
    assert engine.w1 == 0.50
    assert engine.w2 == 0.20
    assert engine.w3 == 0.30
    
    # Invalid initialization should raise AssertionError
    with pytest.raises(AssertionError):
        ScoringEngine(0.35, 0.35, 0.30)
        
    with pytest.raises(AssertionError):
        ScoringEngine(0.50, 0.20, 0.20)

def test_scoring_success_no_capping():
    """Verify standard scoring when all agents succeed and fact-check is found."""
    engine = ScoringEngine()
    # s_fact=0.8, b_ling=0.1 (so 1-b_ling=0.9), v_consensus=0.7, agents_succeeded=3
    # raw composite: 0.5*0.8 + 0.2*0.9 + 0.3*0.7 = 0.40 + 0.18 + 0.21 = 0.79
    result = engine.compute(
        s_fact=0.8,
        b_ling=0.1,
        v_consensus=0.7,
        agents_succeeded=3,
        fact_check_found=True
    )
    
    assert result.c_total == 0.790
    assert result.confidence_level == "high"
    assert len(result.warnings) == 0
    assert result.components == ScoreComponents(0.8, 0.1, 0.7)

def test_scoring_no_fact_check_low_consensus_capping():
    """Verify capping at 0.55 when there is no fact check and low source consensus."""
    # Using w1=0.4, w2=0.3, w3=0.3 to raise raw above 0.55
    engine = ScoringEngine(0.40, 0.30, 0.30)
    
    # s_fact=0.5 (neutral), b_ling=0.0 (1-b_ling=1.0), v_consensus=0.2 (low consensus < 0.3)
    # raw composite: 0.4*0.5 + 0.3*1.0 + 0.3*0.2 = 0.20 + 0.30 + 0.06 = 0.56
    # Should be capped at 0.55
    result = engine.compute(
        s_fact=0.5,
        b_ling=0.0,
        v_consensus=0.2,
        agents_succeeded=3,
        fact_check_found=False
    )
    
    assert result.c_total == 0.550
    assert result.confidence_level == "low"
    assert any("Low confidence: no fact-checks found" in w for w in result.warnings)

def test_scoring_half_true_preserves_fact_check_without_false_warning():
    """Verify that a legitimate 0.5 (e.g. Half True) does NOT trigger 'no fact-checks found' warning."""
    engine = ScoringEngine(0.40, 0.30, 0.30)
    result = engine.compute(
        s_fact=0.5,
        b_ling=0.0,
        v_consensus=0.2,
        agents_succeeded=3,
        fact_check_found=True  # Fact-checker explicitly rated it Half True / Mixed
    )
    
    # Not capped by Rule 2
    assert result.c_total == 0.560
    assert not any("no fact-checks found" in w for w in result.warnings)

def test_scoring_partial_analysis_capping():
    """Verify capping at 0.50 when some agents fail (partial analysis)."""
    engine = ScoringEngine()
    # s_fact=1.0, b_ling=0.0 (1-b_ling=1.0), v_consensus=1.0, agents_succeeded=2
    # raw composite: 0.5*1.0 + 0.2*1.0 + 0.3*1.0 = 1.0
    # Should be capped at 0.50
    result = engine.compute(
        s_fact=1.0,
        b_ling=0.0,
        v_consensus=1.0,
        agents_succeeded=2,
        fact_check_found=True
    )
    
    assert result.c_total == 0.500
    assert result.confidence_level == "medium"  # 2 agents succeeded, v_consensus=1.0 >= 0.3
    assert any("Partial analysis" in w for w in result.warnings)

def test_scoring_all_agents_fail():
    """Verify scoring when all agents fail (scoring engine computation level)."""
    engine = ScoringEngine()
    # s_fact=0.5, b_ling=0.5, v_consensus=0.0, agents_succeeded=0
    result = engine.compute(
        s_fact=0.5,
        b_ling=0.5,
        v_consensus=0.0,
        agents_succeeded=0,
        fact_check_found=False
    )
    
    assert result.c_total == 0.350  # 0.5*0.5 + 0.2*0.5 + 0.3*0.0 = 0.35, capped at 0.50 -> 0.35
    assert result.confidence_level == "insufficient"
    assert any("Partial analysis" in w for w in result.warnings)

def test_scoring_boundary_zeros():
    """Verify boundary conditions for lowest possible scores."""
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.0,
        b_ling=1.0,
        v_consensus=0.0,
        agents_succeeded=3,
        fact_check_found=True
    )
    
    assert result.c_total == 0.0
    assert result.confidence_level == "high"  # All succeeded, fact check found
    assert len(result.warnings) == 0

def test_scoring_boundary_ones():
    """Verify boundary conditions for highest possible scores."""
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=1.0,
        b_ling=0.0,
        v_consensus=1.0,
        agents_succeeded=3,
        fact_check_found=True
    )
    
    assert result.c_total == 1.0
    assert result.confidence_level == "high"
    assert len(result.warnings) == 0

def test_false_verdict_dominance_capping():
    """
    Verify R2.B1 guardrail: When fact_check_found=True and s_fact <= 0.25 (confirmed false claim),
    c_total MUST NOT be inflated by high news consensus (v_consensus=0.6).
    """
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.0,
        b_ling=0.1,
        v_consensus=0.6,
        agents_succeeded=3,
        fact_check_found=True
    )
    # raw composite = 0.5*0.0 + 0.2*0.9 + 0.3*0.6 = 0.36
    # Capped by Rule 3 to min(0.36, 0.0 + 0.15) = 0.150
    assert result.c_total <= 0.150
    assert result.confidence_level == "high"

def test_high_v_consensus_without_fact_check_is_medium_confidence():
    """
    Verify R2.B1 guardrail: High news consensus (v_consensus >= 0.5) without an explicit fact check match
    (fact_check_found=False) MUST cap confidence level at 'medium', NEVER 'high'.
    """
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.5,
        b_ling=0.1,
        v_consensus=0.8,
        agents_succeeded=3,
        fact_check_found=False
    )
    assert result.confidence_level == "medium"
