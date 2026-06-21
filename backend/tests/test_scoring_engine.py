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
        agents_succeeded=3
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
        agents_succeeded=3
    )
    
    assert result.c_total == 0.550
    assert result.confidence_level == "low"
    assert any("Low confidence" in w for w in result.warnings)

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
        agents_succeeded=2
    )
    
    assert result.c_total == 0.500
    assert result.confidence_level == "medium"  # 2 agents succeeded, v_consensus=1.0 >= 0.3
    assert any("Partial analysis" in w for w in result.warnings)

def test_scoring_all_agents_fail():
    """Verify scoring when all agents fail."""
    engine = ScoringEngine()
    # s_fact=0.5, b_ling=0.5, v_consensus=0.5, agents_succeeded=0
    result = engine.compute(
        s_fact=0.5,
        b_ling=0.5,
        v_consensus=0.5,
        agents_succeeded=0
    )
    
    assert result.c_total == 0.500  # Capped at 0.50 due to partial analysis
    assert result.confidence_level == "insufficient"
    assert any("Partial analysis" in w for w in result.warnings)

def test_scoring_boundary_zeros():
    """Verify boundary conditions for lowest possible scores."""
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.0,
        b_ling=1.0,
        v_consensus=0.0,
        agents_succeeded=3
    )
    
    assert result.c_total == 0.0
    assert result.confidence_level == "high"  # All succeeded, s_fact=0.0 != 0.5
    assert len(result.warnings) == 0

def test_scoring_boundary_ones():
    """Verify boundary conditions for highest possible scores."""
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=1.0,
        b_ling=0.0,
        v_consensus=1.0,
        agents_succeeded=3
    )
    
    assert result.c_total == 1.0
    assert result.confidence_level == "high"
    assert len(result.warnings) == 0
