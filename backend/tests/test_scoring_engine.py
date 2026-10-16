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
        fact_check_found=True,
        verdict_found=True
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
        fact_check_found=True,  # Fact-checker explicitly rated it Half True / Mixed
        verdict_found=False
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
        fact_check_found=True,
        verdict_found=True
    )
    
    assert result.c_total == 0.0
    assert result.confidence_level == "high"  # All succeeded, definitive verdict found
    assert len(result.warnings) == 0

def test_scoring_boundary_ones():
    """Verify boundary conditions for highest possible scores."""
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=1.0,
        b_ling=0.0,
        v_consensus=1.0,
        agents_succeeded=3,
        fact_check_found=True,
        verdict_found=True
    )
    
    assert result.c_total == 1.0
    assert result.confidence_level == "high"
    assert len(result.warnings) == 0

def test_false_verdict_dominance_capping():
    """
    Verify R2.B1 guardrail: When verdict_found=True and s_fact <= 0.25 (confirmed false claim),
    c_total MUST NOT be inflated by high news consensus (v_consensus=0.6).
    """
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.0,
        b_ling=0.1,
        v_consensus=0.6,
        agents_succeeded=3,
        fact_check_found=True,
        verdict_found=True
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

def test_verdict_found_without_fact_check_found_is_rejected():
    """R7.D: verdict_found=True with fact_check_found=False is an impossible state and must assert."""
    engine = ScoringEngine()
    with pytest.raises(AssertionError):
        engine.compute(
            s_fact=0.0,
            b_ling=0.1,
            v_consensus=0.5,
            agents_succeeded=3,
            fact_check_found=False,
            verdict_found=True
        )


from app.scoring import determine_evidence_status
from app.scoring.confidence import DISPUTED_CAP


def test_disputed_claim_capped_into_red_band():
    """Rule 4: refuting reputable coverage and no definitive verdict -> capped at 0.33."""
    engine = ScoringEngine()
    # raw = 0.5*0.5 + 0.2*(1-0.0) + 0.3*0.0 = 0.45 -> capped to 0.33
    result = engine.compute(
        s_fact=0.5, b_ling=0.0, v_consensus=0.0, v_refute=0.6,
        agents_succeeded=3, fact_check_found=False, verdict_found=False,
    )
    assert result.c_total == DISPUTED_CAP
    assert result.evidence_status == "disputed"
    assert "Disputed: reputable sources contradict this claim" in result.warnings
    # Rule 2's "insufficient sources" warning is suppressed: Rule 4 is the accurate one
    assert not any(w.startswith("Low confidence") for w in result.warnings)
    assert result.confidence_level == "medium"
    assert result.components.v_refute == 0.6


def test_definitive_verdict_overrides_dispute():
    engine = ScoringEngine()
    # raw = 0.5*1.0 + 0.2*1.0 + 0.3*0.0 = 0.70; no Rule 4 because verdict_found
    result = engine.compute(
        s_fact=1.0, b_ling=0.0, v_consensus=0.0, v_refute=0.8,
        agents_succeeded=3, fact_check_found=True, verdict_found=True,
    )
    assert result.c_total == 0.7
    assert result.evidence_status == "fact_checked"
    assert not any(w.startswith("Disputed") for w in result.warnings)


def test_support_outweighing_refute_is_corroborated_not_disputed():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.5, b_ling=0.2, v_consensus=0.5, v_refute=0.4,
        agents_succeeded=3, fact_check_found=False, verdict_found=False,
    )
    assert result.evidence_status == "corroborated"
    assert not any(w.startswith("Disputed") for w in result.warnings)
    # raw = 0.25 + 0.16 + 0.15 = 0.56, no cap applies
    assert result.c_total == 0.56


def test_refute_below_threshold_is_not_disputed():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.5, b_ling=0.2, v_consensus=0.1, v_refute=0.2,
        agents_succeeded=3, fact_check_found=False, verdict_found=False,
    )
    assert result.evidence_status == "unverified"
    assert "Low confidence: no fact-checks found and insufficient corroborating sources" in result.warnings


def test_v_refute_is_clamped():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.5, b_ling=0.0, v_consensus=0.0, v_refute=1.7,
        agents_succeeded=3,
    )
    assert result.components.v_refute == 1.0


def test_default_v_refute_keeps_existing_behaviour():
    """Callers that don't pass v_refute get exactly the pre-C1 result."""
    engine = ScoringEngine()
    result = engine.compute(s_fact=0.8, b_ling=0.1, v_consensus=0.7, agents_succeeded=3,
                            fact_check_found=True, verdict_found=True)
    assert result.c_total == 0.79
    assert result.evidence_status == "fact_checked"


@pytest.mark.parametrize("fact_check_found,verdict_found,v_consensus,v_refute,expected", [
    (True,  True,  0.0, 0.9, "fact_checked"),   # definitive verdict wins over everything
    (True,  False, 0.0, 0.5, "disputed"),       # "Half True" match + refuting coverage
    (True,  False, 0.0, 0.0, "fact_checked"),   # "Half True" match alone is still a fact-check
    (False, False, 0.4, 0.0, "corroborated"),
    (False, False, 0.3, 0.3, "corroborated"),   # tie is not disputed (refute must exceed support)
    (False, False, 0.29, 0.0, "unverified"),
    (False, False, 0.0, 0.0, "unverified"),
])
def test_determine_evidence_status(fact_check_found, verdict_found, v_consensus, v_refute, expected):
    assert determine_evidence_status(fact_check_found, verdict_found, v_consensus, v_refute) == expected


def test_no_cap_rule_when_uncapped():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.9,
        b_ling=0.1,
        v_consensus=0.8,
        agents_succeeded=3,
        fact_check_found=True,
        verdict_found=True,
    )
    assert result.cap_rule is None
    assert result.raw_c_total == result.c_total


def test_partial_analysis_cap_rule():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.8,
        b_ling=0.1,
        v_consensus=0.8,
        agents_succeeded=2,
        fact_check_found=True,
        verdict_found=False,
    )
    assert result.cap_rule == "partial_analysis"
    assert result.c_total == 0.50


def test_disputed_beats_partial():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.8,
        b_ling=0.1,
        v_consensus=0.1,
        agents_succeeded=2,
        fact_check_found=True,
        verdict_found=False,
        v_refute=0.6,
    )
    assert result.cap_rule == "disputed"
    assert result.c_total == 0.33


def test_low_evidence_rule():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.78,
        b_ling=0.0,
        v_consensus=0.1,
        agents_succeeded=3,
        fact_check_found=False,
        verdict_found=False,
    )
    assert result.cap_rule == "low_evidence"
    assert result.c_total == 0.55


def test_false_verdict_rule():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.0,
        b_ling=0.0,
        v_consensus=0.5,
        agents_succeeded=3,
        fact_check_found=True,
        verdict_found=True,
    )
    assert result.cap_rule == "false_verdict"
    assert result.c_total == 0.15


def test_cap_rule_none_when_cap_not_binding():
    engine = ScoringEngine()
    result = engine.compute(
        s_fact=0.4,
        b_ling=0.5,
        v_consensus=0.333,
        agents_succeeded=2,
        fact_check_found=True,
        verdict_found=False,
    )
    assert result.cap_rule is None
    assert any("Partial analysis" in w for w in result.warnings)
    assert result.c_total == result.raw_c_total


def test_c_total_unchanged_by_transparency():
    cases = [
        (0.9, 0.1, 0.8, 3, True, True, 0.0),
        (0.8, 0.1, 0.8, 2, True, False, 0.0),
        (0.8, 0.1, 0.1, 2, True, False, 0.6),
        (0.78, 0.0, 0.1, 3, False, False, 0.0),
        (0.0, 0.0, 0.5, 3, True, True, 0.0),
        (0.4, 0.5, 0.333, 2, True, False, 0.0),
    ]
    engine = ScoringEngine()
    from app.scoring.confidence import apply_confidence_capping
    for s_fact, b_ling, v_cons, ag_succ, fc_found, v_found, v_ref in cases:
        res = engine.compute(
            s_fact=s_fact, b_ling=b_ling, v_consensus=v_cons,
            agents_succeeded=ag_succ, fact_check_found=fc_found,
            verdict_found=v_found, v_refute=v_ref,
        )
        raw = engine.w1 * s_fact + engine.w2 * (1.0 - b_ling) + engine.w3 * v_cons
        capped, _ = apply_confidence_capping(
            c_total=raw, s_fact=s_fact, v_consensus=v_cons,
            agents_succeeded=ag_succ, fact_check_found=fc_found,
            verdict_found=v_found, v_refute=v_ref,
        )
        expected = round(max(0.0, min(1.0, capped)), 3)
        assert res.c_total == expected


def test_cap_rule_consistency_grid():
    """
    Grid regression test asserting that apply_confidence_capping and binding_cap_rule
    always agree across inputs and boundaries.
    """
    import itertools
    grid = [i / 10 for i in range(11)] + [0.33, 0.50, 0.55, 0.25]
    sfacts = [0.0, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0]
    caps = {"false_verdict": None, "disputed": 0.33, "partial_analysis": 0.50, "low_evidence": 0.55}

    engine = ScoringEngine()
    for s, b, vc in itertools.product(sfacts, grid, grid):
        for vr in (0.0, 0.2, 0.35, 0.8):
            for ag in (1, 2, 3):
                for fc, vf in ((False, False), (True, False), (True, True)):
                    r = engine.compute(
                        s_fact=s, b_ling=b, v_consensus=vc,
                        agents_succeeded=ag, fact_check_found=fc,
                        verdict_found=vf, v_refute=vr,
                    )
                    if r.cap_rule is None:
                        assert r.c_total == r.raw_c_total, f"Uncapped result has c_total != raw_c_total: {r}"
                    else:
                        exp = min(max(0, min(1, s)) + 0.15, 0.25) if r.cap_rule == "false_verdict" else caps[r.cap_rule]
                        assert abs(r.c_total - round(exp, 3)) < 1e-6 and r.c_total < r.raw_c_total, (
                            f"Cap rule inconsistency for rule {r.cap_rule}: raw={r.raw_c_total}, c_total={r.c_total}, exp={exp}"
                        )

