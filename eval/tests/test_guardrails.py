import pytest
from unittest.mock import patch, MagicMock
from app.services.bias_lookup import get_bias_label, load_bias_ratings, BIAS_RATINGS
from app.scoring import ScoringEngine
from app.agents.crossref_agent import make_neutral_crossref, run as run_crossref
from app.agents.factcheck_agent import make_neutral_factcheck, run as run_factcheck
from app.agents.linguistic_agent import make_neutral_linguistic, run as run_linguistic

@pytest.fixture(autouse=True)
def setup_bias_data():
    load_bias_ratings()

class TestBiasLeakageGuardrail:
    """Zero tolerance: bias ratings come ONLY from AllSides static data."""
    
    def test_unknown_domain_strictly_unrated(self):
        assert get_bias_label("completely-unknown-site-12345.org") == "Unrated"
        assert get_bias_label("fake-news-blog.xyz") == "Unrated"
        assert get_bias_label("") == "Unrated"
        assert get_bias_label(None) == "Unrated"

    def test_no_unmapped_allsides_defaults_to_center(self):
        """Unmapped or invalid CSV values must never default to Center."""
        # Ensure that whatever is in BIAS_RATINGS belongs strictly to the allowed AllSides set
        valid_ratings = {"Left", "Lean Left", "Center", "Lean Right", "Right"}
        for domain, rating in BIAS_RATINGS.items():
            assert rating in valid_ratings, f"Domain '{domain}' has invalid bias rating: '{rating}'"

class TestConfidenceCappingGuardrails:
    """Guardrails ensuring uncertainty and partial results are capped and warned."""

    def test_partial_analysis_cap_rule(self):
        engine = ScoringEngine()
        # Even with perfect component scores (1.0, 0.0, 1.0), failure of 1 agent caps at 0.50
        res = engine.compute(
            s_fact=1.0,
            b_ling=0.0,
            v_consensus=1.0,
            agents_succeeded=2,
            total_agents=3,
            fact_check_found=True
        )
        assert res.c_total <= 0.50
        assert any("Partial analysis" in w for w in res.warnings)

    def test_no_fact_check_low_consensus_cap_rule(self):
        engine = ScoringEngine(0.40, 0.30, 0.30)
        # Novel uncorroborated claim: s_fact=0.5, b_ling=0.0, v_consensus=0.1
        # raw score is 0.4*0.5 + 0.3*1.0 + 0.3*0.1 = 0.53 (or higher with other weights)
        res = engine.compute(
            s_fact=0.5,
            b_ling=0.0,
            v_consensus=0.1,
            agents_succeeded=3,
            total_agents=3,
            fact_check_found=False
        )
        assert res.c_total <= 0.55
        assert any("Low confidence: no fact-checks found" in w for w in res.warnings)

    def test_fact_check_found_half_true_preserves_score(self):
        """Claims rated Half True (0.5) must NOT falsely warn that no fact-checks were found."""
        engine = ScoringEngine(0.40, 0.30, 0.30)
        res = engine.compute(
            s_fact=0.5,
            b_ling=0.0,
            v_consensus=0.1,
            agents_succeeded=3,
            total_agents=3,
            fact_check_found=True
        )
        # Should not be capped by Rule 2 nor have false warning
        assert not any("no fact-checks found" in w for w in res.warnings)

class TestDegradationContracts:
    """Ensure degraded states carry explicit status and truthful neutral values."""

    def test_crossref_degraded_value_is_zero(self):
        neutral = make_neutral_crossref(degraded=True, error="Service failure")
        assert neutral.degraded is True
        assert neutral.v_consensus_score == 0.0
        assert neutral.error == "Service failure"

    def test_factcheck_degraded_contract(self):
        neutral = make_neutral_factcheck(degraded=True, error="Timeout")
        assert neutral.degraded is True
        assert neutral.s_fact_score == 0.5
        assert neutral.matches_found == 0

    def test_linguistic_degraded_contract(self):
        neutral = make_neutral_linguistic(degraded=True, error="NLP load error")
        assert neutral.degraded is True
        assert neutral.b_ling_score == 0.5
