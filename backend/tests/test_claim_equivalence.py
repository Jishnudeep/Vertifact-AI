from unittest.mock import patch

import pytest

from app.services.claim_equivalence import are_equivalent, numbers_match
from app.services.nli import NLIScores

HIGH = NLIScores(entailment=0.95, contradiction=0.02, neutral=0.03)
LOW = NLIScores(entailment=0.10, contradiction=0.10, neutral=0.80)
CONTRA = NLIScores(entailment=0.02, contradiction=0.95, neutral=0.03)


def test_identical_claims_skip_the_model():
    with patch("app.services.claim_equivalence.classify_pairs") as mock_nli:
        assert are_equivalent("NASA found water on Mars", "  nasa found WATER on mars ") is True
        mock_nli.assert_not_called()


@pytest.mark.parametrize("a,b,expected", [
    ("Inflation hit 3% in May", "Inflation hit 8% in May", False),
    ("Unemployment was 3.5 percent", "Unemployment was 3.5 percent last month", True),
    ("Unemployment was 3.5%", "Unemployment was 3.5 percent", True),
    ("The bill cost $1,000,000", "The bill cost $1000000", True),
    ("Built in 1999", "Built in 2001", False),
    ("No numbers here", "Still no numbers", True),
])
def test_numbers_match(a, b, expected):
    assert numbers_match(a, b) is expected


def test_number_mismatch_is_not_equivalent_and_skips_the_model():
    with patch("app.services.claim_equivalence.classify_pairs") as mock_nli:
        assert are_equivalent("Inflation hit 3% in May", "Inflation hit 8% in May") is False
        mock_nli.assert_not_called()


def test_bidirectional_entailment_is_equivalent():
    with patch("app.services.claim_equivalence.classify_pairs", return_value=[HIGH, HIGH]) as mock_nli:
        assert are_equivalent("NASA confirms water on Mars", "NASA has confirmed water exists on Mars") is True
    pairs = mock_nli.call_args[0][0]
    assert pairs == [
        ("NASA has confirmed water exists on Mars", "NASA confirms water on Mars"),
        ("NASA confirms water on Mars", "NASA has confirmed water exists on Mars"),
    ]


def test_one_way_entailment_is_not_equivalent():
    """'visited Paris' entails 'visited France' but not the reverse: different claims."""
    with patch("app.services.claim_equivalence.classify_pairs", return_value=[HIGH, LOW]):
        assert are_equivalent("The president visited France", "The president visited Paris") is False


def test_contradiction_is_not_equivalent():
    with patch("app.services.claim_equivalence.classify_pairs", return_value=[CONTRA, CONTRA]):
        assert are_equivalent("Vaccines cause autism", "Vaccines do not cause autism") is False


@pytest.mark.live
@pytest.mark.parametrize("a,b,expected", [
    ("Vaccines cause autism", "Vaccines do not cause autism", False),
    ("The Eiffel Tower is in Paris", "The Eiffel Tower is not in Paris", False),
    ("NASA confirms water found on Mars", "NASA has confirmed that water was found on Mars", True),
])
def test_real_model_equivalence(a, b, expected):
    assert are_equivalent(a, b) is expected
