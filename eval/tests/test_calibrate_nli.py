import pytest

from app.services.nli import NLIScores
from eval.calibrate_nli import recommend, sweep_equivalence, sweep_stance


def S(e, c, n):
    return NLIScores(entailment=e, contradiction=c, neutral=n)


def test_stance_sweep_accuracy_per_threshold():
    rows = [{"label": "supports"}, {"label": "refutes"}, {"label": "neutral"}]
    scores = [S(0.7, 0.2, 0.1), S(0.1, 0.65, 0.25), S(0.3, 0.3, 0.4)]
    sweep = {r["threshold"]: r for r in sweep_stance(rows, scores, [0.6, 0.7])}
    assert sweep[0.6]["accuracy"] == 1.0
    assert sweep[0.7]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)   # refutes at 0.65 falls to neutral


def test_equivalence_sweep_counts_false_positives():
    rows = [{"a": "x", "b": "y", "equivalent": True}, {"a": "p", "b": "q", "equivalent": False}]
    # each row has (forward, backward) scores
    scores = [(S(0.9, 0.05, 0.05), S(0.85, 0.05, 0.1)), (S(0.82, 0.1, 0.08), S(0.81, 0.1, 0.09))]
    sweep = {r["threshold"]: r for r in sweep_equivalence(rows, scores, [0.8, 0.83])}
    assert sweep[0.8]["false_positives"] == 1
    assert sweep[0.83]["false_positives"] == 0
    assert sweep[0.83]["true_positives"] == 1


def test_recommend_requires_zero_false_positives():
    stance = [{"threshold": 0.6, "accuracy": 0.9}, {"threshold": 0.7, "accuracy": 0.9}]
    equiv = [{"threshold": 0.8, "false_positives": 1, "true_positives": 5},
             {"threshold": 0.85, "false_positives": 0, "true_positives": 4},
             {"threshold": 0.9, "false_positives": 0, "true_positives": 2}]
    rec = recommend(stance, equiv)
    assert rec["stance_min_prob"] == 0.7          # tie -> higher threshold
    assert rec["equivalence_min_entailment"] == 0.85


def test_recommend_returns_none_when_no_safe_equivalence_threshold():
    rec = recommend([{"threshold": 0.6, "accuracy": 0.8}],
                    [{"threshold": 0.9, "false_positives": 2, "true_positives": 3}])
    assert rec["equivalence_min_entailment"] is None


@pytest.mark.live
def test_committed_thresholds_have_zero_equivalence_false_positives():
    """Guardrail with the real model: the shipped EQUIVALENCE_MIN_ENTAILMENT must never
    call a negation/antonym/entity-swap/scope/number pair equivalent."""
    from eval.calibrate_nli import run_calibration
    report = run_calibration()
    assert report["current"]["equivalence_false_positives"] == 0
