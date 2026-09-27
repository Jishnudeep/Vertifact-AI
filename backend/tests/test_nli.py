import pytest

from app.services import nli
from app.services.nli import NLIScores, classify_pairs, _resolve_label_index


class _FakeConfig:
    def __init__(self, id2label):
        self.id2label = id2label


class _FakeHFModel:
    def __init__(self, id2label):
        self.config = _FakeConfig(id2label)


class _FakeCrossEncoder:
    """Stands in for sentence_transformers.CrossEncoder: returns fixed logits."""

    def __init__(self, rows, id2label):
        self.rows = rows
        self.model = _FakeHFModel(id2label)
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return self.rows


@pytest.fixture
def fake_model(monkeypatch):
    def _install(rows, id2label=None):
        if id2label is None:
            id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
        model = _FakeCrossEncoder(rows, id2label)
        monkeypatch.setattr(nli, "_model", model)
        monkeypatch.setattr(nli, "_label_index", _resolve_label_index(model))
        return model
    return _install


def test_empty_pairs_do_not_load_model(monkeypatch):
    def _boom():
        raise AssertionError("model must not load for empty input")
    monkeypatch.setattr(nli, "get_nli_model", _boom)
    assert classify_pairs([]) == []


def test_logits_are_softmaxed_and_mapped_by_label(fake_model):
    fake_model([[0.0, 6.0, 0.0], [6.0, 0.0, 0.0]])
    scores = classify_pairs([("p1", "h1"), ("p2", "h2")])
    assert scores[0].entailment > 0.99
    assert scores[1].contradiction > 0.99
    for s in scores:
        assert isinstance(s, NLIScores)
        assert abs(s.entailment + s.contradiction + s.neutral - 1.0) < 1e-6


def test_label_order_follows_model_config(fake_model):
    # A model whose classification head is ordered differently must still map correctly.
    fake_model([[6.0, 0.0, 0.0]], id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
    [s] = classify_pairs([("p", "h")])
    assert s.entailment > 0.99


def test_generic_labels_fall_back_to_documented_order(fake_model):
    # Documented order for cross-encoder/nli-deberta-v3-small: contradiction, entailment, neutral.
    fake_model([[0.0, 6.0, 0.0]], id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
    [s] = classify_pairs([("p", "h")])
    assert s.entailment > 0.99


def test_row_count_mismatch_raises(fake_model):
    fake_model([[0.0, 6.0, 0.0]])
    with pytest.raises(RuntimeError):
        classify_pairs([("p1", "h1"), ("p2", "h2")])


@pytest.mark.live
def test_real_model_entailment_and_contradiction():
    """Loads the real cross-encoder (downloads ~0.5GB on first run)."""
    supports, refutes = classify_pairs([
        ("A man is eating pizza in the park.", "A man eats something."),
        ("The bridge was closed to all traffic on Monday.", "The bridge was open on Monday."),
    ])
    assert supports.entailment > supports.contradiction
    assert refutes.contradiction > refutes.entailment
