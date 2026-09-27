import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# Documented label order of cross-encoder/nli-deberta-v3-small. Used only when the model
# config carries generic labels (LABEL_0..2) instead of named ones.
_FALLBACK_LABELS = ("contradiction", "entailment", "neutral")
_EXPECTED_LABELS = frozenset(_FALLBACK_LABELS)

# Lazy loaded module-level CrossEncoder instance (preloaded once during startup lifespan)
_model = None
_label_index: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class NLIScores:
    """Softmax probabilities for one (premise, hypothesis) pair; they sum to 1.0."""
    entailment: float
    contradiction: float
    neutral: float


def _resolve_label_index(model) -> Dict[str, int]:
    """Map label name -> logit column, trusting the model config over the documented order."""
    id2label = None
    try:
        id2label = model.model.config.id2label
    except AttributeError:
        logger.warning("NLI model exposes no config.id2label; using documented label order.")

    labels = [str(id2label[i]).lower() for i in range(len(id2label))] if id2label else []
    if set(labels) != _EXPECTED_LABELS:
        logger.warning(f"NLI model labels {labels} are not named NLI labels; using documented order.")
        labels = list(_FALLBACK_LABELS)
    return {label: i for i, label in enumerate(labels)}


def get_nli_model():
    """
    Retrieve or initialize the NLI CrossEncoder.
    Loads the model once and caches it at the module level.
    """
    global _model, _label_index
    if _model is None:
        logger.info(f"Initializing NLI CrossEncoder: {settings.nli_model_name}")
        try:
            from sentence_transformers import CrossEncoder
            from torch import nn
            # Identity activation: predict() returns raw logits and we softmax them ourselves,
            # so the probabilities do not depend on the library's default activation.
            model = CrossEncoder(
                settings.nli_model_name,
                max_length=settings.nli_max_length,
                activation_fn=nn.Identity(),
            )
        except Exception as exc:
            logger.error(f"Failed to load NLI model {settings.nli_model_name}: {exc}")
            raise
        _label_index = _resolve_label_index(model)
        _model = model
    return _model


def _softmax(row: Sequence[float]) -> List[float]:
    peak = max(row)
    exps = [math.exp(x - peak) for x in row]
    total = sum(exps)
    return [e / total for e in exps]


def classify_pairs(pairs: Sequence[Tuple[str, str]]) -> List[NLIScores]:
    """
    Score each (premise, hypothesis) pair with the NLI model.
    CPU-bound and synchronous: call it via asyncio.to_thread from async code.

    Raises:
        RuntimeError: the model returned a different number of rows than pairs given.
        Exception: model load or inference failure (callers degrade, they never guess).
    """
    if not pairs:
        return []

    model = get_nli_model()
    raw = model.predict(list(pairs), convert_to_numpy=True, show_progress_bar=False)
    rows = [[float(x) for x in row] for row in raw]
    if len(rows) != len(pairs):
        raise RuntimeError(f"NLI model returned {len(rows)} rows for {len(pairs)} pairs")

    idx = _label_index
    results = []
    for row in rows:
        probs = _softmax(row)
        results.append(NLIScores(
            entailment=probs[idx["entailment"]],
            contradiction=probs[idx["contradiction"]],
            neutral=probs[idx["neutral"]],
        ))
    return results
