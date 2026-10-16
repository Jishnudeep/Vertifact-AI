import logging
import threading

from app.agents.linguistic_agent import get_spacy_nlp
from app.services.bias_lookup import load_bias_ratings
from app.services.embedding import generate_embedding
from app.services.nli import get_nli_model

logger = logging.getLogger(__name__)

_warm_lock = threading.Lock()
_warmed = False


def warm_models() -> None:
    """Load every heavy asset once, per CLAUDE.md §7: never inside a request or its deadline."""
    # 1. Load AllSides bias ratings
    logger.info("Loading AllSides bias ratings...")
    load_bias_ratings()

    # 2. Preload spaCy NLP pipeline. Not wrapped: en_core_web_sm is a pinned dependency, so a
    # failure is a broken install, not a transient error. Booting anyway would degrade
    # Cross-Ref and Linguistic on every request; fail at boot instead. (The embedding and NLI
    # preloads below only warn: they may download from the HF hub, which can fail transiently.)
    logger.info("Preloading spaCy 'en_core_web_sm'...")
    get_spacy_nlp()

    # 3. Preload SentenceTransformer embedding model
    logger.info("Preloading embedding model...")
    try:
        generate_embedding("VeriFact AI pipeline warmup embedding.")
    except Exception as exc:
        logger.warning(f"Embedding model preload warning: {exc}")

    # 4. Preload NLI cross-encoder (source stance + cache equivalence)
    logger.info("Preloading NLI model...")
    try:
        get_nli_model()
    except Exception as exc:
        logger.warning(f"NLI model preload warning: {exc}")


def ensure_models_warm() -> None:
    """
    warm_models() at most once per process, for entry points with no startup hook (adk web).

    Only a bias-ratings or spaCy failure raises (and is retried on the next call); the embedding
    and NLI loads only warn, as at boot, so they count as warmed and may load lazily later.
    """
    global _warmed
    with _warm_lock:
        if _warmed:
            return
        warm_models()
        _warmed = True
