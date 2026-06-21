import logging
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)

# Lazy loaded module-level SentenceTransformer instance
_model = None

def get_embedding_model():
    """
    Retrieve or initialize the SentenceTransformer model.
    Loads the model once and caches it at the module level.
    """
    global _model
    if _model is None:
        logger.info(f"Initializing SentenceTransformer model: {settings.embedding_model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(settings.embedding_model_name)
        except Exception as exc:
            logger.error(f"Failed to load SentenceTransformer model {settings.embedding_model_name}: {exc}")
            raise
    return _model

def generate_embedding(text: str) -> List[float]:
    """
    Generate a normalized embedding vector for the given text.
    Returns a zero vector of the configured dimension if the text is empty.
    """
    if not text or not text.strip():
        logger.warning("Empty text passed to generate_embedding. Returning zero vector.")
        return [0.0] * settings.embedding_dimension

    try:
        model = get_embedding_model()
        # normalize_embeddings=True ensures cosine similarity maps directly to a dot product
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as exc:
        logger.error(f"Embedding generation failed: {exc}")
        # Return fallback zero vector to avoid crashing downstream pipeline
        return [0.0] * settings.embedding_dimension
