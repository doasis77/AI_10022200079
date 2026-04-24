# src/retrieval/embedder.py
# Author: [Your Name] | Index: [Your Index Number]
# Generates dense vector embeddings using sentence-transformers (all-MiniLM-L6-v2)

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level singleton — loaded once per process
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings.
    Returns a float32 numpy array of shape (N, 384).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2-normalize for cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, 384)."""
    return embed_texts([query])
