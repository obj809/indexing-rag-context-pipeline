"""Embed text chunks via a local sentence-transformers model."""

import numpy as np
from sentence_transformers import SentenceTransformer


def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    # normalize_embeddings=True → unit-length vectors → dot product equals cosine similarity in retrieve.py
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
