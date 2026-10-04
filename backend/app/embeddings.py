"""Embedding backends. Models are loaded once per process."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("docusense.embeddings")


class Embedder:
    name: str
    dim: int
    default_min_score: float  # below this, a hit is not considered relevant
    default_high_score: float  # at/above this, a top hit counts as strong

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    default_min_score = 0.30
    default_high_score = 0.55

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = f"sentence-transformers/{model_name}"
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts):
        v = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return np.asarray(v, dtype="float32")


class HashingEmbedder(Embedder):
    """Stateless lexical fallback (unigram+bigram hashing). Weaker than a neural model:
    it matches words, not meaning. Used only when sentence-transformers is unavailable."""

    name = "hashing-lexical"
    default_min_score = 0.10
    default_high_score = 0.30

    def __init__(self, n_features: int = 4096):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = n_features
        self._vec = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2",
                                      ngram_range=(1, 2), stop_words="english")

    def encode(self, texts):
        return self._vec.transform(list(texts)).toarray().astype("float32")


def get_embedder(backend: str, model: str) -> Embedder:
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(model)
        except Exception as e:  # missing package, offline, etc.
            if backend == "sentence-transformers":
                raise
            log.warning("sentence-transformers unavailable (%s); using lexical fallback embedder", type(e).__name__)
    return HashingEmbedder()
