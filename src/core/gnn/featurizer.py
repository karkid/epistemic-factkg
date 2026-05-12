"""Text and modality featurizer for GNN node construction (ADR-014)."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from src.core.gnn.types import (
    MODALITY_TO_INT,
    PRAMANA_TO_INT,
    NUM_MODALITY,
    NUM_PRAMANA,
)

_EMBED_MODEL = "all-MiniLM-L6-v2"
_EMBED_DIM = 384


class Featurizer:
    """Encodes text to sentence embeddings and categorical fields to one-hot vectors.

    Loads the sentence-transformer model once and caches embeddings to a .pkl file
    keyed by text hash so re-runs skip re-embedding expensive text.
    """

    def __init__(self, cache_path: str | Path | None = None):
        self._model: SentenceTransformer | None = None
        self._cache: dict[str, list[float]] = {}
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self._cache = pickle.load(f)

    def _model_(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(_EMBED_MODEL)
        return self._model

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings → float tensor [N, 384].

        Hits the cache for already-seen texts; embeds misses in batch and updates cache.
        """
        results: list[list[float]] = []
        misses: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            h = self._hash(t)
            if h in self._cache:
                results.append(self._cache[h])
            else:
                results.append([])
                misses.append((i, t))

        if misses:
            idxs, miss_texts = zip(*misses)
            embeddings = self._model_().encode(list(miss_texts), convert_to_numpy=True)
            for i, emb in zip(idxs, embeddings):
                h = self._hash(miss_texts[list(idxs).index(i)])
                self._cache[h] = emb.tolist()
                results[i] = emb.tolist()

        return torch.tensor(results, dtype=torch.float32)

    def encode_modality(self, modality: str | None) -> torch.Tensor:
        """One-hot encode evidence modality → float tensor [NUM_MODALITY]."""
        vec = torch.zeros(NUM_MODALITY, dtype=torch.float32)
        idx = MODALITY_TO_INT.get(modality or "", -1)
        if idx >= 0:
            vec[idx] = 1.0
        return vec

    def encode_pramana(self, pramana: str | None) -> torch.Tensor:
        """One-hot encode Pramana type → float tensor [NUM_PRAMANA]."""
        vec = torch.zeros(NUM_PRAMANA, dtype=torch.float32)
        idx = PRAMANA_TO_INT.get(pramana or "", -1)
        if idx >= 0:
            vec[idx] = 1.0
        return vec

    def save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)
