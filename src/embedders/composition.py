"""Amino acid composition embedder (20-dimensional frequency vector)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from .base import SequenceEmbedder

_AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


class CompositionEmbedder(SequenceEmbedder):
    """Represents each sequence as the relative frequency of the 20 standard AAs."""

    @property
    def name(self) -> str:
        return "composition"

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        vectors = []
        for seq in sequences:
            counts = Counter(seq)
            n = max(len(seq), 1)
            vectors.append([counts.get(aa, 0) / n for aa in _AA_LIST])
        return np.array(vectors, dtype=np.float32)
