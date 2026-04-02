"""k-mer TF-IDF embedder."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import SequenceEmbedder


class KmerEmbedder(SequenceEmbedder):
    """Character k-gram TF-IDF over protein sequences.

    Returns a sparse matrix — downstream reducers handle sparse input.
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self._vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
        )

    @property
    def name(self) -> str:
        return f"kmer_k{self.k}"

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        # Returns a scipy sparse matrix; annotated as np.ndarray to satisfy
        # the base class contract. Pipeline and reducers handle both forms.
        return self._vectorizer.fit_transform(sequences)
