"""Abstract base class for all sequence embedders."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SequenceEmbedder(ABC):
    """All embedders share a single interface: fit_transform(sequences) -> np.ndarray."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in filenames, logs, and result tables."""
        ...

    @abstractmethod
    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        """Embed a list of protein sequences. Returns array of shape (n, d).

        May return a scipy sparse matrix for high-dimensional embedders (e.g. k-mer).
        Reducers and the pipeline handle both dense and sparse inputs.
        """
        ...
