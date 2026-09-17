"""Dipeptide composition: the classic order-aware descriptor."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from ..data.alphabets import PROTEIN, Alphabet, get_alphabet
from .base import SequenceEmbedder


class DipeptideEmbedder(SequenceEmbedder):
    """Normalised frequencies of all adjacent token pairs.

    Dipeptide composition (DPC) is one of the most heavily used descriptors in
    sequence-based prediction, and it sits in a genuinely useful place between
    the two representations Prot2Vec already has. Amino acid composition
    discards order entirely; k-mer TF-IDF keeps order but learns its vocabulary
    from the corpus, so the same column means different things in two different
    runs. DPC keeps first-order order information in a **fixed** layout: column
    ``i * |A| + j`` is always the frequency of token ``i`` followed by token
    ``j``, so vectors are comparable across datasets and individual features
    can be interpreted and plotted.

    Dimensionality is ``len(alphabet) ** 2`` — 400 for protein, 16 for DNA.

    Parameters
    ----------
    alphabet
        Token set. Determines both the dimensionality and the column order.

    Notes
    -----
    Counts are normalised by the number of pairs (``len(sequence) - 1``) rather
    than by sequence length, so rows sum to 1 for any sequence with at least
    two tokens.
    """

    def __init__(self, alphabet: str | Alphabet = PROTEIN) -> None:
        self.alphabet = get_alphabet(alphabet).name
        self._spec = get_alphabet(alphabet)
        self._index = self._spec.index

    @property
    def name(self) -> str:
        """Identifier such as ``dipeptide`` or ``dipeptide_dna``."""
        suffix = "" if self.alphabet == PROTEIN.name else f"_{self.alphabet}"
        return f"dipeptide{suffix}"

    @property
    def embedding_dim(self) -> int:
        """Width of the output: one column per ordered token pair."""
        return len(self._spec) ** 2

    def pair_names(self) -> list[str]:
        """Column labels in output order, e.g. ``["AA", "AC", ...]``.

        Useful for reading a loading vector back as chemistry rather than as an
        anonymous index.
        """
        return [f"{a}{b}" for a in self._spec.tokens for b in self._spec.tokens]

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        """Compute dipeptide frequencies for every sequence.

        Parameters
        ----------
        sequences
            Sequences as one-letter token strings. Tokens outside the alphabet
            are skipped, and so are the pairs that contain them.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(sequences), len(alphabet) ** 2)``,
            ``float32``. A sequence with fewer than two in-alphabet tokens
            yields a zero row.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")

        size = len(self._spec)
        vectors = np.zeros((len(sequences), size * size), dtype=np.float32)

        for row, sequence in enumerate(sequences):
            # Drop out-of-alphabet tokens first so that a stray character joins
            # its neighbours into a pair instead of breaking the chain.
            filtered = [c for c in sequence.upper() if c in self._index]
            if len(filtered) < 2:
                continue
            counts = np.zeros(size * size, dtype=np.float64)
            for first, second in pairwise(filtered):
                counts[self._index[first] * size + self._index[second]] += 1.0
            vectors[row] = counts / counts.sum()

        self._validate_rows(vectors, sequences)
        return vectors
