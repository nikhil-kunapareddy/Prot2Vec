"""Amino acid composition: the simplest useful protein representation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .base import SequenceEmbedder

#: The 20 standard amino acids, in the canonical alphabetical order that fixes
#: the meaning of every column in the output matrix.
AA_ALPHABET: str = "ACDEFGHIKLMNPQRSTVWY"


class CompositionEmbedder(SequenceEmbedder):
    """Represent each sequence as the relative frequency of the 20 standard residues.

    This is the classic composition baseline. It discards residue order
    entirely, so any family separation it achieves is attributable to
    compositional bias alone — hydrophobic membrane domains, for instance,
    separate from soluble enzymes without any model learning anything about
    structure. It is the honest floor against which a protein language model
    should be judged, and it is fast enough to run on every dataset.

    Column ``i`` of the output is the frequency of ``AA_ALPHABET[i]``, so rows
    sum to 1 for any sequence containing at least one standard residue.
    """

    @property
    def name(self) -> str:
        """Identifier: ``composition``."""
        return "composition"

    @property
    def embedding_dim(self) -> int:
        """Width of the output: one column per standard amino acid."""
        return len(AA_ALPHABET)

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        """Compute residue frequencies for every sequence.

        Parameters
        ----------
        sequences
            Protein sequences as one-letter amino acid strings. Non-standard
            characters are ignored rather than counted.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(sequences), 20)``, ``float32``. A sequence
            with no standard residues yields a zero row.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")

        vectors = np.zeros((len(sequences), len(AA_ALPHABET)), dtype=np.float32)
        for row, sequence in enumerate(sequences):
            counts = Counter(sequence.upper())
            # Normalise by standard residues only, so that stripped characters
            # cannot silently shrink every frequency in the row.
            total = sum(counts[aa] for aa in AA_ALPHABET)
            if total == 0:
                continue
            for col, aa in enumerate(AA_ALPHABET):
                vectors[row, col] = counts[aa] / total

        self._validate_rows(vectors, sequences)
        return vectors
