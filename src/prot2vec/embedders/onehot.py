"""Positional one-hot encoding: the representation that keeps everything."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from ..data.alphabets import PROTEIN, Alphabet, get_alphabet
from .base import SequenceEmbedder


class OneHotEmbedder(SequenceEmbedder):
    """Encode each position independently, then flatten.

    This is the input format convolutional and recurrent sequence models are
    trained on, and it is the only representation here that throws nothing
    away: position *p* holding token *t* is its own feature. That makes it a
    useful upper reference — whatever a downstream method cannot extract from a
    one-hot encoding was genuinely not in the sequence.

    The catch is that it is not alignment-free. Feature ``p * |A| + t`` only
    means the same thing across two sequences if position *p* is comparable in
    both, which for unaligned proteins of different lengths it generally is
    not. Expect it to underperform composition-style descriptors on families
    that are not length-matched, and read a good score as evidence that the
    sequences are positionally consistent.

    Dimensionality is ``max_len * len(alphabet)``, so the result is sparse.

    Parameters
    ----------
    max_len
        Positions to encode. Shorter sequences are zero-padded, longer ones
        truncated according to ``truncate``.
    alphabet
        Token set; determines the per-position block width and order.
    truncate
        Which part of an over-long sequence to keep:

        ``"center"``
            The middle, which retains the domain core that families are
            usually defined on. The default.
        ``"start"``
            The N-terminal (or 5') region.
        ``"end"``
            The C-terminal (or 3') region.
    """

    def __init__(
        self,
        max_len: int = 256,
        alphabet: str | Alphabet = PROTEIN,
        truncate: str = "center",
    ) -> None:
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}.")
        if truncate not in ("center", "start", "end"):
            raise ValueError(
                f"Unknown truncate {truncate!r}. Choose from 'center', 'start', 'end'."
            )
        self.max_len = max_len
        self.alphabet = get_alphabet(alphabet).name
        self.truncate = truncate
        self._spec = get_alphabet(alphabet)
        self._index = self._spec.index

    @property
    def name(self) -> str:
        """Identifier such as ``onehot_L256``."""
        suffix = "" if self.alphabet == PROTEIN.name else f"_{self.alphabet}"
        return f"onehot{suffix}_L{self.max_len}"

    @property
    def embedding_dim(self) -> int:
        """Width of the flattened encoding."""
        return self.max_len * len(self._spec)

    def _crop(self, tokens: list[str]) -> list[str]:
        """Reduce ``tokens`` to at most ``max_len`` according to ``truncate``."""
        if len(tokens) <= self.max_len:
            return tokens
        if self.truncate == "start":
            return tokens[: self.max_len]
        if self.truncate == "end":
            return tokens[-self.max_len :]
        start = (len(tokens) - self.max_len) // 2
        return tokens[start : start + self.max_len]

    def fit_transform(self, sequences: list[str]) -> csr_matrix:
        """One-hot encode every sequence and flatten to a sparse matrix.

        Parameters
        ----------
        sequences
            Sequences as one-letter token strings. Out-of-alphabet tokens are
            dropped before padding, so they do not consume a position.

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix of shape ``(len(sequences), max_len * len(alphabet))``,
            ``float32``. Padding positions are all-zero blocks.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")

        width = len(self._spec)
        rows: list[int] = []
        columns: list[int] = []

        for row, sequence in enumerate(sequences):
            tokens = self._crop([c for c in sequence.upper() if c in self._index])
            for position, token in enumerate(tokens):
                rows.append(row)
                columns.append(position * width + self._index[token])

        result = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, columns)),
            shape=(len(sequences), self.embedding_dim),
        )
        self._validate_rows(result, sequences)
        return result
