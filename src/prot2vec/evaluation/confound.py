"""Is the embedding measuring what you think it is?

Every other metric here asks how *well* the groups separate. These ask
*why* — and they exist because the commonest way to get a good-looking
embedding result is to accidentally measure something trivial.

Sequence length is the usual culprit. Protein families differ systematically in
length, mean-pooled language model embeddings drift with length, and k-mer
vectors of long sequences are denser than those of short ones. So a
representation can score 0.95 on family separability while encoding little more
than "how long is this sequence". :func:`length_only_knn_accuracy` settles it
directly by classifying on length alone: if that already reaches 0.9, a
0.93 from a 650M-parameter model is not the result it appears to be.

The correlation metrics are diagnostic rather than good-or-bad. A high
``composition_distance_rho`` on a composition embedder is a tautology; the same
value on ESM-2 means the language model is mostly recovering amino acid
frequencies.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from .._matrix import EmbeddingMatrix, as_dense
from ..data.alphabets import PROTEIN, Alphabet, get_alphabet
from .classification import knn_cv_accuracy

logger = logging.getLogger(__name__)


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Flatten the strict upper triangle of a square matrix."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman rho, returning 0.0 when either input is constant."""
    if left.std() == 0 or right.std() == 0:
        return 0.0
    rho = spearmanr(left, right).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def length_distance_correlation(
    X: EmbeddingMatrix,
    sequences: list[str],
    metric: str = "cosine",
) -> float:
    """Correlation between embedding distance and difference in sequence length.

    Parameters
    ----------
    X
        Embedding, shape ``(n_samples, n_features)``.
    sequences
        The sequences the embedding was computed from, in the same order.
    metric
        Distance metric for the embedding space.

    Returns
    -------
    float
        Spearman rho in ``[-1, 1]``. Near 0 is what you want. Above roughly 0.5
        means a large part of what the embedding encodes is length, and any
        group separation it achieves should be checked against
        :func:`length_only_knn_accuracy`.

    Raises
    ------
    ValueError
        If ``X`` and ``sequences`` disagree on count, or there are fewer than
        three sequences.
    """
    dense = as_dense(X)
    if dense.shape[0] != len(sequences):
        raise ValueError(f"X has {dense.shape[0]} rows but {len(sequences)} sequences were given.")
    if dense.shape[0] < 3:
        raise ValueError("Confound metrics need at least three sequences.")

    lengths = np.array([[len(s)] for s in sequences], dtype=np.float64)
    embedding_distances = _upper_triangle(pairwise_distances(dense, metric=metric))
    length_gaps = _upper_triangle(pairwise_distances(lengths, metric="euclidean"))
    return _safe_spearman(embedding_distances, length_gaps)


def composition_distance_correlation(
    X: EmbeddingMatrix,
    sequences: list[str],
    metric: str = "cosine",
    alphabet: str | Alphabet = PROTEIN,
) -> float:
    """Correlation between embedding distance and residue-composition distance.

    Quantifies how much of a representation is recoverable from token
    frequencies alone. For a pretrained model this is the headline question:
    does it encode anything beyond composition?

    Parameters
    ----------
    X
        Embedding.
    sequences
        The sequences the embedding was computed from, in the same order.
    metric
        Distance metric, applied in both the embedding and composition spaces.
    alphabet
        Token set used to build the composition vectors.

    Returns
    -------
    float
        Spearman rho in ``[-1, 1]``. Near 1 means the embedding is
        substantially a re-encoding of composition.

    Raises
    ------
    ValueError
        If ``X`` and ``sequences`` disagree on count, or there are fewer than
        three sequences.
    """
    dense = as_dense(X)
    if dense.shape[0] != len(sequences):
        raise ValueError(f"X has {dense.shape[0]} rows but {len(sequences)} sequences were given.")
    if dense.shape[0] < 3:
        raise ValueError("Confound metrics need at least three sequences.")

    spec = get_alphabet(alphabet)
    composition = np.zeros((len(sequences), len(spec)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        counts = Counter(sequence.upper())
        total = sum(counts[token] for token in spec.tokens)
        if total:
            for column, token in enumerate(spec.tokens):
                composition[row, column] = counts[token] / total

    embedding_distances = _upper_triangle(pairwise_distances(dense, metric=metric))
    composition_distances = _upper_triangle(pairwise_distances(composition, metric=metric))
    return _safe_spearman(embedding_distances, composition_distances)


def length_only_knn_accuracy(
    sequences: list[str],
    y: list[str],
    n_neighbors: int = 5,
    n_splits: int = 5,
    random_state: int = 0,
) -> float:
    """Cross-validated k-NN accuracy using sequence length as the only feature.

    The baseline every reported score should be read against. It uses no
    representation at all — one number per sequence — so anything a real
    embedding achieves needs to beat it by a margin worth the compute.

    Parameters
    ----------
    sequences
        The sequences, in the same order as ``y``.
    y
        Label per sequence.
    n_neighbors
        Neighbours per vote.
    n_splits
        Requested number of stratified folds.
    random_state
        Seed for the fold shuffle.

    Returns
    -------
    float
        Mean accuracy in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``sequences`` and ``y`` disagree on count, or the labels cannot
        support stratified cross-validation.
    """
    if len(sequences) != len(y):
        raise ValueError(f"{len(sequences)} sequences but {len(y)} labels were given.")
    lengths = np.array([[float(len(s))] for s in sequences])
    mean, _ = knn_cv_accuracy(
        lengths,
        y,
        n_neighbors=n_neighbors,
        n_splits=n_splits,
        metric="euclidean",  # cosine is meaningless for a single feature
        random_state=random_state,
    )
    return mean
