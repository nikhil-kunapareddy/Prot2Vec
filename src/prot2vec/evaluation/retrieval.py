"""Could you annotate an unknown sequence from its neighbours?

This is the question an embedding is usually built to answer, and it is not the
same question a classifier answers. A classifier is trained on your labels;
retrieval just ranks by distance, which is what you actually do when a new
sequence arrives and you look for something similar. These metrics are the
embedding-space analogues of reading a BLAST or HMMER hit list.

``precision_at_k`` is the top of the list. ``mean_average_precision`` grades
the whole ranking, so it distinguishes an embedding that puts three true
homologues at ranks 1-3 from one that puts them at 1, 40 and 80.
``same_class_auroc`` drops the ranking entirely and asks whether distance alone
separates same-family pairs from different-family pairs — the standard
formulation of remote homology detection, and the metric least sensitive to how
many sequences each family happens to have.

All of these build a full pairwise distance matrix, which is ``O(n^2)`` in both
time and memory. That is fine for the hundreds-to-thousands of sequences a
benchmark uses and will not scale to a whole proteome.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import pairwise_distances, roc_auc_score

from .._matrix import EmbeddingMatrix, as_dense

logger = logging.getLogger(__name__)


def _check_retrieval_inputs(X: EmbeddingMatrix, y: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Densify and validate inputs for a retrieval metric.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` disagree on sample count, there are fewer than two
        samples, or only one class is present — with one class every retrieved
        neighbour matches, so every score would be a meaningless 1.0.
    """
    dense = as_dense(X)
    if dense.shape[0] != len(y):
        raise ValueError(f"X has {dense.shape[0]} rows but {len(y)} labels were given.")
    if dense.shape[0] < 2:
        raise ValueError("Retrieval metrics need at least two sequences.")
    labels = np.asarray(y)
    if len(np.unique(labels)) < 2:
        raise ValueError("Retrieval metrics need at least two classes to be meaningful.")
    return dense, labels


def _ranked_neighbors(X: np.ndarray, metric: str) -> np.ndarray:
    """Return every other point's index, ordered nearest first, self excluded."""
    distances = pairwise_distances(X, metric=metric)
    np.fill_diagonal(distances, np.inf)  # never retrieve the query itself
    return np.argsort(distances, axis=1, kind="stable")


def retrieval_precision_at_k(
    X: EmbeddingMatrix,
    y: list[str],
    k: int = 5,
    metric: str = "cosine",
) -> float:
    """Fraction of each sequence's ``k`` nearest neighbours in the same class.

    Parameters
    ----------
    X
        Embedding, shape ``(n_samples, n_features)``.
    y
        Label per sample.
    k
        Neighbours to retrieve per query. Clamped to ``n_samples - 1``.
    metric
        Distance metric.

    Returns
    -------
    float
        Mean precision in ``[0, 1]``. Compare against the majority-class
        fraction, which is roughly what random retrieval scores.

    Raises
    ------
    ValueError
        See :func:`_check_retrieval_inputs`.
    """
    dense, labels = _check_retrieval_inputs(X, y)
    n_samples = dense.shape[0]
    effective_k = min(k, n_samples - 1)
    if effective_k < k:
        logger.warning("Reducing k from %d to %d to fit %d samples.", k, effective_k, n_samples)

    ranked = _ranked_neighbors(dense, metric)
    hits = [float(np.mean(labels[ranked[i, :effective_k]] == labels[i])) for i in range(n_samples)]
    return float(np.mean(hits))


def mean_average_precision(
    X: EmbeddingMatrix,
    y: list[str],
    metric: str = "cosine",
) -> float:
    """Mean average precision over the full ranking for every query.

    For each sequence, every other sequence is ranked by distance and average
    precision is computed over the positions of its true class members. Unlike
    ``precision@k`` this is sensitive to the entire list, so it rewards an
    embedding that keeps a family contiguous rather than merely getting the
    first few hits right.

    Parameters
    ----------
    X
        Embedding.
    y
        Label per sample.
    metric
        Distance metric.

    Returns
    -------
    float
        Value in ``(0, 1]``; 1.0 means every query ranks all of its own class
        above everything else.

    Raises
    ------
    ValueError
        See :func:`_check_retrieval_inputs`.
    """
    dense, labels = _check_retrieval_inputs(X, y)
    ranked = _ranked_neighbors(dense, metric)

    scores: list[float] = []
    for query in range(dense.shape[0]):
        relevant = labels[ranked[query]] == labels[query]
        total_relevant = int(relevant.sum())
        if total_relevant == 0:
            continue  # a singleton class has nothing to retrieve
        positions = np.flatnonzero(relevant) + 1
        precisions = np.arange(1, total_relevant + 1) / positions
        scores.append(float(precisions.mean()))

    if not scores:
        raise ValueError("No class has two or more members, so nothing can be retrieved.")
    return float(np.mean(scores))


def r_precision(
    X: EmbeddingMatrix,
    y: list[str],
    metric: str = "cosine",
) -> float:
    """Precision at ``R``, where ``R`` is each query's own class size.

    Self-calibrating: a query from a 40-sequence family is scored on its top 39
    hits and one from a 5-sequence family on its top 4. That removes the
    arbitrary choice of ``k`` and stops large families from dominating the
    average, which matters because Pfam seed alignments are very unevenly
    sized.

    Parameters
    ----------
    X
        Embedding.
    y
        Label per sample.
    metric
        Distance metric.

    Returns
    -------
    float
        Value in ``[0, 1]``.

    Raises
    ------
    ValueError
        See :func:`_check_retrieval_inputs`.
    """
    dense, labels = _check_retrieval_inputs(X, y)
    ranked = _ranked_neighbors(dense, metric)
    counts = {label: int((labels == label).sum()) for label in np.unique(labels)}

    scores: list[float] = []
    for query in range(dense.shape[0]):
        r = counts[labels[query]] - 1  # exclude the query itself
        if r < 1:
            continue
        retrieved = labels[ranked[query, :r]]
        scores.append(float(np.mean(retrieved == labels[query])))

    if not scores:
        raise ValueError("No class has two or more members, so nothing can be retrieved.")
    return float(np.mean(scores))


def same_class_auroc(
    X: EmbeddingMatrix,
    y: list[str],
    metric: str = "cosine",
) -> float:
    """AUROC for separating same-class pairs from different-class pairs by distance.

    Every pair of sequences is one observation: the label is "same family or
    not" and the score is the negated distance. This is the standard remote
    homology detection framing. Because it is computed over pairs rather than
    over queries, it is unaffected by class size imbalance, which makes it the
    fairest single number to compare embeddings on when families are unevenly
    populated.

    Parameters
    ----------
    X
        Embedding.
    y
        Label per sample.
    metric
        Distance metric.

    Returns
    -------
    float
        Value in ``[0, 1]``; 0.5 is chance, 1.0 means every same-family pair is
        closer than every different-family pair.

    Raises
    ------
    ValueError
        See :func:`_check_retrieval_inputs`.
    """
    dense, labels = _check_retrieval_inputs(X, y)
    upper = np.triu_indices(dense.shape[0], k=1)
    distances = pairwise_distances(dense, metric=metric)[upper]
    same = (labels[upper[0]] == labels[upper[1]]).astype(int)

    if same.min() == same.max():
        raise ValueError(
            "All sequence pairs share a label, or none do, so same-class AUROC is undefined."
        )
    # Closer should mean "more likely same family", hence the negation.
    return float(roc_auc_score(same, -distances))
