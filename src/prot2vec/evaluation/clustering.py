"""Is the structure there without the labels?

Everything in :mod:`~prot2vec.evaluation.classification` and
:mod:`~prot2vec.evaluation.retrieval` uses the labels to score the embedding.
These metrics mostly do not, which matters because the situation researchers
are usually in is the unlabelled one: a set of sequences, no annotation, and a
question about whether they fall into groups at all.

k-means is run at the true class count and its partition compared with the
truth. A representation can be perfectly classifiable and cluster badly, when
the class boundary is not aligned with the dominant axes of variance — so a
high k-NN score and a low ARI together are a specific, useful finding, not a
contradiction.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

from .._matrix import EmbeddingMatrix, as_dense

logger = logging.getLogger(__name__)


def _check_grouped(X: EmbeddingMatrix, y: list[str], what: str) -> tuple[np.ndarray, list[str]]:
    """Densify and validate that ``y`` describes at least two usable groups.

    Raises
    ------
    ValueError
        If sample counts disagree, or there are fewer than two groups, or there
        are not enough samples to form them.
    """
    dense = as_dense(X)
    if dense.shape[0] != len(y):
        raise ValueError(f"X has {dense.shape[0]} rows but {len(y)} labels were given.")
    groups = sorted(set(y))
    if len(groups) < 2:
        raise ValueError(f"{what} needs at least two groups, got {len(groups)}.")
    if dense.shape[0] <= len(groups):
        raise ValueError(
            f"{what} needs more samples than groups, got {dense.shape[0]} "
            f"samples and {len(groups)} groups."
        )
    return dense, groups


def compute_silhouette(
    X: EmbeddingMatrix,
    y: list[str],
    metric: str = "cosine",
) -> float:
    """Mean silhouette coefficient of the true labels in ``X``.

    Needs no classifier and no cross-validation, so it is unaffected by fold
    size — useful on the small, unbalanced datasets where a 5-fold split is
    already marginal.

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
        Value in ``[-1, 1]``. Above 0 means groups are on average closer to
        themselves than to each other; negative means samples sit closer to
        another group than their own.

    Raises
    ------
    ValueError
        See :func:`_check_grouped`.
    """
    dense, _ = _check_grouped(X, y, "Silhouette")
    return float(silhouette_score(dense, y, metric=metric))


def clustering_report(
    X: EmbeddingMatrix,
    y: list[str],
    random_state: int = 0,
) -> dict[str, float]:
    """Compare a k-means partition at the true group count against the labels.

    Parameters
    ----------
    X
        Embedding.
    y
        Label per sample, used for the comparison and to set the cluster count.
    random_state
        Seed for k-means initialisation.

    Returns
    -------
    dict of str to float
        ``adjusted_rand`` (chance-corrected, ~0 for a random partition),
        ``normalized_mutual_info``, ``homogeneity`` (do clusters contain a
        single group?), ``completeness`` (is each group kept in one cluster?),
        ``v_measure`` (their harmonic mean) and ``fowlkes_mallows``.

        Homogeneity and completeness are reported separately because they fail
        in opposite ways: a representation that splits one family across three
        tight clusters scores high on the first and low on the second, which is
        a very different problem from merging two families together.

    Raises
    ------
    ValueError
        See :func:`_check_grouped`.
    """
    dense, groups = _check_grouped(X, y, "Clustering agreement")
    kmeans = KMeans(n_clusters=len(groups), n_init=10, random_state=random_state)
    predicted = kmeans.fit_predict(dense)
    return {
        "adjusted_rand": float(adjusted_rand_score(y, predicted)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y, predicted)),
        "homogeneity": float(homogeneity_score(y, predicted)),
        "completeness": float(completeness_score(y, predicted)),
        "v_measure": float(v_measure_score(y, predicted)),
        "fowlkes_mallows": float(fowlkes_mallows_score(y, predicted)),
    }


def clustering_agreement(
    X: EmbeddingMatrix,
    y: list[str],
    random_state: int = 0,
) -> tuple[float, float]:
    """Return just ``(adjusted_rand, normalized_mutual_info)``.

    Kept for callers that want the two headline numbers;
    :func:`clustering_report` computes the same partition and reports
    everything derived from it.
    """
    report = clustering_report(X, y, random_state=random_state)
    return report["adjusted_rand"], report["normalized_mutual_info"]


def internal_cluster_indices(X: EmbeddingMatrix, y: list[str]) -> dict[str, float]:
    """Geometric quality of the labelled grouping, with no clustering step.

    Both indices describe how compact and well-separated the labelled groups
    are in Euclidean space, which is the geometry a 2-D figure actually shows.
    They are scale-free in opposite directions, so the pair is easier to read
    than either alone.

    Parameters
    ----------
    X
        Embedding.
    y
        Label per sample.

    Returns
    -------
    dict of str to float
        ``davies_bouldin`` (lower is better, 0 is perfect) and
        ``calinski_harabasz`` (higher is better, unbounded — comparable across
        methods on the same dataset but not across datasets).

    Raises
    ------
    ValueError
        See :func:`_check_grouped`.
    """
    dense, _ = _check_grouped(X, y, "Internal cluster indices")
    return {
        "davies_bouldin": float(davies_bouldin_score(dense, y)),
        "calinski_harabasz": float(calinski_harabasz_score(dense, y)),
    }
