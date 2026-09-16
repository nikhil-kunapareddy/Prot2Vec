"""Metrics that score an embedding on protein family structure.

Two questions are being asked, and they are not the same question:

``trustworthiness``
    Did the *dimensionality reduction* lie? It compares neighbourhoods in the
    full-dimensional embedding against neighbourhoods in the 2-D projection, so
    it grades the reducer, not the embedder.

``knn_accuracy``
    Are protein families *linearly retrievable* from the representation? A
    cross-validated k-nearest-neighbour classifier is a direct proxy for "would
    a homology search over these vectors return the right family?".

``silhouette``
    Is the family structure *geometric*? Compactness relative to separation,
    computed against the true labels, with no classifier in the loop.

``precision_at_k``
    The question a homology search actually asks: given one protein, are its
    nearest neighbours in the same family? This is the metric that translates
    most directly into "could I use these vectors to annotate an unknown
    sequence?".

``adjusted_rand`` / ``normalized_mutual_info``
    Could the families be recovered *without* labels? k-means is run at the
    known family count and its partition compared to the truth, which is the
    relevant question when you are exploring sequences that have no
    annotation yet.

All are reported for every (embedder, reducer) pair so the effects can be told
apart: a representation can score well on trustworthiness simply because its
structure is trivial, and poorly on kNN because that structure is not
family-related. Except for ``trustworthiness`` and the 2-D kNN score, metrics
are computed on the full-dimensional embedding — that is the representation
under test, and reducing first would grade the reducer instead.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness as _trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from .._matrix import EmbeddingMatrix, as_dense

logger = logging.getLogger(__name__)

#: Minimum members a family needs before it can appear in a stratified split.
MIN_MEMBERS_PER_FAMILY = 2


def compute_trustworthiness(
    X_high: EmbeddingMatrix,
    X_low: EmbeddingMatrix,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> float:
    """Measure how well local neighbourhoods survive dimensionality reduction.

    Parameters
    ----------
    X_high
        Full-dimensional embedding, shape ``(n_samples, n_features)``.
    X_low
        Reduced embedding of the same samples, shape ``(n_samples, n_components)``.
    n_neighbors
        Neighbourhood size. Must be less than ``n_samples / 2``; scikit-learn
        rejects larger values.
    metric
        Distance metric used in both spaces.

    Returns
    -------
    float
        Value in ``(0, 1]``; ``1.0`` means every k-neighbourhood is preserved
        exactly.

    Raises
    ------
    ValueError
        If the two arrays describe different numbers of samples, or if
        ``n_neighbors`` is too large for the sample count.
    """
    X_high = as_dense(X_high)
    X_low = as_dense(X_low)

    if X_high.shape[0] != X_low.shape[0]:
        raise ValueError(
            "X_high and X_low must describe the same samples, got "
            f"{X_high.shape[0]} and {X_low.shape[0]} rows. This usually means an "
            "embedder returned fewer rows than it was given."
        )

    n_samples = X_high.shape[0]
    max_neighbors = (n_samples - 1) // 2
    if max_neighbors < 1:
        raise ValueError(f"Need at least 3 samples to compute trustworthiness, got {n_samples}.")
    if n_neighbors > max_neighbors:
        logger.warning(
            "n_neighbors=%d is too large for %d samples; clamping to %d.",
            n_neighbors,
            n_samples,
            max_neighbors,
        )
        n_neighbors = max_neighbors

    return float(_trustworthiness(X_high, X_low, n_neighbors=n_neighbors, metric=metric))


def knn_cv_accuracy(
    X: EmbeddingMatrix,
    y: list[str],
    n_neighbors: int = 5,
    n_splits: int = 5,
    metric: str = "cosine",
    random_state: int = 0,
) -> tuple[float, float]:
    """Cross-validated k-NN accuracy for recovering family labels from ``X``.

    Fold count and neighbourhood size are both clamped to what the data can
    actually support: Pfam seed alignments are small and unbalanced, and a
    rarely-populated family would otherwise make ``StratifiedKFold`` raise.

    Parameters
    ----------
    X
        Embedding to classify, shape ``(n_samples, n_features)``.
    y
        Family label per sample.
    n_neighbors
        Neighbours per vote.
    n_splits
        Requested number of stratified folds.
    metric
        Distance metric for the k-NN classifier.
    random_state
        Seed for the fold shuffle, so repeated runs agree.

    Returns
    -------
    tuple of float
        ``(mean_accuracy, std_accuracy)`` across folds.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` disagree on sample count, or if fewer than two
        families have enough members to be classified.
    """
    X = as_dense(X)
    if X.shape[0] != len(y):
        raise ValueError(f"X has {X.shape[0]} rows but {len(y)} labels were given.")

    counts = Counter(y)
    usable = {fam: n for fam, n in counts.items() if n >= MIN_MEMBERS_PER_FAMILY}
    if len(usable) < 2:
        raise ValueError(
            "k-NN accuracy needs at least two families with "
            f"{MIN_MEMBERS_PER_FAMILY}+ sequences each; got {dict(counts)}. "
            "Lower `min_seq_length` or pick families with larger seed alignments."
        )

    smallest = min(counts.values())
    effective_splits = min(n_splits, smallest)
    if effective_splits < n_splits:
        logger.warning(
            "Smallest family has %d sequences; reducing cross-validation from %d to %d folds.",
            smallest,
            n_splits,
            effective_splits,
        )
    # Every fold holds out 1/k of the data, so the training set can be smaller
    # than the requested neighbourhood.
    max_neighbors = int(len(y) * (effective_splits - 1) / effective_splits)
    effective_neighbors = max(1, min(n_neighbors, max_neighbors))
    if effective_neighbors < n_neighbors:
        logger.warning(
            "Reducing k from %d to %d to fit the training folds.", n_neighbors, effective_neighbors
        )

    knn = KNeighborsClassifier(n_neighbors=effective_neighbors, metric=metric)
    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(knn, X, y, cv=cv)
    return float(scores.mean()), float(scores.std())


def compute_silhouette(
    X: EmbeddingMatrix,
    y: list[str],
    metric: str = "cosine",
) -> float:
    """Mean silhouette coefficient of the true family labels in ``X``.

    Unlike the k-NN score this needs no classifier and no cross-validation, so
    it is unaffected by fold size — useful on the small, unbalanced seed
    alignments where a 5-fold split is already marginal.

    Parameters
    ----------
    X
        Embedding, shape ``(n_samples, n_features)``. Sparse input is densified.
    y
        Family label per sample.
    metric
        Distance metric.

    Returns
    -------
    float
        Value in ``[-1, 1]``. Above 0 means families are on average closer to
        themselves than to each other; near 0 means they overlap; negative
        means sequences sit closer to another family than their own.

    Raises
    ------
    ValueError
        If there are fewer than two families, or fewer samples than families
        plus one.
    """
    X = as_dense(X)
    n_labels = len(set(y))
    if n_labels < 2:
        raise ValueError(f"Silhouette needs at least two families, got {n_labels}.")
    if not 2 <= n_labels <= X.shape[0] - 1:
        raise ValueError(
            f"Silhouette needs 2 <= n_families <= n_samples - 1, got "
            f"{n_labels} families and {X.shape[0]} samples."
        )
    return float(silhouette_score(X, y, metric=metric))


def retrieval_precision_at_k(
    X: EmbeddingMatrix,
    y: list[str],
    k: int = 5,
    metric: str = "cosine",
) -> float:
    """Fraction of each sequence's ``k`` nearest neighbours in the same family.

    This is the embedding-space analogue of a BLAST or HMMER hit list: take one
    protein, retrieve its closest matches, and ask how many are true family
    members. It is reported instead of raw accuracy because retrieval is what
    an embedding is usually *for* — clustering a new proteome, or transferring
    annotation from a labelled neighbour.

    Each query excludes itself from its own neighbour list.

    Parameters
    ----------
    X
        Embedding, shape ``(n_samples, n_features)``.
    y
        Family label per sample.
    k
        Neighbours to retrieve per query. Clamped to ``n_samples - 1``.
    metric
        Distance metric.

    Returns
    -------
    float
        Mean precision in ``[0, 1]``, averaged over all queries. Compare it
        against the majority-class fraction, which is what random retrieval
        would score.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` disagree on sample count, there are fewer than two
        samples, or there is only one family (every neighbour would match, so
        the score would be a trivial 1.0).
    """
    X = as_dense(X)
    if X.shape[0] != len(y):
        raise ValueError(f"X has {X.shape[0]} rows but {len(y)} labels were given.")

    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("Retrieval precision needs at least two sequences.")
    if len(set(y)) < 2:
        raise ValueError("Retrieval precision needs at least two families to be meaningful.")

    effective_k = min(k, n_samples - 1)
    if effective_k < k:
        logger.warning("Reducing k from %d to %d to fit %d samples.", k, effective_k, n_samples)

    labels = np.asarray(y)
    # Ask for k+1 so the query itself can be discarded from its own results.
    finder = NearestNeighbors(n_neighbors=effective_k + 1, metric=metric).fit(X)
    neighbours = finder.kneighbors(X, return_distance=False)

    hits = 0.0
    for query, row in enumerate(neighbours):
        # Duplicate points mean the query is not guaranteed to rank first.
        others = [idx for idx in row if idx != query][:effective_k]
        hits += float(np.mean(labels[others] == labels[query]))

    return hits / n_samples


def clustering_agreement(
    X: EmbeddingMatrix,
    y: list[str],
    random_state: int = 0,
) -> tuple[float, float]:
    """Score how well unsupervised k-means recovers the known families.

    k-means is run at the true family count, so this isolates "is the structure
    there?" from "can a supervised model find it?". A representation can be
    perfectly classifiable yet cluster badly when families differ along a
    direction that is not the dominant axis of variance.

    Parameters
    ----------
    X
        Embedding, shape ``(n_samples, n_features)``.
    y
        Family label per sample, used only for the comparison and for setting
        the number of clusters.
    random_state
        Seed for k-means initialisation, so runs are reproducible.

    Returns
    -------
    tuple of float
        ``(adjusted_rand_index, normalized_mutual_info)``. ARI is
        chance-corrected and sits near 0 for a random partition; NMI is in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If there are fewer than two families, or fewer samples than families.
    """
    X = as_dense(X)
    families = sorted(set(y))
    if len(families) < 2:
        raise ValueError(f"Clustering agreement needs at least two families, got {len(families)}.")
    if X.shape[0] < len(families):
        raise ValueError(f"Cannot form {len(families)} clusters from {X.shape[0]} samples.")

    kmeans = KMeans(n_clusters=len(families), n_init=10, random_state=random_state)
    predicted = kmeans.fit_predict(X)
    return (
        float(adjusted_rand_score(y, predicted)),
        float(normalized_mutual_info_score(y, predicted)),
    )


def evaluate(
    X_high: EmbeddingMatrix,
    X_low: EmbeddingMatrix,
    y: list[str],
    n_neighbors_trust: int = 10,
    n_neighbors_knn: int = 5,
    precision_at_k: int = 5,
    metric: str = "cosine",
    knn_space: Literal["low", "high", "both"] = "both",
    random_state: int = 0,
) -> dict[str, float]:
    """Score one (embedder, reducer) pair and return a flat metrics dict.

    Metrics that fail on a given dataset are omitted rather than fatal: a
    benchmark of five families should still report everything it can when one
    metric's preconditions are not met.

    Parameters
    ----------
    X_high
        Embedding as produced by the embedder, before reduction.
    X_low
        The same samples after dimensionality reduction.
    y
        Family label per sample.
    n_neighbors_trust
        Neighbourhood size for trustworthiness.
    n_neighbors_knn
        Neighbours per vote for the k-NN classifier.
    precision_at_k
        Neighbours retrieved per query for ``precision_at_k``.
    metric
        Distance metric shared by the k-NN, silhouette and retrieval metrics.
    knn_space
        Where to run the k-NN evaluation. ``"low"`` scores only the 2-D
        projection; ``"high"`` only the raw embedding; ``"both"`` (default)
        reports each, which is the only way to tell an embedding failure apart
        from a projection failure.
    random_state
        Seed for cross-validation folds and k-means initialisation.

    Returns
    -------
    dict of str to float
        ``trustworthiness``, plus the k-NN scores selected by ``knn_space``,
        plus ``silhouette``, ``precision_at_k``, ``adjusted_rand`` and
        ``normalized_mutual_info`` computed on the full-dimensional embedding,
        and ``silhouette_2d`` on the projection.
    """
    # Trustworthiness is label-free, so it is the one metric that always
    # applies. Everything else depends on the family labels and is skipped with
    # a warning when the dataset cannot support it.
    metrics: dict[str, float] = {
        "trustworthiness": compute_trustworthiness(X_high, X_low, n_neighbors=n_neighbors_trust)
    }

    def knn(X: np.ndarray) -> tuple[float, float]:
        return knn_cv_accuracy(
            X, y, n_neighbors=n_neighbors_knn, metric=metric, random_state=random_state
        )

    optional: list[tuple[tuple[str, ...], Callable[[], Any]]] = []
    if knn_space in ("low", "both"):
        optional.append((("knn_accuracy_mean", "knn_accuracy_std"), lambda: knn(X_low)))
    if knn_space in ("high", "both"):
        optional.append(
            (("knn_accuracy_highdim_mean", "knn_accuracy_highdim_std"), lambda: knn(X_high))
        )
    optional += [
        (("silhouette",), lambda: compute_silhouette(X_high, y, metric=metric)),
        (("silhouette_2d",), lambda: compute_silhouette(X_low, y, metric=metric)),
        (
            ("precision_at_k",),
            lambda: retrieval_precision_at_k(X_high, y, k=precision_at_k, metric=metric),
        ),
        (
            ("adjusted_rand", "normalized_mutual_info"),
            lambda: clustering_agreement(X_high, y, random_state=random_state),
        ),
    ]

    for keys, compute in optional:
        try:
            value = compute()
        except ValueError as exc:
            logger.warning("Skipping %s: %s", "/".join(keys), exc)
            continue
        values = value if isinstance(value, tuple) else (value,)
        metrics.update(dict(zip(keys, (float(v) for v in values), strict=True)))

    return metrics
