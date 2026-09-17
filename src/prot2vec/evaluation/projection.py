"""How much of the picture should you believe?

These metrics compare neighbourhoods in the full-dimensional embedding against
neighbourhoods in the projection. They grade the *reducer*, not the
representation, and they are the antidote to the single most common error in
embedding papers: reading cluster structure off a t-SNE plot without checking
whether the projection put it there.

Trustworthiness and continuity are duals and answer different questions.
Trustworthiness asks whether points that *look* close really are — it catches
invented structure. Continuity asks whether points that really are close still
look close — it catches destroyed structure. A projection can score well on one
and badly on the other, and which failure you care about depends on whether you
are about to believe a cluster or dismiss one.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness as _sklearn_trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .._matrix import EmbeddingMatrix, as_dense

logger = logging.getLogger(__name__)


def _max_neighbors(n_samples: int) -> int:
    """Largest neighbourhood scikit-learn's trustworthiness will accept."""
    return (n_samples - 1) // 2


def _check_paired(X_high: EmbeddingMatrix, X_low: EmbeddingMatrix) -> tuple[np.ndarray, np.ndarray]:
    """Densify and verify two views describe the same samples.

    Raises
    ------
    ValueError
        If the row counts differ, which almost always means an embedder
        returned fewer rows than it was given.
    """
    high = as_dense(X_high)
    low = as_dense(X_low)
    if high.shape[0] != low.shape[0]:
        raise ValueError(
            "X_high and X_low must describe the same samples, got "
            f"{high.shape[0]} and {low.shape[0]} rows. This usually means an "
            "embedder returned fewer rows than it was given."
        )
    if high.shape[0] < 3:
        raise ValueError(f"Projection metrics need at least 3 samples, got {high.shape[0]}.")
    return high, low


def _clamped_neighbors(requested: int, n_samples: int) -> int:
    """Clamp ``requested`` to a neighbourhood the sample count supports."""
    ceiling = _max_neighbors(n_samples)
    if requested > ceiling:
        logger.warning(
            "n_neighbors=%d is too large for %d samples; clamping to %d.",
            requested,
            n_samples,
            ceiling,
        )
        return max(1, ceiling)
    return requested


def _neighbor_sets(X: np.ndarray, k: int, metric: str) -> list[set[int]]:
    """Return the ``k`` nearest neighbours of every point, excluding itself."""
    finder = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), metric=metric).fit(X)
    found = finder.kneighbors(X, return_distance=False)
    # Query for k+1 and drop self: duplicate points mean the query is not
    # guaranteed to rank first, so filter by index rather than slicing it off.
    sets: list[set[int]] = []
    for i, row in enumerate(found):
        others = [int(index) for index in row if index != i]
        sets.append(set(others[:k]))
    return sets


def compute_trustworthiness(
    X_high: EmbeddingMatrix,
    X_low: EmbeddingMatrix,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> float:
    """Measure how well local neighbourhoods survive dimensionality reduction.

    Penalises points that appear in a projected neighbourhood without being
    genuine high-dimensional neighbours — that is, structure the reducer
    invented.

    Parameters
    ----------
    X_high
        Full-dimensional embedding, shape ``(n_samples, n_features)``.
    X_low
        Reduced embedding of the same samples.
    n_neighbors
        Neighbourhood size. Clamped to ``(n_samples - 1) // 2``.
    metric
        Distance metric used in both spaces.

    Returns
    -------
    float
        Value in ``(0, 1]``; ``1.0`` means every neighbourhood is preserved.

    Raises
    ------
    ValueError
        If the arrays disagree on sample count, or there are fewer than 3
        samples.
    """
    high, low = _check_paired(X_high, X_low)
    k = _clamped_neighbors(n_neighbors, high.shape[0])
    return float(_sklearn_trustworthiness(high, low, n_neighbors=k, metric=metric))


def compute_continuity(
    X_high: EmbeddingMatrix,
    X_low: EmbeddingMatrix,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> float:
    """Measure how much genuine neighbourhood structure the projection lost.

    The dual of :func:`compute_trustworthiness`: it penalises true
    high-dimensional neighbours that were pushed apart in the projection. It is
    computed by swapping the two spaces, which is exactly the definition.

    Parameters
    ----------
    X_high
        Full-dimensional embedding.
    X_low
        Reduced embedding of the same samples.
    n_neighbors
        Neighbourhood size.
    metric
        Distance metric.

    Returns
    -------
    float
        Value in ``(0, 1]``; ``1.0`` means no true neighbour was separated.
    """
    high, low = _check_paired(X_high, X_low)
    k = _clamped_neighbors(n_neighbors, high.shape[0])
    return float(_sklearn_trustworthiness(low, high, n_neighbors=k, metric=metric))


def neighborhood_preservation(
    X_high: EmbeddingMatrix,
    X_low: EmbeddingMatrix,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> float:
    """Mean fraction of each point's ``k`` nearest neighbours that the projection keeps.

    The most directly interpretable projection metric: 0.7 means that, on
    average, 7 of every 10 true nearest neighbours are still among the 10
    nearest after reduction.

    Parameters
    ----------
    X_high
        Full-dimensional embedding.
    X_low
        Reduced embedding of the same samples.
    n_neighbors
        Neighbourhood size.
    metric
        Distance metric.

    Returns
    -------
    float
        Value in ``[0, 1]``.
    """
    high, low = _check_paired(X_high, X_low)
    n_samples = high.shape[0]
    k = max(1, min(n_neighbors, n_samples - 1))

    high_sets = _neighbor_sets(high, k, metric)
    low_sets = _neighbor_sets(low, k, metric)
    overlaps = [len(a & b) / k for a, b in zip(high_sets, low_sets, strict=True)]
    return float(np.mean(overlaps))


def local_continuity_meta_criterion(
    X_high: EmbeddingMatrix,
    X_low: EmbeddingMatrix,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> float:
    """Chance-corrected neighbourhood overlap (LCMC).

    :func:`neighborhood_preservation` with the overlap you would get from a
    random projection subtracted off. With ``k`` neighbours among ``n`` points,
    random agreement is ``k / (n - 1)`` — which is not negligible on the small
    datasets typical of seed alignments, and is why a raw overlap of 0.2 can
    look meaningful when it is not.

    Parameters
    ----------
    X_high
        Full-dimensional embedding.
    X_low
        Reduced embedding of the same samples.
    n_neighbors
        Neighbourhood size.
    metric
        Distance metric.

    Returns
    -------
    float
        ``0.0`` means no better than chance; ``1 - k / (n - 1)`` is the maximum.
    """
    high, _ = _check_paired(X_high, X_low)
    n_samples = high.shape[0]
    k = max(1, min(n_neighbors, n_samples - 1))
    overlap = neighborhood_preservation(X_high, X_low, n_neighbors=k, metric=metric)
    return float(overlap - k / (n_samples - 1))


def distance_correlation(
    X_high: EmbeddingMatrix,
    X_low: EmbeddingMatrix,
    metric: str = "euclidean",
) -> float:
    """Spearman correlation between all pairwise distances, high versus low.

    The Shepard-diagram statistic, and the only metric here that is global
    rather than local. A projection can preserve every neighbourhood perfectly
    and still scramble the distances *between* clusters — t-SNE and UMAP
    routinely do — in which case this drops while trustworthiness stays high.
    Check it before reading anything into how far apart two families appear.

    Parameters
    ----------
    X_high
        Full-dimensional embedding.
    X_low
        Reduced embedding of the same samples.
    metric
        Distance metric for the high-dimensional space; the projection always
        uses Euclidean, since that is how a figure is read.

    Returns
    -------
    float
        Spearman rho in ``[-1, 1]``; near 1 means the projection preserves the
        distance ordering globally.
    """
    high, low = _check_paired(X_high, X_low)
    upper = np.triu_indices(high.shape[0], k=1)
    high_distances = pairwise_distances(high, metric=metric)[upper]
    low_distances = pairwise_distances(low, metric="euclidean")[upper]
    if high_distances.std() == 0 or low_distances.std() == 0:
        return 0.0
    rho = spearmanr(high_distances, low_distances).statistic
    return 0.0 if np.isnan(rho) else float(rho)
