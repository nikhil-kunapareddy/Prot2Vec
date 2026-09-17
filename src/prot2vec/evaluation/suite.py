"""Assemble the metric panel for one (embedder, reducer) pair.

Metrics are organised into five groups, because different research questions
need different subsets and reporting all of them for every run makes the table
unreadable:

``projection``
    Did the reducer preserve the structure, or create it?
``classification``
    Can the labels be recovered from the representation?
``retrieval``
    Would a nearest-neighbour lookup return true group members?
``clustering``
    Is the structure findable without labels at all?
``confound``
    Is the embedding measuring group identity, or just sequence length and
    composition?

Select groups with ``metric_groups``. Every metric degrades gracefully: when
its preconditions are not met — a singleton class, a dataset too small for the
requested neighbourhood, missing sequences for the confound checks — it is
omitted with a warning rather than failing the run. Only ``trustworthiness`` is
unconditional, because it is the one metric that needs no labels.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from .._matrix import EmbeddingMatrix
from ..data.alphabets import PROTEIN, Alphabet
from .classification import knn_classification_report, knn_cv_accuracy
from .clustering import (
    clustering_report,
    compute_silhouette,
    internal_cluster_indices,
)
from .confound import (
    composition_distance_correlation,
    length_distance_correlation,
    length_only_knn_accuracy,
)
from .projection import (
    compute_continuity,
    compute_trustworthiness,
    distance_correlation,
    local_continuity_meta_criterion,
    neighborhood_preservation,
)
from .retrieval import (
    mean_average_precision,
    r_precision,
    retrieval_precision_at_k,
    same_class_auroc,
)

logger = logging.getLogger(__name__)

#: Every available metric group, in report order.
METRIC_GROUPS: tuple[str, ...] = (
    "projection",
    "classification",
    "retrieval",
    "clustering",
    "confound",
)

#: Groups computed when ``metric_groups`` is not given.
DEFAULT_METRIC_GROUPS: tuple[str, ...] = METRIC_GROUPS

KnnSpace = Literal["low", "high", "both"]

#: A metric entry: the result keys it fills, and a callable producing either a
#: single float or a dict keyed by suffix.
_Entry = tuple[tuple[str, ...], Callable[[], Any]]


def available_metrics() -> dict[str, tuple[str, ...]]:
    """List the result keys each group can contribute.

    Returns
    -------
    dict of str to tuple of str
        Group name to the metric keys it produces. Useful for building a
        results table without running a benchmark first.
    """
    return {
        "projection": (
            "trustworthiness",
            "continuity",
            "neighborhood_preservation",
            "lcmc",
            "distance_correlation",
        ),
        "classification": (
            "knn_accuracy_mean",
            "knn_accuracy_std",
            "knn_accuracy_highdim_mean",
            "knn_accuracy_highdim_std",
            "knn_f1_macro",
            "knn_balanced_accuracy",
            "knn_mcc",
            "knn_cohen_kappa",
            "knn_auroc",
        ),
        "retrieval": (
            "precision_at_k",
            "mean_average_precision",
            "r_precision",
            "same_class_auroc",
        ),
        "clustering": (
            "silhouette",
            "silhouette_2d",
            "adjusted_rand",
            "normalized_mutual_info",
            "homogeneity",
            "completeness",
            "v_measure",
            "fowlkes_mallows",
            "davies_bouldin",
            "calinski_harabasz",
        ),
        "confound": (
            "length_distance_rho",
            "composition_distance_rho",
            "length_only_knn_accuracy",
        ),
    }


def resolve_groups(metric_groups: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Validate and order a requested group selection.

    Raises
    ------
    ValueError
        If a group name is not recognised.
    """
    if metric_groups is None:
        return DEFAULT_METRIC_GROUPS
    requested = tuple(metric_groups)
    unknown = [g for g in requested if g not in METRIC_GROUPS]
    if unknown:
        raise ValueError(f"Unknown metric group(s) {unknown}. Choose from {list(METRIC_GROUPS)}.")
    # Report in canonical order regardless of how they were listed.
    return tuple(g for g in METRIC_GROUPS if g in requested)


def evaluate(
    X_high: EmbeddingMatrix,
    X_low: EmbeddingMatrix,
    y: list[str],
    sequences: list[str] | None = None,
    metric_groups: list[str] | tuple[str, ...] | None = None,
    n_neighbors_trust: int = 10,
    n_neighbors_knn: int = 5,
    precision_at_k: int = 5,
    metric: str = "cosine",
    knn_space: KnnSpace = "both",
    alphabet: str | Alphabet = PROTEIN,
    random_state: int = 0,
) -> dict[str, float]:
    """Score one (embedder, reducer) pair and return a flat metrics dict.

    Parameters
    ----------
    X_high
        Embedding as produced by the embedder, before reduction. Everything
        except ``trustworthiness``, ``continuity``, the other projection
        metrics and ``silhouette_2d`` is computed here, because this is the
        representation under test.
    X_low
        The same samples after dimensionality reduction.
    y
        Group label per sample.
    sequences
        The sequences the embedding came from, in the same order. Required only
        by the ``confound`` group, which is skipped without them.
    metric_groups
        Groups to compute; defaults to all of :data:`METRIC_GROUPS`.
    n_neighbors_trust
        Neighbourhood size for the projection metrics.
    n_neighbors_knn
        Neighbours per vote for the k-NN classifier.
    precision_at_k
        Neighbours retrieved per query for ``precision_at_k``.
    metric
        Distance metric shared by the classification, retrieval, silhouette and
        confound metrics.
    knn_space
        Where to run k-NN: ``"low"`` scores only the projection, ``"high"``
        only the embedding, ``"both"`` (default) reports each — the only way to
        tell an embedding failure apart from a projection failure.
    alphabet
        Token set used to build composition vectors for the confound check.
    random_state
        Seed for cross-validation folds and k-means initialisation.

    Returns
    -------
    dict of str to float
        The metrics that could be computed. See :func:`available_metrics` for
        the full set of keys per group.

    Raises
    ------
    ValueError
        If ``metric_groups`` names an unknown group, or the projection metrics
        cannot run at all (mismatched row counts, fewer than three samples).
    """
    groups = resolve_groups(metric_groups)

    # Trustworthiness is label-free and cheap, so it is the one metric that is
    # never optional -- a run with no usable labels still says something about
    # the projection.
    metrics: dict[str, float] = {}
    if "projection" in groups:
        metrics["trustworthiness"] = compute_trustworthiness(
            X_high, X_low, n_neighbors=n_neighbors_trust
        )

    entries: list[_Entry] = []

    if "projection" in groups:
        entries += [
            (
                ("continuity",),
                lambda: compute_continuity(X_high, X_low, n_neighbors=n_neighbors_trust),
            ),
            (
                ("neighborhood_preservation",),
                lambda: neighborhood_preservation(X_high, X_low, n_neighbors=n_neighbors_trust),
            ),
            (
                ("lcmc",),
                lambda: local_continuity_meta_criterion(
                    X_high, X_low, n_neighbors=n_neighbors_trust
                ),
            ),
            (("distance_correlation",), lambda: distance_correlation(X_high, X_low)),
        ]

    if "classification" in groups:
        if knn_space in ("low", "both"):
            entries.append(
                (
                    ("knn_accuracy_mean", "knn_accuracy_std"),
                    lambda: knn_cv_accuracy(
                        X_low,
                        y,
                        n_neighbors=n_neighbors_knn,
                        metric=metric,
                        random_state=random_state,
                    ),
                )
            )
        if knn_space in ("high", "both"):
            entries.append(
                (
                    ("knn_accuracy_highdim_mean", "knn_accuracy_highdim_std"),
                    lambda: knn_cv_accuracy(
                        X_high,
                        y,
                        n_neighbors=n_neighbors_knn,
                        metric=metric,
                        random_state=random_state,
                    ),
                )
            )
            entries.append(
                (
                    (
                        "knn_f1_macro",
                        "knn_balanced_accuracy",
                        "knn_mcc",
                        "knn_cohen_kappa",
                        "knn_auroc",
                    ),
                    lambda: {
                        f"knn_{key}": value
                        for key, value in knn_classification_report(
                            X_high,
                            y,
                            n_neighbors=n_neighbors_knn,
                            metric=metric,
                            random_state=random_state,
                        ).items()
                        # accuracy is already reported, with a fold-wise std.
                        if key != "accuracy"
                    },
                )
            )

    if "retrieval" in groups:
        entries += [
            (
                ("precision_at_k",),
                lambda: retrieval_precision_at_k(X_high, y, k=precision_at_k, metric=metric),
            ),
            (
                ("mean_average_precision",),
                lambda: mean_average_precision(X_high, y, metric=metric),
            ),
            (("r_precision",), lambda: r_precision(X_high, y, metric=metric)),
            (("same_class_auroc",), lambda: same_class_auroc(X_high, y, metric=metric)),
        ]

    if "clustering" in groups:
        entries += [
            (("silhouette",), lambda: compute_silhouette(X_high, y, metric=metric)),
            (("silhouette_2d",), lambda: compute_silhouette(X_low, y, metric=metric)),
            (
                (
                    "adjusted_rand",
                    "normalized_mutual_info",
                    "homogeneity",
                    "completeness",
                    "v_measure",
                    "fowlkes_mallows",
                ),
                lambda: clustering_report(X_high, y, random_state=random_state),
            ),
            (
                ("davies_bouldin", "calinski_harabasz"),
                lambda: internal_cluster_indices(X_high, y),
            ),
        ]

    if "confound" in groups:
        if sequences is None:
            logger.warning(
                "Skipping the confound group: it needs the original sequences, "
                "which were not supplied."
            )
        else:
            entries += [
                (
                    ("length_distance_rho",),
                    lambda: length_distance_correlation(X_high, sequences, metric=metric),
                ),
                (
                    ("composition_distance_rho",),
                    lambda: composition_distance_correlation(
                        X_high, sequences, metric=metric, alphabet=alphabet
                    ),
                ),
                (
                    ("length_only_knn_accuracy",),
                    lambda: length_only_knn_accuracy(
                        sequences,
                        y,
                        n_neighbors=n_neighbors_knn,
                        random_state=random_state,
                    ),
                ),
            ]

    for keys, compute in entries:
        try:
            value = compute()
        except ValueError as exc:
            logger.warning("Skipping %s: %s", "/".join(keys), exc)
            continue
        if isinstance(value, dict):
            metrics.update({key: float(v) for key, v in value.items()})
        elif isinstance(value, tuple):
            metrics.update(dict(zip(keys, (float(v) for v in value), strict=True)))
        else:
            metrics[keys[0]] = float(value)

    return metrics
