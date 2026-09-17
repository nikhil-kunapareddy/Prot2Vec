"""Evaluation metrics, grouped by the question they answer.

``projection``
    Did the reducer preserve structure, or invent it?
``classification``
    Can the labels be recovered from the representation?
``retrieval``
    Would a nearest-neighbour lookup return true group members?
``clustering``
    Is the structure findable without labels?
``confound``
    Is the embedding measuring group identity, or just length and composition?

:func:`evaluate` assembles the panel; every metric is also importable and
usable on its own.
"""

from .metrics import (
    DEFAULT_METRIC_GROUPS,
    METRIC_GROUPS,
    MIN_MEMBERS_PER_FAMILY,
    available_metrics,
    clustering_agreement,
    clustering_report,
    composition_distance_correlation,
    compute_continuity,
    compute_silhouette,
    compute_trustworthiness,
    distance_correlation,
    evaluate,
    internal_cluster_indices,
    knn_classification_report,
    knn_cv_accuracy,
    length_distance_correlation,
    length_only_knn_accuracy,
    local_continuity_meta_criterion,
    mean_average_precision,
    neighborhood_preservation,
    r_precision,
    resolve_groups,
    retrieval_precision_at_k,
    same_class_auroc,
)

__all__ = [
    "DEFAULT_METRIC_GROUPS",
    "METRIC_GROUPS",
    "MIN_MEMBERS_PER_FAMILY",
    "available_metrics",
    "clustering_agreement",
    "clustering_report",
    "composition_distance_correlation",
    "compute_continuity",
    "compute_silhouette",
    "compute_trustworthiness",
    "distance_correlation",
    "evaluate",
    "internal_cluster_indices",
    "knn_classification_report",
    "knn_cv_accuracy",
    "length_distance_correlation",
    "length_only_knn_accuracy",
    "local_continuity_meta_criterion",
    "mean_average_precision",
    "neighborhood_preservation",
    "r_precision",
    "resolve_groups",
    "retrieval_precision_at_k",
    "same_class_auroc",
]
