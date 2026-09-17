"""Backwards-compatible re-exports.

The metric implementations moved into focused modules when the suite grew past
thirty measures — :mod:`~prot2vec.evaluation.projection`,
:mod:`~prot2vec.evaluation.classification`,
:mod:`~prot2vec.evaluation.retrieval`,
:mod:`~prot2vec.evaluation.clustering`,
:mod:`~prot2vec.evaluation.confound` and
:mod:`~prot2vec.evaluation.suite`. Importing from
``prot2vec.evaluation.metrics`` still works, and so does importing from
``prot2vec.evaluation`` or the top-level ``prot2vec`` package.
"""

from __future__ import annotations

from .classification import (
    MIN_MEMBERS_PER_FAMILY,
    knn_classification_report,
    knn_cv_accuracy,
)
from .clustering import (
    clustering_agreement,
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
from .suite import (
    DEFAULT_METRIC_GROUPS,
    METRIC_GROUPS,
    available_metrics,
    evaluate,
    resolve_groups,
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
