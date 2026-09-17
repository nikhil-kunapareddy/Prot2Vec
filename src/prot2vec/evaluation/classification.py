"""Can a classifier recover the labels from the embedding?

A cross-validated k-nearest-neighbour classifier is the standard proxy, and
accuracy alone is a poor way to report it. Protein family datasets are
unbalanced — Pfam seed alignments differ in size by more than an order of
magnitude — and on an unbalanced problem accuracy rewards a model for ignoring
the small classes. The alternatives here are the ones that do not:
``balanced_accuracy`` averages recall per class, ``f1_macro`` weights every
class equally regardless of size, and ``mcc`` stays near zero for any
degenerate predictor.

k-NN is used rather than a linear model deliberately. It makes no assumption
about the shape of the class boundaries, so a low score means the label
information is genuinely absent from the neighbourhood structure rather than
merely non-linear.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from .._matrix import EmbeddingMatrix, as_dense

logger = logging.getLogger(__name__)

#: Minimum members a class needs before it can appear in a stratified split.
MIN_MEMBERS_PER_FAMILY = 2


def _prepare(
    X: EmbeddingMatrix,
    y: list[str],
    n_neighbors: int,
    n_splits: int,
) -> tuple[np.ndarray, int, int]:
    """Validate inputs and clamp fold count and ``k`` to what the data supports.

    Returns
    -------
    tuple
        ``(dense_X, effective_splits, effective_neighbors)``.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` disagree on sample count, or fewer than two classes
        have enough members to be classified.
    """
    dense = as_dense(X)
    if dense.shape[0] != len(y):
        raise ValueError(f"X has {dense.shape[0]} rows but {len(y)} labels were given.")

    counts = Counter(y)
    usable = {name: n for name, n in counts.items() if n >= MIN_MEMBERS_PER_FAMILY}
    if len(usable) < 2:
        raise ValueError(
            "Classification needs at least two classes with "
            f"{MIN_MEMBERS_PER_FAMILY}+ members each; got {dict(counts)}. "
            "Lower `min_seq_length` or pick groups with more sequences."
        )

    smallest = min(counts.values())
    splits = min(n_splits, smallest)
    if splits < n_splits:
        logger.warning(
            "Smallest class has %d members; reducing cross-validation from %d to %d folds.",
            smallest,
            n_splits,
            splits,
        )

    # Every fold holds out 1/k of the data, so the training set can be smaller
    # than the requested neighbourhood.
    ceiling = int(len(y) * (splits - 1) / splits)
    neighbors = max(1, min(n_neighbors, ceiling))
    if neighbors < n_neighbors:
        logger.warning(
            "Reducing k from %d to %d to fit the training folds.", n_neighbors, neighbors
        )

    return dense, splits, neighbors


def knn_cv_accuracy(
    X: EmbeddingMatrix,
    y: list[str],
    n_neighbors: int = 5,
    n_splits: int = 5,
    metric: str = "cosine",
    random_state: int = 0,
) -> tuple[float, float]:
    """Cross-validated k-NN accuracy for recovering labels from ``X``.

    Parameters
    ----------
    X
        Embedding to classify.
    y
        Label per sample.
    n_neighbors
        Neighbours per vote.
    n_splits
        Requested number of stratified folds.
    metric
        Distance metric for the classifier.
    random_state
        Seed for the fold shuffle.

    Returns
    -------
    tuple of float
        ``(mean_accuracy, std_accuracy)`` across folds.

    Raises
    ------
    ValueError
        See :func:`_prepare`.
    """
    dense, splits, neighbors = _prepare(X, y, n_neighbors, n_splits)
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(knn, dense, y, cv=cv)
    return float(scores.mean()), float(scores.std())


def knn_classification_report(
    X: EmbeddingMatrix,
    y: list[str],
    n_neighbors: int = 5,
    n_splits: int = 5,
    metric: str = "cosine",
    random_state: int = 0,
) -> dict[str, float]:
    """Run one cross-validation and report the full classification panel.

    Predictions are generated once with
    :func:`sklearn.model_selection.cross_val_predict` and every metric is
    derived from them, so this costs about the same as computing accuracy alone
    rather than refitting the classifier per metric.

    Parameters
    ----------
    X
        Embedding to classify.
    y
        Label per sample.
    n_neighbors
        Neighbours per vote.
    n_splits
        Requested number of stratified folds.
    metric
        Distance metric for the classifier.
    random_state
        Seed for the fold shuffle.

    Returns
    -------
    dict of str to float
        ``accuracy``, ``f1_macro``, ``balanced_accuracy``, ``mcc``,
        ``cohen_kappa`` and, when probabilities are usable, ``auroc``.
        ``auroc`` is one-vs-rest and macro-averaged.

    Raises
    ------
    ValueError
        See :func:`_prepare`.
    """
    dense, splits, neighbors = _prepare(X, y, n_neighbors, n_splits)
    labels = np.asarray(y)
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric=metric)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

    predicted = cross_val_predict(knn, dense, labels, cv=cv)
    report = {
        "accuracy": float((predicted == labels).mean()),
        "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "cohen_kappa": float(cohen_kappa_score(labels, predicted)),
    }

    # AUROC needs calibrated-ish probabilities. k-NN gives vote fractions,
    # which are coarse but ordered, and it fails outright if a fold never sees
    # a class -- so treat it as best-effort rather than required.
    try:
        probabilities = cross_val_predict(knn, dense, labels, cv=cv, method="predict_proba")
        classes = np.unique(labels)
        if len(classes) == 2:
            report["auroc"] = float(
                roc_auc_score((labels == classes[1]).astype(int), probabilities[:, 1])
            )
        else:
            report["auroc"] = float(
                roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")
            )
    except (ValueError, IndexError) as exc:
        logger.warning("Skipping auroc: %s", exc)

    return report
