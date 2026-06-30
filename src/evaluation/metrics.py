"""Evaluation metrics for embedding quality."""
from __future__ import annotations

import numpy as np
from sklearn.manifold import trustworthiness as _trustworthiness
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def compute_trustworthiness(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 10,
    metric: str = "euclidean",
) -> float:
    """Measures how well local neighborhoods in X_high are preserved in X_low.

    Returns a value in (0, 1]; higher is better.
    """
    return float(_trustworthiness(X_high, X_low, n_neighbors=n_neighbors, metric=metric))


def knn_cv_accuracy(
    X: np.ndarray,
    y: list[str],
    n_neighbors: int = 5,
    n_splits: int = 5,
    metric: str = "cosine",
) -> tuple[float, float]:
    """Stratified k-fold cross-validated kNN accuracy on the 2D embedding.

    Returns (mean_accuracy, std_accuracy).
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(knn, X, y, cv=cv)
    return float(scores.mean()), float(scores.std())


def evaluate(
    X_high: np.ndarray,
    X_low: np.ndarray,
    y: list[str],
    n_neighbors_trust: int = 10,
    n_neighbors_knn: int = 5,
) -> dict[str, float]:
    """Run all metrics and return a flat dict."""
    tw = compute_trustworthiness(X_high, X_low, n_neighbors=n_neighbors_trust)
    acc_mean, acc_std = knn_cv_accuracy(X_low, y, n_neighbors=n_neighbors_knn)
    return {
        "trustworthiness": tw,
        "knn_accuracy_mean": acc_mean,
        "knn_accuracy_std": acc_std,
    }
