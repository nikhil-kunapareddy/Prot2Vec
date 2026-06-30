"""Tests for evaluation metrics."""
import numpy as np
import pytest

from src.evaluation.metrics import compute_trustworthiness, knn_cv_accuracy, evaluate


def _make_data(n: int = 40, d_high: int = 10, d_low: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_high = rng.standard_normal((n, d_high))
    X_low = rng.standard_normal((n, d_low))
    labels = ["A"] * (n // 2) + ["B"] * (n // 2)
    return X_high, X_low, labels


class TestTrustworthiness:
    def test_range(self):
        X_high, X_low, _ = _make_data()
        tw = compute_trustworthiness(X_high, X_low, n_neighbors=5)
        assert 0.0 <= tw <= 1.0

    def test_perfect_preservation(self):
        # When X_low == X_high[:, :2], trustworthiness should be very high.
        X_high = np.eye(20)
        X_low = X_high[:, :2]
        tw = compute_trustworthiness(X_high, X_low, n_neighbors=5)
        assert tw > 0.8


class TestKnnCvAccuracy:
    def test_returns_tuple(self):
        _, X_low, labels = _make_data()
        mean, std = knn_cv_accuracy(X_low, labels, n_neighbors=3, n_splits=5)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_perfectly_separable(self):
        # Two well-separated clusters → high accuracy.
        X = np.vstack([np.zeros((20, 2)), np.ones((20, 2)) * 100])
        y = ["A"] * 20 + ["B"] * 20
        mean, _ = knn_cv_accuracy(X, y, n_neighbors=3, metric="euclidean")
        assert mean == pytest.approx(1.0)


class TestEvaluate:
    def test_returns_expected_keys(self):
        X_high, X_low, labels = _make_data()
        result = evaluate(X_high, X_low, labels)
        assert set(result.keys()) == {"trustworthiness", "knn_accuracy_mean", "knn_accuracy_std"}

    def test_values_in_valid_range(self):
        X_high, X_low, labels = _make_data()
        result = evaluate(X_high, X_low, labels)
        assert 0.0 <= result["trustworthiness"] <= 1.0
        assert 0.0 <= result["knn_accuracy_mean"] <= 1.0
        assert result["knn_accuracy_std"] >= 0.0
