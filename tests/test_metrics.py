"""Tests for evaluation metrics."""

import numpy as np
import pytest

from prot2vec.evaluation.metrics import compute_trustworthiness, evaluate, knn_cv_accuracy

#: Metrics reported regardless of the `knn_space` setting, when the confound
#: group is unavailable (no sequences supplied).
ALWAYS_REPORTED = frozenset(
    {
        # projection
        "trustworthiness",
        "continuity",
        "neighborhood_preservation",
        "lcmc",
        "distance_correlation",
        # retrieval
        "precision_at_k",
        "mean_average_precision",
        "r_precision",
        "same_class_auroc",
        # clustering
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
    }
)
KNN_LOW = frozenset({"knn_accuracy_mean", "knn_accuracy_std"})
KNN_HIGH = frozenset(
    {
        "knn_accuracy_highdim_mean",
        "knn_accuracy_highdim_std",
        "knn_f1_macro",
        "knn_balanced_accuracy",
        "knn_mcc",
        "knn_cohen_kappa",
        "knn_auroc",
    }
)
CONFOUND = frozenset(
    {"length_distance_rho", "composition_distance_rho", "length_only_knn_accuracy"}
)


def _make_data(n: int = 40, d_high: int = 10, d_low: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    X_high = rng.standard_normal((n, d_high))
    X_low = rng.standard_normal((n, d_low))
    labels = ["A"] * (n // 2) + ["B"] * (n // 2)
    return X_high, X_low, labels


def _two_clusters(n_per: int = 20, d: int = 8, spread: float = 0.3, seed: int = 0):
    """Two tight clusters separated by ~90 degrees, embedded in ``d`` dimensions.

    All structure lives in the first two dimensions and the rest are exactly
    zero, so ``X[:, :2]`` preserves every pairwise distance -- which makes this
    a valid fixture for "a faithful projection scores 1.0". The clusters are
    offset along different axes rather than around the origin, because cosine
    distance (the default k-NN metric) is undefined in direction for points
    sitting on the origin.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((n_per, d))
    b = np.zeros((n_per, d))
    a[:, 0] = 10.0 + rng.normal(scale=spread, size=n_per)
    a[:, 1] = rng.normal(scale=spread, size=n_per)
    b[:, 0] = rng.normal(scale=spread, size=n_per)
    b[:, 1] = 10.0 + rng.normal(scale=spread, size=n_per)
    return np.vstack([a, b]), ["A"] * n_per + ["B"] * n_per


class TestTrustworthiness:
    def test_range(self):
        X_high, X_low, _ = _make_data()
        tw = compute_trustworthiness(X_high, X_low, n_neighbors=5)
        assert 0.0 <= tw <= 1.0

    def test_identity_projection_is_perfect(self):
        # Projecting onto the axes that carry the structure preserves every
        # neighbourhood, so trustworthiness must be exactly 1.
        X_high, _ = _two_clusters(d=8)
        X_low = X_high[:, :2]
        assert compute_trustworthiness(X_high, X_low, n_neighbors=5) == pytest.approx(1.0)

    def test_structure_destroying_projection_scores_worse_than_identity(self):
        X_high, _ = _two_clusters(d=8)
        faithful = compute_trustworthiness(X_high, X_high[:, :2], n_neighbors=5)
        rng = np.random.default_rng(1)
        shuffled = compute_trustworthiness(
            X_high, rng.standard_normal((X_high.shape[0], 2)), n_neighbors=5
        )
        assert shuffled < faithful

    def test_fully_degenerate_input_is_handled_but_not_meaningful(self):
        # Rows of the identity matrix are mutually equidistant -- every
        # pairwise distance is sqrt(2) -- so the k-neighbourhood of each point
        # is decided entirely by how the sort breaks ties. Projecting to two
        # dimensions then collapses 18 of the 20 points onto the origin.
        #
        # Trustworthiness is therefore *undefined* for this input, not merely
        # low, and its value tracks the library's tie-breaking rather than
        # anything about the data: scikit-learn 1.8 returns 0.27 and 1.9
        # returns 0.95 for exactly this call. Two earlier versions of this test
        # asserted > 0.8 and < 0.5 respectively, and each passed on one of
        # those releases and failed on the other.
        #
        # So assert only what is mathematically guaranteed: a finite value in
        # range, and no crash. The meaningful behaviour is covered by the tests
        # above, which use inputs whose neighbourhoods are well defined.
        X_high = np.eye(20)
        tw = compute_trustworthiness(X_high, X_high[:, :2], n_neighbors=5)
        assert np.isfinite(tw)
        assert 0.0 <= tw <= 1.0

    def test_mismatched_rows_raise(self):
        X_high, X_low, _ = _make_data()
        with pytest.raises(ValueError, match="same samples"):
            compute_trustworthiness(X_high, X_low[:10])

    def test_too_many_neighbors_is_clamped(self):
        X_high, X_low, _ = _make_data(n=10)
        assert 0.0 <= compute_trustworthiness(X_high, X_low, n_neighbors=50) <= 1.0

    def test_accepts_sparse_input(self):
        import scipy.sparse

        X_high, X_low, _ = _make_data()
        tw_dense = compute_trustworthiness(X_high, X_low, n_neighbors=5)
        tw_sparse = compute_trustworthiness(scipy.sparse.csr_matrix(X_high), X_low, n_neighbors=5)
        assert tw_sparse == pytest.approx(tw_dense)


class TestKnnCvAccuracy:
    def test_returns_tuple(self):
        _, X_low, labels = _make_data()
        mean, std = knn_cv_accuracy(X_low, labels, n_neighbors=3, n_splits=5)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_perfectly_separable(self):
        X = np.vstack([np.zeros((20, 2)), np.ones((20, 2)) * 100])
        y = ["A"] * 20 + ["B"] * 20
        mean, _ = knn_cv_accuracy(X, y, n_neighbors=3, metric="euclidean")
        assert mean == pytest.approx(1.0)

    def test_is_deterministic(self):
        _, X_low, labels = _make_data()
        assert knn_cv_accuracy(X_low, labels) == knn_cv_accuracy(X_low, labels)

    def test_folds_reduced_for_small_families(self, caplog):
        # Three members cannot support five stratified folds; the fold count
        # should drop rather than raise.
        X = np.vstack([np.zeros((3, 2)), np.ones((20, 2)) * 100])
        y = ["A"] * 3 + ["B"] * 20
        with caplog.at_level("WARNING"):
            mean, _ = knn_cv_accuracy(X, y, n_neighbors=1, n_splits=5, metric="euclidean")
        assert 0.0 <= mean <= 1.0
        assert "reducing cross-validation" in caplog.text.lower()

    def test_single_family_raises(self):
        X, _ = _two_clusters()
        with pytest.raises(ValueError, match="at least two classes"):
            knn_cv_accuracy(X, ["A"] * X.shape[0])

    def test_singleton_family_raises(self):
        # A family of one cannot appear in both a train and a test fold.
        X = np.vstack([np.zeros((1, 2)), np.ones((20, 2))])
        with pytest.raises(ValueError, match="at least two classes"):
            knn_cv_accuracy(X, ["A"] + ["B"] * 20)

    def test_label_count_mismatch_raises(self):
        X, y = _two_clusters()
        with pytest.raises(ValueError, match="labels"):
            knn_cv_accuracy(X, y[:5])


class TestEvaluate:
    def test_reports_both_spaces_by_default(self):
        X_high, X_low, labels = _make_data()
        result = evaluate(X_high, X_low, labels)
        assert set(result) == ALWAYS_REPORTED | KNN_LOW | KNN_HIGH

    @pytest.mark.parametrize("knn_space", ["low", "high", "both"])
    def test_knn_space_selects_knn_metrics(self, knn_space):
        X_high, X_low, labels = _make_data()
        expected = (
            ALWAYS_REPORTED
            | {
                "low": KNN_LOW,
                "high": KNN_HIGH,
                "both": KNN_LOW | KNN_HIGH,
            }[knn_space]
        )
        assert set(evaluate(X_high, X_low, labels, knn_space=knn_space)) == expected

    def test_degenerate_input_skips_metrics_rather_than_failing(self, caplog):
        # One family makes silhouette, retrieval and clustering meaningless,
        # but trustworthiness is still well defined and should be reported.
        X_high, X_low, _ = _make_data()
        labels = ["A"] * X_high.shape[0]
        with caplog.at_level("WARNING"):
            result = evaluate(X_high, X_low, labels, knn_space="low")
        assert "trustworthiness" in result
        assert "silhouette" not in result
        assert "Skipping" in caplog.text

    def test_metric_panel_is_deterministic(self):
        X_high, X_low, labels = _make_data()
        assert evaluate(X_high, X_low, labels) == evaluate(X_high, X_low, labels)

    def test_values_in_valid_range(self):
        X_high, X_low, labels = _make_data()
        result = evaluate(X_high, X_low, labels)
        assert 0.0 <= result["trustworthiness"] <= 1.0
        assert 0.0 <= result["knn_accuracy_mean"] <= 1.0
        assert result["knn_accuracy_std"] >= 0.0

    def test_separable_data_scores_high_in_both_spaces(self):
        X_high, labels = _two_clusters(d=8)
        result = evaluate(X_high, X_high[:, :2], labels, n_neighbors_knn=3)
        assert result["knn_accuracy_mean"] == pytest.approx(1.0)
        assert result["knn_accuracy_highdim_mean"] == pytest.approx(1.0)
