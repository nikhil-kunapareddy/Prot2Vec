"""Tests for the expanded metric modules and the suite orchestrator."""

import numpy as np
import pytest

from prot2vec.evaluation.classification import knn_classification_report
from prot2vec.evaluation.clustering import (
    clustering_report,
    internal_cluster_indices,
)
from prot2vec.evaluation.confound import (
    composition_distance_correlation,
    length_distance_correlation,
    length_only_knn_accuracy,
)
from prot2vec.evaluation.projection import (
    compute_continuity,
    compute_trustworthiness,
    distance_correlation,
    local_continuity_meta_criterion,
    neighborhood_preservation,
)
from prot2vec.evaluation.retrieval import (
    mean_average_precision,
    r_precision,
    same_class_auroc,
)
from prot2vec.evaluation.suite import (
    METRIC_GROUPS,
    available_metrics,
    evaluate,
    resolve_groups,
)


def separable(n_per=20, d=8, spread=0.3, seed=0):
    """Two clusters separated in angle, with all structure in the first 2 dims."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n_per, d))
    b = np.zeros((n_per, d))
    a[:, 0] = 10.0 + rng.normal(scale=spread, size=n_per)
    a[:, 1] = rng.normal(scale=spread, size=n_per)
    b[:, 0] = rng.normal(scale=spread, size=n_per)
    b[:, 1] = 10.0 + rng.normal(scale=spread, size=n_per)
    return np.vstack([a, b]), ["A"] * n_per + ["B"] * n_per


def noise(n=40, d=8, seed=1):
    return np.random.default_rng(seed).standard_normal((n, d))


class TestProjectionMetrics:
    def test_faithful_projection_scores_perfectly(self):
        X, _ = separable()
        for metric in (
            compute_trustworthiness,
            compute_continuity,
            neighborhood_preservation,
        ):
            assert metric(X, X[:, :2], n_neighbors=5) == pytest.approx(1.0)
        assert distance_correlation(X, X[:, :2]) == pytest.approx(1.0)

    def test_random_projection_scores_near_chance(self):
        X, _ = separable()
        random = noise(n=X.shape[0], d=2)
        assert neighborhood_preservation(X, random, n_neighbors=5) < 0.4
        assert abs(distance_correlation(X, random)) < 0.3

    def test_continuity_is_the_dual_of_trustworthiness(self):
        # Swapping the spaces is the definition, so continuity of (a, b) must
        # equal trustworthiness of (b, a).
        X, _ = separable()
        Z = noise(n=X.shape[0], d=2)
        assert compute_continuity(X, Z, n_neighbors=5) == pytest.approx(
            compute_trustworthiness(Z, X, n_neighbors=5)
        )

    def test_lcmc_subtracts_chance_agreement(self):
        X, _ = separable()
        k, n = 5, X.shape[0]
        overlap = neighborhood_preservation(X, X[:, :2], n_neighbors=k)
        lcmc = local_continuity_meta_criterion(X, X[:, :2], n_neighbors=k)
        assert lcmc == pytest.approx(overlap - k / (n - 1))

    def test_lcmc_near_zero_for_random_projection(self):
        X, _ = separable()
        assert abs(local_continuity_meta_criterion(X, noise(n=40, d=2), n_neighbors=5)) < 0.15

    def test_neighborhood_preservation_handles_duplicate_points(self):
        duplicates = np.vstack([np.zeros((5, 3)), np.ones((5, 3))])
        assert neighborhood_preservation(duplicates, duplicates[:, :2], n_neighbors=3) == 1.0

    def test_mismatched_rows_rejected(self):
        X, _ = separable()
        with pytest.raises(ValueError, match="same samples"):
            compute_continuity(X, X[:5, :2])


class TestClassificationPanel:
    def test_separable_data_scores_perfectly(self):
        X, y = separable()
        report = knn_classification_report(X, y, n_neighbors=3)
        for key in ("accuracy", "f1_macro", "balanced_accuracy", "mcc", "cohen_kappa"):
            assert report[key] == pytest.approx(1.0)

    def test_reports_every_metric(self):
        X, y = separable()
        report = knn_classification_report(X, y, n_neighbors=3)
        assert {"accuracy", "f1_macro", "balanced_accuracy", "mcc", "cohen_kappa"} <= set(report)

    def test_multiclass_auroc(self):
        rng = np.random.default_rng(0)
        blocks, labels = [], []
        for i, group in enumerate("ABC"):
            block = np.zeros((15, 8))
            block[:, i] = 10 + rng.normal(scale=0.3, size=15)
            blocks.append(block)
            labels += [group] * 15
        report = knn_classification_report(np.vstack(blocks), labels, n_neighbors=3)
        assert report["auroc"] == pytest.approx(1.0)

    def test_unbalanced_data_exposes_the_accuracy_illusion(self):
        # This is why accuracy alone is not reported: a predictor that ignores
        # the small class still scores well on it.
        X = noise(n=40)
        y = ["A"] * 36 + ["B"] * 4
        report = knn_classification_report(X, y, n_neighbors=3)
        assert report["accuracy"] > report["balanced_accuracy"]
        assert report["mcc"] < 0.5

    def test_degenerate_labels_rejected(self):
        X, _ = separable()
        with pytest.raises(ValueError, match="at least two classes"):
            knn_classification_report(X, ["A"] * X.shape[0])


class TestRetrievalMetrics:
    def test_separable_data_scores_high(self):
        X, y = separable()
        assert mean_average_precision(X, y) > 0.95
        assert r_precision(X, y) == pytest.approx(1.0)
        assert same_class_auroc(X, y) == pytest.approx(1.0)

    def test_random_data_scores_near_chance(self):
        y = ["A"] * 20 + ["B"] * 20
        X = noise()
        assert 0.35 < mean_average_precision(X, y) < 0.65
        assert 0.35 < r_precision(X, y) < 0.65
        assert 0.35 < same_class_auroc(X, y) < 0.65

    def test_auroc_is_insensitive_to_class_imbalance(self):
        balanced_X, balanced_y = separable(n_per=20)
        assert same_class_auroc(balanced_X, balanced_y) == pytest.approx(1.0)
        rng = np.random.default_rng(3)
        big = np.zeros((35, 8))
        big[:, 0] = 10 + rng.normal(scale=0.3, size=35)
        small = np.zeros((5, 8))
        small[:, 1] = 10 + rng.normal(scale=0.3, size=5)
        unbalanced_y = ["A"] * 35 + ["B"] * 5
        assert same_class_auroc(np.vstack([big, small]), unbalanced_y) == pytest.approx(1.0)

    @pytest.mark.parametrize("metric", [mean_average_precision, r_precision, same_class_auroc])
    def test_single_class_rejected(self, metric):
        X, _ = separable()
        with pytest.raises(ValueError, match="at least two"):
            metric(X, ["A"] * X.shape[0])


class TestClusteringPanel:
    def test_separable_data_recovers_the_partition(self):
        X, y = separable()
        report = clustering_report(X, y)
        for value in report.values():
            assert value == pytest.approx(1.0)

    def test_random_data_is_near_chance(self):
        y = ["A"] * 20 + ["B"] * 20
        report = clustering_report(noise(), y)
        assert abs(report["adjusted_rand"]) < 0.2
        assert report["normalized_mutual_info"] < 0.2

    def test_homogeneity_and_completeness_fail_differently(self):
        # One true group split across two tight clusters: the clusters stay
        # relatively pure (homogeneous) while the group is torn apart
        # (incomplete). Reporting only their harmonic mean would hide which of
        # the two failures occurred.
        rng = np.random.default_rng(0)
        left = np.zeros((15, 4))
        left[:, 0] = rng.normal(scale=0.1, size=15)
        right = np.zeros((15, 4))
        right[:, 0] = 50 + rng.normal(scale=0.1, size=15)
        X = np.vstack([left, right])
        y = ["A"] * 30  # essentially one label spread over two obvious clusters
        y[0] = "B"  # a second label, so the metric has something to compare
        report = clustering_report(X, y)
        assert report["homogeneity"] > report["completeness"]
        assert report["completeness"] < 0.2  # class A is split in half

    def test_internal_indices_agree_in_direction(self):
        X, y = separable()
        good = internal_cluster_indices(X, y)
        bad = internal_cluster_indices(noise(), y)
        assert good["davies_bouldin"] < bad["davies_bouldin"]
        assert good["calinski_harabasz"] > bad["calinski_harabasz"]

    def test_single_group_rejected(self):
        X, _ = separable()
        with pytest.raises(ValueError, match="at least two groups"):
            clustering_report(X, ["A"] * X.shape[0])


class TestConfoundMetrics:
    def test_length_confound_is_detected(self):
        rng = np.random.default_rng(0)
        sequences = ["A" * int(rng.integers(50, 60)) for _ in range(20)]
        sequences += ["A" * int(rng.integers(200, 220)) for _ in range(20)]
        lengths = np.array([[len(s)] for s in sequences], dtype=float)
        assert length_distance_correlation(lengths, sequences, metric="euclidean") > 0.9

    def test_length_only_baseline_separates_length_split_groups(self):
        rng = np.random.default_rng(0)
        sequences = ["A" * int(rng.integers(50, 60)) for _ in range(20)]
        sequences += ["A" * int(rng.integers(200, 220)) for _ in range(20)]
        y = ["short"] * 20 + ["long"] * 20
        assert length_only_knn_accuracy(sequences, y) == pytest.approx(1.0)

    def test_length_only_baseline_is_chance_when_lengths_match(self):
        rng = np.random.default_rng(0)
        sequences = ["".join(rng.choice(list("ACDEFG"), 120)) for _ in range(40)]
        y = ["A"] * 20 + ["B"] * 20
        assert length_only_knn_accuracy(sequences, y) == pytest.approx(0.5, abs=0.2)

    def test_composition_embedder_correlates_with_composition_by_construction(self):
        from prot2vec.embedders.composition import CompositionEmbedder

        rng = np.random.default_rng(0)
        sequences = ["".join(rng.choice(list("AGST"), 120)) for _ in range(20)]
        sequences += ["".join(rng.choice(list("DEKR"), 120)) for _ in range(20)]
        X = CompositionEmbedder().fit_transform(sequences)
        assert composition_distance_correlation(X, sequences) > 0.95

    def test_random_embedding_does_not_correlate_with_composition(self):
        rng = np.random.default_rng(0)
        sequences = ["".join(rng.choice(list("ACDEFG"), 120)) for _ in range(40)]
        assert abs(composition_distance_correlation(noise(), sequences)) < 0.3

    def test_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="sequences were given"):
            length_distance_correlation(noise(), ["AAA"] * 5)

    def test_dna_alphabet_supported(self):
        rng = np.random.default_rng(0)
        sequences = ["".join(rng.choice(list("ACGT"), 100)) for _ in range(20)]
        value = composition_distance_correlation(noise(n=20, d=4), sequences, alphabet="dna")
        assert -1.0 <= value <= 1.0


class TestSuite:
    def test_group_catalogue_matches_group_names(self):
        assert set(available_metrics()) == set(METRIC_GROUPS)

    def test_catalogue_is_substantial(self):
        assert sum(len(keys) for keys in available_metrics().values()) >= 30

    def test_resolve_defaults_to_everything(self):
        assert resolve_groups(None) == METRIC_GROUPS

    def test_resolve_returns_canonical_order(self):
        assert resolve_groups(["clustering", "projection"]) == ("projection", "clustering")

    def test_resolve_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown metric group"):
            resolve_groups(["projection", "nonsense"])

    def test_all_groups_reported_when_sequences_supplied(self):
        rng = np.random.default_rng(0)
        sequences = ["".join(rng.choice(list("AGST"), 120)) for _ in range(20)]
        sequences += ["".join(rng.choice(list("DEKR"), 120)) for _ in range(20)]
        y = ["A"] * 20 + ["B"] * 20
        from prot2vec.embedders.composition import CompositionEmbedder
        from prot2vec.reduction.reducers import PCAReducer

        X = CompositionEmbedder().fit_transform(sequences)
        Z = PCAReducer().fit_transform(X)
        result = evaluate(X, Z, y, sequences=sequences)
        expected = {k for keys in available_metrics().values() for k in keys}
        assert set(result) == expected

    def test_selecting_one_group_narrows_the_output(self):
        X, y = separable()
        result = evaluate(X, X[:, :2], y, metric_groups=["retrieval"])
        assert set(result) == set(available_metrics()["retrieval"])

    def test_confound_group_skipped_without_sequences(self, caplog):
        X, y = separable()
        with caplog.at_level("WARNING"):
            result = evaluate(X, X[:, :2], y, metric_groups=["confound"])
        assert result == {}
        assert "confound" in caplog.text

    def test_knn_space_restricts_classification_only(self):
        X, y = separable()
        low = evaluate(X, X[:, :2], y, metric_groups=["classification"], knn_space="low")
        high = evaluate(X, X[:, :2], y, metric_groups=["classification"], knn_space="high")
        assert "knn_accuracy_mean" in low
        assert "knn_accuracy_highdim_mean" not in low
        assert "knn_accuracy_highdim_mean" in high
        assert "knn_accuracy_mean" not in high

    def test_degenerate_labels_skip_rather_than_fail(self, caplog):
        X, _ = separable()
        with caplog.at_level("WARNING"):
            result = evaluate(X, X[:, :2], ["A"] * X.shape[0])
        # Projection metrics are label-free and must survive.
        assert "trustworthiness" in result
        assert "adjusted_rand" not in result
        assert "Skipping" in caplog.text

    def test_deterministic(self):
        X, y = separable()
        assert evaluate(X, X[:, :2], y) == evaluate(X, X[:, :2], y)
