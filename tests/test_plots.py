"""Tests for the figure helpers."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prot2vec.visualization.plots import metrics_bar_chart, scatter_2d


@pytest.fixture(autouse=True)
def no_leaked_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def projection():
    rng = np.random.default_rng(0)
    return rng.standard_normal((20, 2)), ["A"] * 10 + ["B"] * 10


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "method": ["composition+pca", "kmer_k3+pca"],
            "knn_accuracy_mean": [0.9, 0.8],
            "knn_accuracy_std": [0.01, 0.02],
        }
    )


class TestScatter2d:
    def test_returns_a_figure(self, projection):
        Z, labels = projection
        assert scatter_2d(Z, labels, "title") is not None

    def test_saves_to_disk_and_creates_parents(self, projection, tmp_path):
        Z, labels = projection
        path = tmp_path / "nested" / "fig.png"
        scatter_2d(Z, labels, "title", save_path=path)
        assert path.exists()

    def test_closed_after_saving_but_not_otherwise(self, projection, tmp_path):
        # Long benchmark matrices must not accumulate open figures.
        Z, labels = projection
        scatter_2d(Z, labels, "t", save_path=tmp_path / "a.png")
        assert plt.get_fignums() == []
        scatter_2d(Z, labels, "t")
        assert len(plt.get_fignums()) == 1

    def test_close_can_be_forced(self, projection):
        Z, labels = projection
        scatter_2d(Z, labels, "t", close=True)
        assert plt.get_fignums() == []

    def test_legend_reports_group_sizes(self, projection):
        Z, labels = projection
        fig = scatter_2d(Z, labels, "t")
        texts = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("n=10" in t for t in texts)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            scatter_2d(np.zeros((10, 1)), ["A"] * 10, "t")

    def test_rejects_label_mismatch(self, projection):
        Z, labels = projection
        with pytest.raises(ValueError, match="labels"):
            scatter_2d(Z, labels[:5], "t")


class TestMetricsBarChart:
    def test_returns_a_figure(self, results):
        assert metrics_bar_chart(results) is not None

    def test_saves_to_disk(self, results, tmp_path):
        path = tmp_path / "bar.png"
        metrics_bar_chart(results, save_path=path)
        assert path.exists()
        assert plt.get_fignums() == []

    def test_unknown_metric_rejected(self, results):
        with pytest.raises(ValueError, match="not in the results columns"):
            metrics_bar_chart(results, metric="nope")

    def test_metric_without_std_column(self, results):
        frame = results.drop(columns=["knn_accuracy_std"])
        assert metrics_bar_chart(frame) is not None
