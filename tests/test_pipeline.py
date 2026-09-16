"""Tests for run orchestration: caching, exports, manifest and the run matrix."""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from prot2vec.embedders.kmer import KmerEmbedder
from prot2vec.pipeline import RunConfig, _load_embedding, _save_embedding, run
from prot2vec.reduction.reducers import PCAReducer
from tests.conftest import ConstantEmbedder, ShortEmbedder


@pytest.fixture
def base_config(two_family_dataset, tmp_path):
    return RunConfig(
        dataset=two_family_dataset,
        embedders=[ConstantEmbedder()],
        reducers=[PCAReducer()],
        results_dir=tmp_path / "results",
        save_figures=False,
    )


class TestRunConfig:
    def test_single_reducer_normalised_to_list(self, two_family_dataset):
        config = RunConfig(
            dataset=two_family_dataset,
            embedders=[ConstantEmbedder()],
            reducers=PCAReducer(),
        )
        assert isinstance(config.reducers, list)
        assert len(config.reducers) == 1

    def test_n_steps_covers_the_matrix(self, two_family_dataset):
        config = RunConfig(
            dataset=two_family_dataset,
            embedders=[ConstantEmbedder(tag="a"), ConstantEmbedder(tag="b")],
            reducers=[PCAReducer(), PCAReducer()],
        )
        # 2 embeds + 2 embedders * 2 reducers * (reduce + score)
        assert config.n_steps == 2 * (1 + 2 * 2)

    def test_requires_an_embedder(self, two_family_dataset):
        with pytest.raises(ValueError, match="one embedder"):
            RunConfig(dataset=two_family_dataset, embedders=[], reducers=[PCAReducer()])

    def test_requires_a_reducer(self, two_family_dataset):
        with pytest.raises(ValueError, match="one reducer"):
            RunConfig(dataset=two_family_dataset, embedders=[ConstantEmbedder()], reducers=[])

    def test_rejects_single_family(self, two_family_dataset):
        two_family_dataset.labels = ["PF00069"] * len(two_family_dataset)
        with pytest.raises(ValueError, match="at least two"):
            RunConfig(
                dataset=two_family_dataset,
                embedders=[ConstantEmbedder()],
                reducers=[PCAReducer()],
            )


class TestEmbeddingCache:
    def test_sparse_roundtrip_preserves_sparsity(self, tmp_path):
        X = scipy.sparse.random(10, 50, density=0.2, format="csr", random_state=0)
        path = _save_embedding(tmp_path / "emb", X)
        assert path.suffix == ".npz"
        loaded = _load_embedding(tmp_path / "emb")
        assert scipy.sparse.issparse(loaded)
        assert (loaded != X).nnz == 0

    def test_dense_roundtrip(self, tmp_path):
        X = np.arange(20, dtype=np.float32).reshape(4, 5)
        path = _save_embedding(tmp_path / "emb", X)
        assert path.suffix == ".npy"
        np.testing.assert_array_equal(_load_embedding(tmp_path / "emb"), X)

    def test_missing_cache_returns_none(self, tmp_path):
        assert _load_embedding(tmp_path / "absent") is None

    def test_second_run_reuses_the_cache(self, base_config):
        embedder = base_config.embedders[0]
        run(base_config)
        assert embedder.call_count == 1
        run(base_config)
        assert embedder.call_count == 1  # loaded from disk, not recomputed

    def test_cache_can_be_disabled(self, base_config):
        base_config.cache_embeddings = False
        embedder = base_config.embedders[0]
        run(base_config)
        run(base_config)
        assert embedder.call_count == 2

    def test_cache_key_separates_embedder_settings(self, two_family_dataset, tmp_path):
        # Two k-mer embedders differing only in k must not share a cache entry.
        config = RunConfig(
            dataset=two_family_dataset,
            embedders=[KmerEmbedder(k=2), KmerEmbedder(k=3)],
            reducers=[PCAReducer()],
            results_dir=tmp_path / "results",
            save_figures=False,
        )
        run(config)
        cached = list((tmp_path / "results" / "embeddings").glob("kmer_k*_*.npz"))
        assert len({p.name for p in cached}) == 2


class TestRunMatrix:
    def test_every_pair_is_scored(self, two_family_dataset, tmp_path):
        config = RunConfig(
            dataset=two_family_dataset,
            embedders=[ConstantEmbedder(tag="a"), ConstantEmbedder(tag="b")],
            reducers=[PCAReducer(), PCAReducer()],
            results_dir=tmp_path / "results",
            save_figures=False,
        )
        results = run(config)
        assert len(results) == 4
        assert {"embedder", "reducer", "method"} <= set(results.columns)

    def test_embedding_computed_once_per_embedder(self, two_family_dataset, tmp_path):
        embedder = ConstantEmbedder()
        config = RunConfig(
            dataset=two_family_dataset,
            embedders=[embedder],
            reducers=[PCAReducer(), PCAReducer(), PCAReducer()],
            results_dir=tmp_path / "results",
            cache_embeddings=False,
            save_figures=False,
        )
        run(config)
        assert embedder.call_count == 1  # reused across all three reducers

    def test_metrics_csv_written(self, base_config):
        run(base_config)
        csv = base_config.results_dir / "metrics" / "benchmark.csv"
        assert csv.exists()
        assert len(pd.read_csv(csv)) == 1

    def test_metric_params_forwarded(self, base_config):
        base_config.metric_params = {"knn_space": "high"}
        results = run(base_config)
        assert "knn_accuracy_highdim_mean" in results.columns
        assert "knn_accuracy_mean" not in results.columns

    def test_progress_callback_receives_events(self, base_config):
        events = []
        base_config.on_progress = lambda event, payload: events.append(event)
        run(base_config)
        assert events.count("embed_done") == 1
        assert events.count("evaluate_done") == 1

    def test_row_count_violation_is_caught(self, two_family_dataset, tmp_path):
        config = RunConfig(
            dataset=two_family_dataset,
            embedders=[ShortEmbedder()],
            reducers=[PCAReducer()],
            results_dir=tmp_path / "results",
            save_figures=False,
        )
        with pytest.raises(RuntimeError, match="rows"):
            run(config)


class TestExports:
    def test_sequence_index_is_the_join_key(self, base_config):
        run(base_config)
        index = pd.read_csv(base_config.results_dir / "embeddings" / "sequences.tsv", sep="\t")
        assert list(index.columns) == ["row", "sequence_id", "family", "length", "sequence"]
        assert len(index) == len(base_config.dataset)
        assert index.sequence_id.is_unique

    def test_embedding_export_aligns_with_index(self, base_config):
        run(base_config)
        index = pd.read_csv(base_config.results_dir / "embeddings" / "sequences.tsv", sep="\t")
        vectors = np.load(base_config.results_dir / "embeddings" / "constant.npy")
        assert vectors.shape[0] == len(index)

    def test_projection_export_is_tidy(self, base_config):
        run(base_config)
        proj = pd.read_csv(base_config.results_dir / "projections" / "constant__pca.tsv", sep="\t")
        assert list(proj.columns) == ["sequence_id", "family", "dim1", "dim2"]
        assert list(proj.sequence_id) == list(base_config.dataset.ids)

    def test_exports_can_be_disabled(self, base_config):
        base_config.export_embeddings = False
        run(base_config)
        assert not (base_config.results_dir / "projections").exists()
        assert not (base_config.results_dir / "embeddings" / "sequences.tsv").exists()


class TestManifest:
    def test_records_provenance(self, base_config):
        results = run(base_config)
        manifest = json.loads((base_config.results_dir / "run_manifest.json").read_text())
        assert manifest["prot2vec_version"]
        assert manifest["generated_at"]
        assert manifest["python"]
        assert manifest["dataset"]["n_sequences"] == len(base_config.dataset)
        assert [e["name"] for e in manifest["embedders"]] == ["constant"]
        assert len(manifest["results"]) == len(results)

    def test_records_reducer_parameters(self, two_family_dataset, tmp_path):
        config = RunConfig(
            dataset=two_family_dataset,
            embedders=[ConstantEmbedder()],
            reducers=[PCAReducer(n_components=2, random_state=42)],
            results_dir=tmp_path / "results",
            save_figures=False,
        )
        run(config)
        manifest = json.loads((tmp_path / "results" / "run_manifest.json").read_text())
        assert manifest["reducers"][0]["settings"]["random_state"] == 42

    def test_dependency_versions_captured(self, base_config):
        run(base_config)
        manifest = json.loads((base_config.results_dir / "run_manifest.json").read_text())
        assert "numpy" in manifest["dependencies"]


class TestFigures:
    def test_figures_written_when_enabled(self, base_config):
        base_config.save_figures = True
        run(base_config)
        figures = base_config.results_dir / "figures"
        assert (figures / "constant_pca.png").exists()
        assert (figures / "metrics_summary.png").exists()
