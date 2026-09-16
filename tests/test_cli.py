"""Tests for the CLI: config parsing, object construction and rendering."""

import pandas as pd
import pytest
import yaml

from prot2vec.cli import (
    _build_embedder,
    _build_parser,
    _build_reducer,
    _load_config,
    _load_dataset,
    _reducer_configs,
    print_config_summary,
    print_results_table,
)


def write_config(tmp_path, **overrides):
    config = {
        "fasta": {"path": "unused.fasta"},
        "embedders": [{"name": "composition"}],
        "reducer": {"name": "pca"},
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class TestLoadConfig:
    def test_valid_config(self, tmp_path):
        assert _load_config(write_config(tmp_path))["embedders"]

    def test_missing_file_points_at_the_presets(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _load_config(tmp_path / "absent.yaml")
        assert "configs/experiments" in capsys.readouterr().out

    def test_unparseable_yaml(self, tmp_path, capsys):
        path = tmp_path / "bad.yaml"
        path.write_text("embedders: [unclosed\n")
        with pytest.raises(SystemExit):
            _load_config(path)
        assert "Could not parse" in capsys.readouterr().out

    def test_non_mapping_rejected(self, tmp_path, capsys):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(SystemExit):
            _load_config(path)
        assert "mapping" in capsys.readouterr().out

    def test_missing_embedders(self, tmp_path, capsys):
        path = write_config(tmp_path)
        config = yaml.safe_load(path.read_text())
        del config["embedders"]
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(SystemExit):
            _load_config(path)
        assert "embedders" in capsys.readouterr().out

    def test_missing_reducer_and_reducers(self, tmp_path, capsys):
        path = write_config(tmp_path)
        config = yaml.safe_load(path.read_text())
        del config["reducer"]
        path.write_text(yaml.safe_dump(config))
        with pytest.raises(SystemExit):
            _load_config(path)
        assert "reducers" in capsys.readouterr().out

    def test_reducers_list_satisfies_validation(self, tmp_path):
        path = write_config(tmp_path, reducer=None, reducers=[{"name": "pca"}])
        assert _load_config(path)["reducers"]

    def test_missing_data_source(self, tmp_path, capsys):
        path = write_config(tmp_path, fasta=None)
        with pytest.raises(SystemExit):
            _load_config(path)
        assert "pfam" in capsys.readouterr().out


class TestReducerConfigs:
    def test_single_reducer_mapping(self):
        assert _reducer_configs({"reducer": {"name": "pca"}}) == [{"name": "pca"}]

    def test_reducers_list_takes_precedence(self):
        config = {"reducer": {"name": "pca"}, "reducers": [{"name": "umap"}]}
        assert _reducer_configs(config) == [{"name": "umap"}]

    def test_single_mapping_under_reducers_key(self):
        assert _reducer_configs({"reducers": {"name": "tsne"}}) == [{"name": "tsne"}]


class TestBuilders:
    @pytest.mark.parametrize(
        ("config", "expected_name"),
        [
            ({"name": "composition"}, "composition"),
            ({"name": "kmer", "k": 4}, "kmer_k4"),
            ({"name": "esm2", "model_key": "esm2_t6_8M"}, "esm_esm2_t6_8M"),
            ({"name": "llm"}, "llm_google_gemini-embedding-001"),
        ],
    )
    def test_build_embedder(self, config, expected_name):
        assert _build_embedder(config).name == expected_name

    def test_kmer_min_df_forwarded(self):
        assert _build_embedder({"name": "kmer", "k": 3, "min_df": 2}).min_df == 2

    def test_llm_output_dim_forwarded(self):
        assert _build_embedder({"name": "llm", "output_dim": 768}).output_dim == 768

    def test_unknown_embedder_lists_the_options(self):
        with pytest.raises(ValueError, match="composition, kmer, esm2, llm"):
            _build_embedder({"name": "prot-bert"})

    @pytest.mark.parametrize("name", ["pca", "umap", "tsne"])
    def test_build_reducer(self, name):
        assert _build_reducer({"name": name}).name == name

    def test_reducer_params_forwarded(self):
        reducer = _build_reducer({"name": "umap", "n_neighbors": 7, "metric": "euclidean"})
        assert reducer.params["n_neighbors"] == 7
        assert reducer.params["metric"] == "euclidean"

    def test_unknown_reducer_lists_the_options(self):
        with pytest.raises(ValueError, match="pca, umap, tsne"):
            _build_reducer({"name": "phate"})


class TestLoadDataset:
    def test_from_fasta_block(self, fasta_file):
        config = {"fasta": {"path": str(fasta_file)}, "data": {"min_seq_length": 50}}
        dataset = _load_dataset(config)
        assert len(dataset) == 4
        assert dataset.families == ["PF00069", "PF00072"]

    def test_fasta_as_bare_string(self, fasta_file):
        assert len(_load_dataset({"fasta": str(fasta_file)})) == 4

    def test_max_per_family_applied(self, fasta_file):
        config = {"fasta": {"path": str(fasta_file)}, "data": {"max_per_family": 1}}
        assert len(_load_dataset(config)) == 2


class TestRendering:
    def test_config_summary_shows_reducers_and_baseline(self, two_family_dataset, capsys, tmp_path):
        config = {
            "embedders": [{"name": "composition"}],
            "reducers": [{"name": "pca"}, {"name": "tsne"}],
        }
        print_config_summary(config, two_family_dataset)
        out = capsys.readouterr().out
        assert "pca" in out and "tsne" in out
        assert "Chance baseline" in out

    def test_config_summary_names_curated_families(self, two_family_dataset, capsys):
        print_config_summary(
            {"embedders": [{"name": "composition"}], "reducer": {"name": "pca"}},
            two_family_dataset,
        )
        # PF00072 must render as Response_reg, not as a bare accession.
        assert "Response_reg" in capsys.readouterr().out

    def test_results_table_renders_available_columns(self, capsys):
        results = pd.DataFrame(
            {
                "method": ["composition+pca", "kmer_k3+pca"],
                "trustworthiness": [0.9, 0.7],
                "knn_accuracy_mean": [0.95, 0.80],
                "knn_accuracy_std": [0.01, 0.02],
                "knn_accuracy_highdim_mean": [0.99, 0.85],
                "knn_accuracy_highdim_std": [0.01, 0.03],
                "precision_at_k": [0.97, 0.82],
                "silhouette": [0.5, -0.1],
                "adjusted_rand": [0.9, 0.2],
            }
        )
        print_results_table(results, baseline=0.5)
        out = capsys.readouterr().out
        assert "composition+pca" in out
        assert "Chance baseline" in out
        assert "kNN-full" in out

    def test_results_table_tolerates_a_minimal_frame(self, capsys):
        results = pd.DataFrame(
            {"method": ["a"], "trustworthiness": [0.8], "knn_accuracy_mean": [0.9]}
        )
        print_results_table(results)
        assert "a" in capsys.readouterr().out


class TestParser:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--version"])
        assert "prot2vec" in capsys.readouterr().out

    def test_defaults(self):
        args = _build_parser().parse_args([])
        assert args.config == "configs/default.yaml"
        assert args.log_level == "warning"
        assert args.no_cache is False

    def test_operational_flags(self):
        args = _build_parser().parse_args(["--no-cache", "--no-figures", "--log-level", "debug"])
        assert args.no_cache and args.no_figures
        assert args.log_level == "debug"
