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
            ({"name": "dipeptide"}, "dipeptide"),
            ({"name": "physicochemical"}, "physicochemical"),
            ({"name": "ctd"}, "ctd"),
            ({"name": "kmer", "k": 4}, "kmer_k4"),
            ({"name": "onehot", "max_len": 64}, "onehot_L64"),
            ({"name": "esm2", "model_key": "esm2_t6_8M"}, "esm_esm2_t6_8M"),
            ({"name": "hf", "model": "protbert"}, "hf_prot_bert_mean"),
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
        with pytest.raises(ValueError, match="composition, dipeptide"):
            _build_embedder({"name": "prot-bert"})

    @pytest.mark.parametrize(
        "name",
        [
            "pca",
            "svd",
            "nmf",
            "random_projection",
            "umap",
            "tsne",
            "isomap",
            "mds",
            "spectral",
            "lle",
            "kernel_pca",
            "phate",
            "pacmap",
        ],
    )
    def test_build_reducer(self, name):
        assert _build_reducer({"name": name}).name == name

    def test_reducer_params_forwarded(self):
        reducer = _build_reducer({"name": "umap", "n_neighbors": 7, "metric": "euclidean"})
        assert reducer.params["n_neighbors"] == 7
        assert reducer.params["metric"] == "euclidean"

    def test_unknown_reducer_lists_the_options(self):
        with pytest.raises(ValueError, match="pca, svd, nmf"):
            _build_reducer({"name": "trimap"})


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
        assert "kNN-hi" in out

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


class TestAlphabetWiring:
    def test_dna_alphabet_reaches_the_dataset(self, tmp_path):
        path = tmp_path / "dna.fasta"
        path.write_text(">G1 s1\n" + "ACGT" * 40 + "\n>G2 s2\n" + "TTGGCCAA" * 20 + "\n")
        dataset = _load_dataset(
            {"fasta": {"path": str(path)}, "data": {"alphabet": "dna", "min_seq_length": 50}}
        )
        assert dataset.alphabet == "dna"

    def test_alphabet_reaches_alphabet_aware_embedders(self):
        assert _build_embedder({"name": "dipeptide"}, alphabet="dna").name == "dipeptide_dna"
        assert _build_embedder({"name": "onehot"}, alphabet="dna").name.startswith("onehot_dna")

    def test_per_embedder_alphabet_overrides_the_dataset(self):
        embedder = _build_embedder({"name": "dipeptide", "alphabet": "rna"}, alphabet="dna")
        assert embedder.alphabet == "rna"

    @pytest.mark.parametrize("name", ["physicochemical", "ctd"])
    def test_protein_only_embedders_reject_nucleotide_runs(self, name):
        # Scoring DNA with amino acid scales would produce a number that looks
        # like a result and means nothing, so refuse -- whether the alphabet
        # came from the dataset or from the embedder's own block.
        with pytest.raises(ValueError, match="amino acids only"):
            _build_embedder({"name": name}, alphabet="dna")
        with pytest.raises(ValueError, match="amino acids only"):
            _build_embedder({"name": name, "alphabet": "rna"})

    def test_unknown_alphabet_rejected(self, tmp_path):
        path = tmp_path / "x.fasta"
        path.write_text(">G1 s1\n" + "ACGT" * 40 + "\n")
        with pytest.raises(ValueError, match="Unknown alphabet"):
            _load_dataset({"fasta": {"path": str(path)}, "data": {"alphabet": "peptide"}})


class TestBuilderOptions:
    def test_onehot_options_forwarded(self):
        embedder = _build_embedder({"name": "onehot", "max_len": 32, "truncate": "start"})
        assert embedder.max_len == 32
        assert embedder.truncate == "start"

    def test_huggingface_options_forwarded(self):
        embedder = _build_embedder(
            {
                "name": "hf",
                "model": "esm2_35m",
                "pooling": "cls",
                "layer": -3,
                "trust_remote_code": True,
            }
        )
        assert embedder.pooling == "cls"
        assert embedder.layer == -3
        assert embedder.trust_remote_code is True

    def test_huggingface_alias(self):
        assert _build_embedder({"name": "huggingface"}).name.startswith("hf_")

    def test_kernel_pca_options_forwarded(self):
        reducer = _build_reducer({"name": "kernel_pca", "kernel": "rbf", "gamma": 0.5})
        assert reducer.params["kernel"] == "rbf"
        assert reducer.params["gamma"] == 0.5

    def test_n_components_forwarded_to_every_reducer(self):
        for name in ("pca", "svd", "nmf", "umap", "tsne", "isomap", "mds", "kernel_pca"):
            reducer = _build_reducer({"name": name, "n_components": 3})
            assert reducer.params["n_components"] == 3


class TestMetricGroupConfig:
    def test_groups_are_resolved_in_canonical_order(self):
        from prot2vec.evaluation import resolve_groups

        assert resolve_groups(["retrieval", "projection"]) == ("projection", "retrieval")

    def test_unknown_group_is_rejected(self):
        # main() runs this before embedding anything, so a typo in the config
        # fails in milliseconds rather than after an ESM-2 pass.
        from prot2vec.evaluation import resolve_groups

        with pytest.raises(ValueError, match="Unknown metric group"):
            resolve_groups(["projection", "nonsense"])

    def test_config_groups_key_survives_a_round_trip(self, tmp_path, fasta_file):
        path = write_config(
            tmp_path,
            fasta={"path": str(fasta_file)},
            metrics={"groups": ["retrieval"]},
        )
        assert _load_config(path)["metrics"]["groups"] == ["retrieval"]


class TestSkippedPairReporting:
    def test_nothing_printed_when_nothing_skipped(self, capsys):
        from prot2vec.cli import print_skipped_pairs

        print_skipped_pairs([])
        assert capsys.readouterr().out == ""

    def test_each_omission_is_named(self, capsys):
        from prot2vec.cli import print_skipped_pairs

        print_skipped_pairs([("physicochemical+nmf", "NMF requires non-negative input")])
        out = capsys.readouterr().out
        assert "1 pair(s) skipped" in out
        assert "physicochemical+nmf" in out
        assert "non-negative" in out

    def test_long_reasons_are_trimmed(self, capsys):
        from prot2vec.cli import print_skipped_pairs

        print_skipped_pairs([("a+b", "x" * 300)])
        for line in capsys.readouterr().out.splitlines():
            assert len(line) < 140


class TestConfoundWarnings:
    def test_length_confound_is_surfaced(self, capsys):
        from prot2vec.cli import print_results_table

        results = pd.DataFrame(
            {
                "embedder": ["esm"],
                "method": ["esm+umap"],
                "trustworthiness": [0.9],
                "knn_accuracy_highdim_mean": [0.92],
                "knn_accuracy_highdim_std": [0.01],
                "length_only_knn_accuracy": [0.91],
            }
        )
        print_results_table(results)
        assert "differ mainly in length" in capsys.readouterr().out

    def test_no_warning_when_the_embedding_clearly_beats_length(self, capsys):
        from prot2vec.cli import print_results_table

        results = pd.DataFrame(
            {
                "embedder": ["esm"],
                "method": ["esm+umap"],
                "trustworthiness": [0.9],
                "knn_accuracy_highdim_mean": [0.98],
                "knn_accuracy_highdim_std": [0.01],
                "length_only_knn_accuracy": [0.55],
            }
        )
        print_results_table(results)
        assert "differ mainly in length" not in capsys.readouterr().out

    def test_composition_warning_exempts_composition_embedders(self, capsys):
        # Correlating with composition is tautological for these two.
        from prot2vec.cli import print_results_table

        results = pd.DataFrame(
            {
                "embedder": ["composition"],
                "method": ["composition+pca"],
                "trustworthiness": [0.9],
                "knn_accuracy_highdim_mean": [0.98],
                "knn_accuracy_highdim_std": [0.01],
                "composition_distance_rho": [1.0],
            }
        )
        print_results_table(results)
        assert "re-encoding residue frequencies" not in capsys.readouterr().out

    def test_composition_warning_fires_for_learned_embedders(self, capsys):
        from prot2vec.cli import print_results_table

        results = pd.DataFrame(
            {
                "embedder": ["esm_esm2_t12_35M"],
                "method": ["esm_esm2_t12_35M+umap"],
                "trustworthiness": [0.9],
                "knn_accuracy_highdim_mean": [0.98],
                "knn_accuracy_highdim_std": [0.01],
                "composition_distance_rho": [0.97],
            }
        )
        print_results_table(results)
        assert "re-encoding residue frequencies" in capsys.readouterr().out
