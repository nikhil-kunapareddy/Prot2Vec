"""Tests for the embedders and the shared embedder contract."""

import numpy as np
import pytest
import scipy.sparse

from prot2vec.embedders.composition import AA_ALPHABET, CompositionEmbedder
from prot2vec.embedders.esm import MODEL_CHECKPOINTS, MODEL_DIMS, ESMEmbedder
from prot2vec.embedders.kmer import KmerEmbedder
from prot2vec.embedders.llm import PROVIDER_DEFAULTS, LLMEmbedder, _is_retryable
from tests.conftest import ConstantEmbedder, ShortEmbedder

SEQS = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQ", "ACACACACACACACACACAC"]


class TestCompositionEmbedder:
    def test_output_shape(self):
        assert CompositionEmbedder().fit_transform(SEQS).shape == (len(SEQS), 20)

    def test_rows_sum_to_one(self):
        X = CompositionEmbedder().fit_transform(SEQS)
        np.testing.assert_allclose(X.sum(axis=1), np.ones(len(SEQS)), atol=1e-5)

    def test_non_standard_residues_do_not_dilute_the_row(self):
        # Normalising by raw length would make this row sum to 0.5.
        X = CompositionEmbedder().fit_transform(["AAAA----XXXX"])
        assert X.sum() == pytest.approx(1.0)

    def test_column_order_matches_the_alphabet(self):
        X = CompositionEmbedder().fit_transform(["A" * 10])
        assert X[0, AA_ALPHABET.index("A")] == pytest.approx(1.0)
        assert X[0, AA_ALPHABET.index("C")] == pytest.approx(0.0)

    def test_sequence_with_no_standard_residues_is_a_zero_row(self):
        X = CompositionEmbedder().fit_transform(["----"])
        assert X.shape == (1, 20)
        assert X.sum() == pytest.approx(0.0)

    def test_case_insensitive(self):
        upper = CompositionEmbedder().fit_transform(["ACDE"])
        lower = CompositionEmbedder().fit_transform(["acde"])
        np.testing.assert_array_equal(upper, lower)

    def test_name_and_dim(self):
        assert CompositionEmbedder().name == "composition"
        assert CompositionEmbedder().embedding_dim == 20

    def test_identical_sequences_identical_vectors(self):
        X = CompositionEmbedder().fit_transform([SEQS[0], SEQS[0]])
        np.testing.assert_array_equal(X[0], X[1])

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="No sequences"):
            CompositionEmbedder().fit_transform([])


class TestKmerEmbedder:
    def test_output_shape_and_sparsity(self):
        X = KmerEmbedder(k=3).fit_transform(SEQS)
        assert X.shape[0] == len(SEQS)
        assert scipy.sparse.issparse(X)

    def test_name(self):
        assert KmerEmbedder(k=3).name == "kmer_k3"
        assert KmerEmbedder(k=2).name == "kmer_k2"

    def test_vocabulary_size_after_fit(self):
        embedder = KmerEmbedder(k=3)
        with pytest.raises(RuntimeError, match="after fit_transform"):
            _ = embedder.vocabulary_size
        embedder.fit_transform(SEQS)
        assert embedder.vocabulary_size > 0

    def test_min_df_prunes_rare_kmers(self):
        # These share "ACDEFG" but each carries a unique tail, so raising
        # min_df must discard the unique k-mers and keep the shared ones.
        shared = ["ACDEFGHIKLMNPQRST" + tail for tail in ("WWWWWW", "YYYYYY", "VVVVVV")]
        permissive = KmerEmbedder(k=3, min_df=1)
        strict = KmerEmbedder(k=3, min_df=3)
        permissive.fit_transform(shared)
        strict.fit_transform(shared)
        assert 0 < strict.vocabulary_size < permissive.vocabulary_size

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError, match="k must be"):
            KmerEmbedder(k=0)

    def test_sequences_shorter_than_k_rejected(self):
        with pytest.raises(ValueError, match="at least k="):
            KmerEmbedder(k=5).fit_transform(["AA", "CC"])

    def test_empty_vocabulary_is_explained(self):
        with pytest.raises(ValueError, match="empty vocabulary"):
            KmerEmbedder(k=3, min_df=10).fit_transform(SEQS)

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="No sequences"):
            KmerEmbedder().fit_transform([])


class TestESMEmbedderConfig:
    """Configuration only — running ESM-2 needs the [esm] extra and weights."""

    def test_checkpoints_and_dims_agree(self):
        assert set(MODEL_CHECKPOINTS) == set(MODEL_DIMS)

    def test_name_and_dim(self):
        embedder = ESMEmbedder("esm2_t6_8M")
        assert embedder.name == "esm_esm2_t6_8M"
        assert embedder.embedding_dim == 320
        assert embedder.device is None  # not resolved until load

    @pytest.mark.parametrize(
        "kwargs",
        [{"model_key": "nope"}, {"batch_size": 0}, {"max_len": 0}],
    )
    def test_invalid_config_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ESMEmbedder(**kwargs)

    def test_empty_input_rejected_before_loading_weights(self):
        with pytest.raises(ValueError, match="No sequences"):
            ESMEmbedder("esm2_t6_8M").fit_transform([])


class TestLLMEmbedderConfig:
    """Configuration only — embedding requires an API key and network."""

    def test_default_model_per_provider(self):
        assert LLMEmbedder().model == PROVIDER_DEFAULTS["google"]

    def test_name(self):
        assert LLMEmbedder().name == "llm_google_gemini-embedding-001"

    @pytest.mark.parametrize(
        "kwargs", [{"provider": "openai"}, {"batch_size": 0}, {"output_dim": 0}]
    )
    def test_invalid_config_rejected(self, kwargs):
        with pytest.raises(ValueError):
            LLMEmbedder(**kwargs)

    def test_batching_truncates_and_chunks(self):
        embedder = LLMEmbedder(batch_size=32, max_len=100)
        batches = embedder._batches(["A" * 500] * 70)
        assert [len(b) for b in batches] == [32, 32, 6]
        assert all(len(s) == 100 for b in batches for s in b)

    def test_missing_api_key_is_explained(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            LLMEmbedder()._resolve_api_key()

    def test_blank_api_key_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "   ")
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            LLMEmbedder()._resolve_api_key()

    @pytest.mark.parametrize(
        ("message", "retryable"),
        [
            ("429 RESOURCE_EXHAUSTED", True),
            ("503 Service Unavailable", True),
            ("deadline exceeded", True),
            ("401 API key not valid", False),
            ("400 invalid argument", False),
        ],
    )
    def test_retry_classification(self, message, retryable):
        assert _is_retryable(Exception(message)) is retryable

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="No sequences"):
            LLMEmbedder().fit_transform([])


class TestEmbedderContract:
    def test_params_exclude_runtime_state(self):
        # ConstantEmbedder keeps a public call counter; it must not become part
        # of the configuration identity or caching would break mid-run.
        embedder = ConstantEmbedder(dim=4, tag="x")
        assert set(embedder.params) == {"dim", "tag"}

    def test_cache_key_stable_across_calls(self):
        embedder = KmerEmbedder(k=3)
        before = embedder.cache_key
        embedder.fit_transform(SEQS)
        assert embedder.cache_key == before

    def test_cache_key_separates_settings(self):
        assert KmerEmbedder(k=3).cache_key != KmerEmbedder(k=4).cache_key

    def test_cache_key_ignores_retry_settings(self):
        assert LLMEmbedder(max_retries=3).cache_key == LLMEmbedder(max_retries=9).cache_key

    def test_cache_key_respects_output_dim(self):
        assert LLMEmbedder(output_dim=768).cache_key != LLMEmbedder(output_dim=1536).cache_key

    def test_row_count_violation_is_caught(self):
        with pytest.raises(RuntimeError, match="one row per"):
            ShortEmbedder()._validate_rows(np.zeros((2, 3)), ["a", "b", "c"])

    def test_repr(self):
        assert repr(CompositionEmbedder()) == "CompositionEmbedder(name='composition')"


class TestHuggingFaceEmbedderConfig:
    """Configuration only — running a checkpoint needs [hf] and weights."""

    def test_presets_resolve_to_hub_ids(self):
        from prot2vec.embedders.huggingface import PRESETS, HuggingFaceEmbedder

        assert HuggingFaceEmbedder("protbert").repo_id == PRESETS["protbert"]
        assert HuggingFaceEmbedder("prott5").repo_id.startswith("Rostlab/")

    def test_raw_hub_id_passes_through(self):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        assert HuggingFaceEmbedder("facebook/esm2_t6_8M_UR50D").repo_id == (
            "facebook/esm2_t6_8M_UR50D"
        )

    def test_presets_cover_the_major_families(self):
        from prot2vec.embedders.huggingface import PRESETS

        for expected in ("protbert", "prott5", "ankh_base", "esm2_650m", "dnabert2"):
            assert expected in PRESETS

    def test_name_encodes_pooling_and_layer(self):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        assert HuggingFaceEmbedder("protbert").name == "hf_prot_bert_mean"
        assert HuggingFaceEmbedder("protbert", pooling="cls").name == "hf_prot_bert_cls"
        assert HuggingFaceEmbedder("protbert", layer=-4).name == "hf_prot_bert_mean_L-4"

    def test_rostlab_models_get_space_separated_residues(self):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder("protbert")
        embedder._space_separated = True
        assert embedder._prepare(["MKT"]) == ["M K T"]

    def test_esm_models_are_not_space_separated(self):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder("esm2_35m")
        assert embedder._prepare(["mkt"]) == ["MKT"]

    def test_prostt5_gets_its_direction_token(self):
        # ProstT5 is bilingual over sequence and structure alphabets and has to
        # be told which one it is being given.
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        embedder = HuggingFaceEmbedder("prostt5")
        embedder._space_separated = True
        assert embedder._prepare(["MKT"]) == ["<AA2fold> M K T"]

    def test_trust_remote_code_is_off_by_default(self):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        assert HuggingFaceEmbedder("dnabert2").trust_remote_code is False

    @pytest.mark.parametrize("kwargs", [{"pooling": "sum"}, {"batch_size": 0}, {"max_len": 1}])
    def test_invalid_config_rejected(self, kwargs):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        with pytest.raises(ValueError):
            HuggingFaceEmbedder(**kwargs)

    def test_cache_key_separates_pooling_and_layer(self):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        keys = {
            HuggingFaceEmbedder("protbert").cache_key,
            HuggingFaceEmbedder("protbert", pooling="cls").cache_key,
            HuggingFaceEmbedder("protbert", layer=-4).cache_key,
            HuggingFaceEmbedder("prott5").cache_key,
        }
        assert len(keys) == 4

    def test_device_unresolved_before_loading(self):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        assert HuggingFaceEmbedder("protbert").device is None

    def test_empty_input_rejected_before_downloading_weights(self):
        from prot2vec.embedders.huggingface import HuggingFaceEmbedder

        with pytest.raises(ValueError, match="No sequences"):
            HuggingFaceEmbedder("protbert").fit_transform([])
