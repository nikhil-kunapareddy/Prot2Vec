"""Tests for the interpretable descriptor embedders."""

import numpy as np
import pytest
import scipy.sparse

from prot2vec.embedders.ctd import (
    DISTRIBUTION_QUANTILES,
    FEATURES_PER_PROPERTY,
    PROPERTY_GROUPS,
    CTDEmbedder,
)
from prot2vec.embedders.dipeptide import DipeptideEmbedder
from prot2vec.embedders.onehot import OneHotEmbedder
from prot2vec.embedders.physicochemical import (
    AROMATIC,
    CHARGED,
    SCALES,
    STATISTICS,
    PhysicochemicalEmbedder,
)

SEQS = ["ACDEFGHIKLMNPQRSTVWY", "MKTAYIAKQRQISFVKSHFSRQ", "ACACACACACACACACACAC"]


class TestDipeptideEmbedder:
    def test_shape_and_row_sums(self):
        embedder = DipeptideEmbedder()
        X = embedder.fit_transform(SEQS)
        assert X.shape == (3, 400)
        np.testing.assert_allclose(X.sum(axis=1), 1.0, atol=1e-5)

    def test_fixed_column_order(self):
        embedder = DipeptideEmbedder()
        X = embedder.fit_transform(["AAAACCCC"])
        # 7 pairs: AA AA AA AC CC CC CC
        assert X[0, embedder.pair_names().index("AA")] == pytest.approx(3 / 7)
        assert X[0, embedder.pair_names().index("AC")] == pytest.approx(1 / 7)
        assert X[0, embedder.pair_names().index("CC")] == pytest.approx(3 / 7)

    def test_pair_names_match_dimension(self):
        embedder = DipeptideEmbedder()
        assert len(embedder.pair_names()) == embedder.embedding_dim

    def test_order_matters_unlike_composition(self):
        embedder = DipeptideEmbedder()
        X = embedder.fit_transform(["AAAACCCC", "ACACACAC"])
        assert not np.allclose(X[0], X[1])

    def test_dna_alphabet(self):
        embedder = DipeptideEmbedder("dna")
        X = embedder.fit_transform(["ACGTACGT"])
        assert embedder.name == "dipeptide_dna"
        assert X.shape == (1, 16)
        assert X.sum() == pytest.approx(1.0)

    def test_out_of_alphabet_tokens_do_not_break_the_chain(self):
        # "AC-CA" should yield the pairs of "ACCA", not treat "-" as a residue.
        embedder = DipeptideEmbedder()
        gapped = embedder.fit_transform(["AC-CA"])
        clean = embedder.fit_transform(["ACCA"])
        np.testing.assert_allclose(gapped, clean)

    def test_single_token_is_zero_row(self):
        assert DipeptideEmbedder().fit_transform(["A"]).sum() == 0.0

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="No sequences"):
            DipeptideEmbedder().fit_transform([])


class TestPhysicochemicalEmbedder:
    def test_shape(self):
        embedder = PhysicochemicalEmbedder()
        X = embedder.fit_transform(SEQS)
        assert X.shape == (3, embedder.embedding_dim)
        assert embedder.embedding_dim == len(SCALES) * len(STATISTICS) + 3

    def test_all_scales_cover_the_alphabet(self):
        for name, scale in SCALES.items():
            assert len(scale) == 20, name

    def test_feature_names_match_dimension(self):
        embedder = PhysicochemicalEmbedder()
        assert len(embedder.feature_names()) == embedder.embedding_dim

    def test_hydropathy_ranks_as_chemistry_predicts(self):
        embedder = PhysicochemicalEmbedder()
        index = embedder.feature_names().index("hydropathy_mean")
        X = embedder.fit_transform(["IIIILLLLVVVV", "DDDDEEEEKKKK"])
        assert X[0, index] > X[1, index]

    def test_charge_sign_is_correct(self):
        embedder = PhysicochemicalEmbedder()
        index = embedder.feature_names().index("charge_mean")
        X = embedder.fit_transform(["KKKKRRRR", "DDDDEEEE"])
        assert X[0, index] > 0 > X[1, index]

    def test_aromatic_and_charged_fractions(self):
        embedder = PhysicochemicalEmbedder()
        names = embedder.feature_names()
        X = embedder.fit_transform(["FWYFWY", "DEKRDEKR"])
        assert X[0, names.index("aromatic_fraction")] == pytest.approx(1.0)
        assert X[1, names.index("charged_fraction")] == pytest.approx(1.0)

    def test_aromatic_and_charged_sets(self):
        assert frozenset("FWY") == AROMATIC
        assert frozenset("DEKR") == CHARGED

    def test_no_standard_residues_is_zero_row(self):
        assert PhysicochemicalEmbedder().fit_transform(["----"]).sum() == 0.0

    @pytest.mark.parametrize("alphabet", ["dna", "rna"])
    def test_nucleotides_rejected(self, alphabet):
        with pytest.raises(ValueError, match="amino acids only"):
            PhysicochemicalEmbedder(alphabet)

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="No sequences"):
            PhysicochemicalEmbedder().fit_transform([])


class TestCTDEmbedder:
    def test_shape(self):
        embedder = CTDEmbedder()
        X = embedder.fit_transform(SEQS)
        assert X.shape == (3, 147)
        assert embedder.embedding_dim == len(PROPERTY_GROUPS) * FEATURES_PER_PROPERTY

    def test_every_property_partitions_the_alphabet(self):
        # A residue missing from a property, or duplicated across its groups,
        # would silently skew that property's composition features.
        for name, groups in PROPERTY_GROUPS.items():
            combined = "".join(groups)
            assert len(combined) == 20, name
            assert set(combined) == set("ACDEFGHIKLMNPQRSTVWY"), name

    def test_feature_names_match_dimension(self):
        embedder = CTDEmbedder()
        assert len(embedder.feature_names()) == embedder.embedding_dim

    def test_composition_identical_but_transition_differs(self):
        # The distinguishing property of CTD: same residues, different order.
        embedder = CTDEmbedder()
        names = embedder.feature_names()
        X = embedder.fit_transform(["IIIIIIIIRRRRRRRR", "IRIRIRIRIRIRIRIR"])
        composition = [i for i, n in enumerate(names) if n.startswith("hydrophobicity_C")]
        transition = [i for i, n in enumerate(names) if n.startswith("hydrophobicity_T")]
        np.testing.assert_allclose(X[0, composition], X[1, composition])
        assert not np.allclose(X[0, transition], X[1, transition])

    def test_alternating_sequence_has_maximum_transitions(self):
        embedder = CTDEmbedder()
        names = embedder.feature_names()
        index = names.index("hydrophobicity_T_g1g3")
        X = embedder.fit_transform(["IRIRIRIRIRIRIRIR"])
        assert X[0, index] == pytest.approx(1.0)

    def test_distribution_features_are_fractions(self):
        embedder = CTDEmbedder()
        names = embedder.feature_names()
        distribution = [i for i, n in enumerate(names) if "_D_" in n]
        X = embedder.fit_transform(SEQS)
        assert X[:, distribution].min() >= 0.0
        assert X[:, distribution].max() <= 1.0

    def test_distribution_quantile_count(self):
        assert len(DISTRIBUTION_QUANTILES) == 5

    def test_absent_class_gives_zero_distribution(self):
        embedder = CTDEmbedder()
        names = embedder.feature_names()
        # "KR" is the whole positive-charge group, so the negative group is absent.
        X = embedder.fit_transform(["KRKRKRKR"])
        negative = [i for i, n in enumerate(names) if n.startswith("charge_D_g3")]
        assert X[0, negative].sum() == 0.0

    def test_no_standard_residues_is_zero_row(self):
        assert CTDEmbedder().fit_transform(["---"]).sum() == 0.0

    def test_nucleotides_rejected(self):
        with pytest.raises(ValueError, match="amino acids only"):
            CTDEmbedder("dna")

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="No sequences"):
            CTDEmbedder().fit_transform([])


class TestOneHotEmbedder:
    def test_shape_and_sparsity(self):
        embedder = OneHotEmbedder(max_len=8)
        X = embedder.fit_transform(SEQS)
        assert scipy.sparse.issparse(X)
        assert X.shape == (3, 8 * 20)
        assert embedder.embedding_dim == 160

    def test_one_nonzero_per_encoded_position(self):
        embedder = OneHotEmbedder(max_len=8)
        X = embedder.fit_transform(["ACDE", "ACDEFGHIKLMN", "A"])
        # padded to 4, cropped to 8, and a single residue
        assert list(X.getnnz(axis=1)) == [4, 8, 1]

    def test_name_includes_length(self):
        assert OneHotEmbedder(max_len=64).name == "onehot_L64"
        assert OneHotEmbedder(max_len=64, alphabet="dna").name == "onehot_dna_L64"

    @pytest.mark.parametrize(
        ("truncate", "expected_columns"),
        [("start", [0, 21]), ("end", [2, 23]), ("center", [1, 22])],
    )
    def test_truncation_modes(self, truncate, expected_columns):
        embedder = OneHotEmbedder(max_len=2, truncate=truncate)
        assert sorted(embedder.fit_transform(["ACDE"]).nonzero()[1]) == expected_columns

    def test_dna_alphabet(self):
        embedder = OneHotEmbedder(max_len=6, alphabet="dna")
        assert embedder.fit_transform(["ACGTACGT"]).shape == (1, 24)

    def test_out_of_alphabet_tokens_do_not_consume_positions(self):
        embedder = OneHotEmbedder(max_len=4)
        gapped = embedder.fit_transform(["A-C-D-E"])
        clean = embedder.fit_transform(["ACDE"])
        assert (gapped != clean).nnz == 0

    @pytest.mark.parametrize("kwargs", [{"max_len": 0}, {"truncate": "middle"}])
    def test_invalid_config_rejected(self, kwargs):
        with pytest.raises(ValueError):
            OneHotEmbedder(**kwargs)

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="No sequences"):
            OneHotEmbedder().fit_transform([])


class TestDescriptorContract:
    @pytest.mark.parametrize(
        "embedder",
        [
            DipeptideEmbedder(),
            PhysicochemicalEmbedder(),
            CTDEmbedder(),
            OneHotEmbedder(max_len=16),
        ],
    )
    def test_one_row_per_sequence(self, embedder):
        assert embedder.fit_transform(SEQS).shape[0] == len(SEQS)

    @pytest.mark.parametrize(
        "embedder",
        [
            DipeptideEmbedder(),
            PhysicochemicalEmbedder(),
            CTDEmbedder(),
            OneHotEmbedder(max_len=16),
        ],
    )
    def test_deterministic(self, embedder):
        first = embedder.fit_transform(SEQS)
        second = embedder.fit_transform(SEQS)
        dense_first = first.toarray() if scipy.sparse.issparse(first) else first
        dense_second = second.toarray() if scipy.sparse.issparse(second) else second
        np.testing.assert_array_equal(dense_first, dense_second)

    def test_cache_keys_are_distinct_across_descriptors(self):
        keys = {
            DipeptideEmbedder().cache_key,
            DipeptideEmbedder("dna").cache_key,
            PhysicochemicalEmbedder().cache_key,
            CTDEmbedder().cache_key,
            OneHotEmbedder(max_len=16).cache_key,
            OneHotEmbedder(max_len=32).cache_key,
        }
        assert len(keys) == 6
