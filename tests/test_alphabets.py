"""Tests for sequence alphabets."""

import pytest

from prot2vec.data.alphabets import (
    ALPHABETS,
    DNA,
    PROTEIN,
    RNA,
    Alphabet,
    clean_sequence,
    get_alphabet,
    require_protein,
)


class TestAlphabet:
    def test_sizes(self):
        assert len(PROTEIN) == 20
        assert len(DNA) == 4
        assert len(RNA) == 4

    def test_registry_keys_match_names(self):
        for name, alphabet in ALPHABETS.items():
            assert name == alphabet.name

    def test_membership(self):
        assert "A" in PROTEIN
        assert "U" not in DNA
        assert "U" in RNA

    def test_index_is_positional(self):
        assert DNA.index == {"A": 0, "C": 1, "G": 2, "T": 3}

    def test_token_set(self):
        assert DNA.token_set == frozenset("ACGT")

    def test_str(self):
        assert str(PROTEIN) == "protein (20 tokens)"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            PROTEIN.tokens = "AC"

    def test_protein_excludes_ambiguity_and_noncanonical(self):
        for token in "BZJXUO":
            assert token not in PROTEIN


class TestGetAlphabet:
    @pytest.mark.parametrize("name", ["protein", "dna", "rna"])
    def test_by_name(self, name):
        assert get_alphabet(name).name == name

    def test_passes_instances_through(self):
        assert get_alphabet(DNA) is DNA

    def test_default_is_protein(self):
        assert get_alphabet().name == "protein"

    def test_unknown_lists_options(self):
        with pytest.raises(ValueError, match="Choose from"):
            get_alphabet("peptide")


class TestCleanSequence:
    def test_protein_strips_gaps_and_ambiguity(self):
        assert clean_sequence("MKT-AY..iXBZUO") == "MKTAYI"

    def test_uppercases(self):
        assert clean_sequence("acgt", alphabet="dna") == "ACGT"

    def test_dna_drops_n(self):
        assert clean_sequence("ACGTNNNACGT", alphabet="dna") == "ACGTACGT"

    def test_rna_keeps_u_drops_t(self):
        assert clean_sequence("ACGUT", alphabet="rna") == "ACGU"

    def test_empty(self):
        assert clean_sequence("----") == ""

    def test_the_hazard_this_parameter_exists_for(self):
        # A, C, G, T and N are all valid amino acid codes, so DNA cleaned
        # against the protein alphabet is not rejected -- it passes through
        # untouched and silently means nothing. This is why the alphabet must
        # be stated rather than guessed.
        dna = "ACGTACGTNNN"
        assert clean_sequence(dna, alphabet="protein") == dna
        assert clean_sequence(dna, alphabet="dna") == "ACGTACGT"


class TestRequireProtein:
    def test_accepts_protein(self):
        require_protein(PROTEIN, "CTD")

    @pytest.mark.parametrize("alphabet", [DNA, RNA])
    def test_rejects_nucleotides_with_an_alternative(self, alphabet):
        with pytest.raises(ValueError, match="amino acids only"):
            require_protein(alphabet, "CTD")

    def test_message_names_the_descriptor(self):
        with pytest.raises(ValueError, match="PhysicochemicalEmbedder"):
            require_protein(DNA, "PhysicochemicalEmbedder")


class TestCustomAlphabet:
    def test_user_defined_alphabet_works(self):
        # Reduced alphabets (hydrophobic/polar/charged classes) are a common
        # protein-engineering idiom, so the type has to be usable directly.
        reduced = Alphabet("reduced", "HPC", "hydrophobic/polar/charged classes")
        assert clean_sequence("HPCXHP", alphabet=reduced) == "HPCHP"
        assert len(reduced) == 3
