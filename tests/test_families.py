"""Tests for the curated Pfam family catalogue."""

import pytest

from prot2vec.data.families import (
    CURATED_FAMILIES,
    PFAM_ACCESSION_RE,
    PfamFamily,
    describe,
    validate_accessions,
)


class TestCatalogue:
    def test_keys_match_accessions(self):
        for accession, family in CURATED_FAMILIES.items():
            assert accession == family.accession

    def test_all_accessions_well_formed(self):
        for accession in CURATED_FAMILIES:
            assert PFAM_ACCESSION_RE.match(accession)

    def test_known_identities(self):
        # PF00072 was previously mislabelled as a GPCR across the configs; it
        # is the two-component response regulator receiver domain. PF00001 is
        # the rhodopsin-family GPCR.
        assert CURATED_FAMILIES["PF00072"].identifier == "Response_reg"
        assert CURATED_FAMILIES["PF00001"].identifier == "7tm_1"
        assert CURATED_FAMILIES["PF00069"].identifier == "Pkinase"

    def test_entries_are_immutable(self):
        with pytest.raises(AttributeError):
            CURATED_FAMILIES["PF00069"].identifier = "nope"

    def test_str(self):
        assert str(PfamFamily("PF00069", "Pkinase", "x")) == "PF00069 (Pkinase)"


class TestDescribe:
    def test_known(self):
        assert describe("PF00069") == "PF00069 (Pkinase)"

    def test_unknown_falls_back_to_accession(self):
        assert describe("PF99999") == "PF99999"


class TestValidateAccessions:
    def test_accepts_valid(self):
        validate_accessions(["PF00069", "PF12345"])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            validate_accessions([])

    @pytest.mark.parametrize("bad", ["PF69", "pf00069", "P00069", "PF000691", "00069", ""])
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError, match="Malformed"):
            validate_accessions([bad])

    def test_rejects_duplicates(self):
        with pytest.raises(ValueError, match="Duplicate"):
            validate_accessions(["PF00069", "PF00069"])
