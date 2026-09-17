"""Tests for the data layer: dataset construction, cleaning and FASTA I/O."""

import pytest

from prot2vec.data.dataset import STANDARD_AAS, ProteinDataset, clean_sequence
from tests.conftest import make_record


class TestCleanSequence:
    def test_strips_gaps_and_ambiguity_codes(self):
        assert clean_sequence("MKT-AY..iXBZUO") == "MKTAYI"

    def test_uppercases(self):
        assert clean_sequence("mkta") == "MKTA"

    def test_empty_input(self):
        assert clean_sequence("----") == ""


class TestFromPfamRecords:
    def test_basic(self):
        records = {
            "PF00001": [make_record("ACDEFGHIKLMNPQRSTVWY" * 4, "s1")],
            "PF00002": [make_record("ACDEFGHIKLMNPQRSTVWY" * 4, "s2")],
        }
        ds = ProteinDataset.from_pfam_records(records)
        assert len(ds) == 2
        assert set(ds.labels) == {"PF00001", "PF00002"}

    def test_short_sequences_filtered(self):
        records = {
            "PF00001": [
                make_record("ACDE", "short"),
                make_record("A" * 60, "long"),
            ]
        }
        ds = ProteinDataset.from_pfam_records(records, min_length=50)
        assert len(ds) == 1
        assert ds.ids == ["long"]

    def test_non_standard_aas_stripped(self):
        records = {"PF00001": [make_record("ACDE-XB" * 10, "s1")]}
        ds = ProteinDataset.from_pfam_records(records, min_length=1)
        assert all(c in STANDARD_AAS for c in ds.sequences[0])

    def test_families_sorted(self):
        records = {
            "PF00003": [make_record("A" * 60, "s1")],
            "PF00001": [make_record("A" * 60, "s2")],
        }
        assert ProteinDataset.from_pfam_records(records).families == ["PF00001", "PF00003"]

    def test_max_per_family_caps_and_is_reproducible(self):
        records = {"PF00001": [make_record("A" * 60, f"s{i}") for i in range(20)]}
        first = ProteinDataset.from_pfam_records(records, max_per_family=5, random_state=7)
        second = ProteinDataset.from_pfam_records(records, max_per_family=5, random_state=7)
        assert len(first) == 5
        assert first.ids == second.ids

    def test_max_per_family_different_seeds_differ(self):
        records = {"PF00001": [make_record("A" * 60, f"s{i}") for i in range(50)]}
        a = ProteinDataset.from_pfam_records(records, max_per_family=5, random_state=1)
        b = ProteinDataset.from_pfam_records(records, max_per_family=5, random_state=2)
        assert a.ids != b.ids

    def test_duplicate_ids_are_disambiguated(self):
        records = {"PF00001": [make_record("A" * 60, "dup") for _ in range(3)]}
        ds = ProteinDataset.from_pfam_records(records)
        assert ds.ids == ["dup", "dup#2", "dup#3"]


class TestFromFasta:
    def test_reads_and_filters(self, fasta_file):
        ds = ProteinDataset.from_fasta(fasta_file)
        assert len(ds) == 4  # the 4-residue fragment is dropped
        assert ds.families == ["PF00069", "PF00072"]
        assert ds.source == "fasta:proteins.fasta"

    def test_first_token_label_keeps_informative_id(self, fasta_file):
        # record.id is the label under this strategy, so the id must come from
        # the rest of the header or every row would share one identifier.
        ds = ProteinDataset.from_fasta(fasta_file)
        assert sorted(ds.ids) == ["kinase_001", "kinase_002", "reg_001", "reg_002"]

    def test_ids_are_unique(self, tmp_path):
        path = tmp_path / "dup.fasta"
        path.write_text(">GRP same\n" + "A" * 60 + "\n>GRP same\n" + "C" * 60 + "\n")
        assert ProteinDataset.from_fasta(path).ids == ["same", "same#2"]

    @pytest.mark.parametrize(
        ("label_from", "expected"),
        [
            ("first_token", ["PF00069", "PF00072"]),
            ("last_token", ["fragment", "kinase_001", "kinase_002", "reg_001", "reg_002"]),
            ("none", ["all"]),
            ("pipe:0", None),
        ],
    )
    def test_label_strategies(self, fasta_file, label_from, expected):
        ds = ProteinDataset.from_fasta(fasta_file, label_from=label_from, min_length=1)
        if expected is not None:
            assert ds.families == expected
        else:
            assert len(ds.families) >= 1

    def test_pipe_label(self, tmp_path):
        path = tmp_path / "pipe.fasta"
        path.write_text(">sp|P12345|KIN_HUMAN\n" + "A" * 60 + "\n")
        ds = ProteinDataset.from_fasta(path, label_from="pipe:0", min_length=1)
        assert ds.families == ["sp"]

    def test_pipe_index_out_of_range(self, tmp_path):
        path = tmp_path / "pipe.fasta"
        path.write_text(">onlyone\n" + "A" * 60 + "\n")
        with pytest.raises(ValueError, match="out of range"):
            ProteinDataset.from_fasta(path, label_from="pipe:9")

    def test_unknown_label_strategy(self, fasta_file):
        with pytest.raises(ValueError, match="Unknown label_from"):
            ProteinDataset.from_fasta(fasta_file, label_from="nonsense")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProteinDataset.from_fasta(tmp_path / "nope.fasta")

    def test_everything_filtered_out(self, fasta_file):
        with pytest.raises(ValueError, match="No usable sequences"):
            ProteinDataset.from_fasta(fasta_file, min_length=100_000)

    def test_gzip_roundtrip(self, tmp_path, two_family_dataset):
        import gzip

        plain = two_family_dataset.to_fasta(tmp_path / "out.fasta")
        gz = tmp_path / "out.fasta.gz"
        gz.write_bytes(gzip.compress(plain.read_bytes()))
        assert ProteinDataset.from_fasta(gz).family_counts == two_family_dataset.family_counts


class TestDatasetIntrospection:
    def test_length_and_repr(self, two_family_dataset):
        assert len(two_family_dataset) == 24
        assert "n=24" in repr(two_family_dataset)

    def test_family_counts_sorted(self, two_family_dataset):
        assert list(two_family_dataset.family_counts) == ["PF00069", "PF00072"]

    def test_summary(self, two_family_dataset):
        summary = two_family_dataset.summary()
        assert summary["n_sequences"] == 24
        assert summary["n_families"] == 2
        assert summary["majority_class_fraction"] == pytest.approx(0.5)
        assert summary["length_min"] == summary["length_max"] == 80

    def test_mismatched_lists_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            ProteinDataset(sequences=["A" * 60], labels=[], ids=[])

    def test_to_fasta_roundtrip(self, tmp_path, two_family_dataset):
        # An export must survive a trip through the default loader.
        path = two_family_dataset.to_fasta(tmp_path / "nested" / "out.fasta")
        assert path.exists()
        reloaded = ProteinDataset.from_fasta(path)
        assert reloaded.sequences == two_family_dataset.sequences
        assert reloaded.labels == two_family_dataset.labels
        assert reloaded.ids == two_family_dataset.ids
