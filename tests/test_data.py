"""Tests for data layer."""
from unittest.mock import MagicMock

from src.data.dataset import ProteinDataset, STANDARD_AAS


def _make_mock_record(seq_str: str, rec_id: str = "seq1"):
    rec = MagicMock()
    rec.seq = seq_str
    rec.id = rec_id
    return rec


class TestProteinDataset:
    def test_from_pfam_records_basic(self):
        records = {
            "PF00001": [_make_mock_record("ACDEFGHIKLMNPQRSTVWY" * 4, "s1")],
            "PF00002": [_make_mock_record("ACDEFGHIKLMNPQRSTVWY" * 4, "s2")],
        }
        ds = ProteinDataset.from_pfam_records(records)
        assert len(ds) == 2
        assert set(ds.labels) == {"PF00001", "PF00002"}

    def test_short_sequences_filtered(self):
        records = {
            "PF00001": [
                _make_mock_record("ACDE", "short"),  # 4 AA — below default min_length=50
                _make_mock_record("A" * 60, "long"),
            ]
        }
        ds = ProteinDataset.from_pfam_records(records, min_length=50)
        assert len(ds) == 1
        assert ds.ids == ["long"]

    def test_non_standard_aas_stripped(self):
        records = {"PF00001": [_make_mock_record("ACDE-XB" * 10, "s1")]}
        ds = ProteinDataset.from_pfam_records(records, min_length=1)
        assert all(c in STANDARD_AAS for c in ds.sequences[0])

    def test_families_property(self):
        records = {
            "PF00003": [_make_mock_record("A" * 60, "s1")],
            "PF00001": [_make_mock_record("A" * 60, "s2")],
        }
        ds = ProteinDataset.from_pfam_records(records)
        assert ds.families == ["PF00001", "PF00003"]  # sorted
