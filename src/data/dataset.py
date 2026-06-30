"""ProteinDataset — clean sequence store with family labels."""
from __future__ import annotations

from dataclasses import dataclass

from Bio.SeqRecord import SeqRecord

STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class ProteinDataset:
    sequences: list[str]  # cleaned sequences (gaps removed, standard AAs only)
    labels: list[str]     # family accession per sequence
    ids: list[str]        # original sequence identifier

    @classmethod
    def from_pfam_records(
        cls,
        records_by_family: dict[str, list[SeqRecord]],
        min_length: int = 50,
    ) -> "ProteinDataset":
        """Build a dataset from a records_by_family dict, filtering short sequences."""
        sequences, labels, ids = [], [], []
        for family, recs in records_by_family.items():
            for rec in recs:
                seq = "".join(c for c in str(rec.seq) if c in STANDARD_AAS)
                if len(seq) >= min_length:
                    sequences.append(seq)
                    labels.append(family)
                    ids.append(rec.id)
        return cls(sequences=sequences, labels=labels, ids=ids)

    def __len__(self) -> int:
        return len(self.sequences)

    def __repr__(self) -> str:
        fam_counts = {f: self.labels.count(f) for f in self.families}
        return f"ProteinDataset(n={len(self)}, families={fam_counts})"

    @property
    def families(self) -> list[str]:
        return sorted(set(self.labels))

    @property
    def family_counts(self) -> dict[str, int]:
        return {f: self.labels.count(f) for f in self.families}
