"""Shared fixtures."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from prot2vec.data.dataset import ProteinDataset
from prot2vec.embedders.base import SequenceEmbedder


def make_record(seq: str, rec_id: str = "seq1", description: str | None = None):
    """Build a stand-in for a Bio.SeqRecord."""
    record = MagicMock()
    record.seq = seq
    record.id = rec_id
    record.description = description if description is not None else rec_id
    return record


@pytest.fixture
def two_family_dataset() -> ProteinDataset:
    """A small two-family dataset with compositionally distinct groups."""
    rng = np.random.default_rng(0)
    sequences, labels, ids = [], [], []
    for family, alphabet in (("PF00069", "AGSTVIL"), ("PF00072", "DEKRNQH")):
        for i in range(12):
            sequences.append("".join(rng.choice(list(alphabet)) for _ in range(80)))
            labels.append(family)
            ids.append(f"{family.lower()}_{i:02d}")
    return ProteinDataset(sequences=sequences, labels=labels, ids=ids, source="fixture")


@pytest.fixture
def fasta_file(tmp_path):
    """A FASTA file with two groups and one too-short record."""
    path = tmp_path / "proteins.fasta"
    path.write_text(
        ">PF00069 kinase_001\n" + "ACDEFGHIKL" * 8 + "\n"
        ">PF00069 kinase_002\n" + "MKTAYIAKQR" * 8 + "\n"
        ">PF00072 reg_001\n" + "GHIKLMNPQR" * 8 + "\n"
        ">PF00072 reg_002\n" + "DEKRNQHSTV" * 8 + "\n"
        ">PF00072 fragment\nACDE\n"
    )
    return path


class ConstantEmbedder(SequenceEmbedder):
    """Deterministic embedder for pipeline tests: one row per sequence, no deps."""

    def __init__(self, dim: int = 4, tag: str = "constant") -> None:
        self.dim = dim
        self.tag = tag
        self.call_count = 0

    @property
    def name(self) -> str:
        return self.tag

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        self.call_count += 1
        # Encode length and a couple of residue counts so that the output has
        # real structure rather than being constant across sequences.
        rows = [[len(s), s.count("A"), s.count("D"), s.count("K")][: self.dim] for s in sequences]
        return np.asarray(rows, dtype=np.float32)


class ShortEmbedder(SequenceEmbedder):
    """Violates the row-count contract, to prove the pipeline catches it."""

    @property
    def name(self) -> str:
        return "short"

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        return np.zeros((len(sequences) - 1, 3), dtype=np.float32)
