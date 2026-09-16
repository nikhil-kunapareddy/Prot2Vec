"""The sequence container every embedder and metric agrees on."""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)

#: The 20 standard amino acids. Everything else -- alignment gaps (``-``,
#: ``.``), the ambiguity codes ``B``/``Z``/``J``/``X``, and the non-canonical
#: residues ``U`` (selenocysteine) and ``O`` (pyrrolysine) -- is removed during
#: loading, because embedders disagree on how to tokenise them and the choice
#: would otherwise silently differ between methods being compared.
STANDARD_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence: str) -> str:
    """Strip everything that is not one of the 20 standard amino acids.

    Parameters
    ----------
    sequence
        Raw sequence, possibly gapped and in mixed case.

    Returns
    -------
    str
        Uppercase sequence containing only standard residues.

    Examples
    --------
    >>> clean_sequence("mkt-AY..iX")
    'MKTAYI'
    """
    return "".join(c for c in sequence.upper() if c in STANDARD_AAS)


@dataclass
class ProteinDataset:
    """Cleaned protein sequences with a family label for each one.

    The three lists are positionally aligned, and every embedder in Prot2Vec
    promises one output row per sequence in the same order, so ``sequences[i]``,
    ``labels[i]``, ``ids[i]`` and embedding row ``i`` all describe the same
    protein. Construction goes through :meth:`from_pfam_records` or
    :meth:`from_fasta` rather than the raw constructor so that cleaning and
    length filtering are applied consistently.

    Attributes
    ----------
    sequences
        Cleaned sequences, standard residues only.
    labels
        Group label per sequence — a Pfam accession, or whatever grouping the
        FASTA headers encode.
    ids
        Original sequence identifier, preserved so results can be traced back
        to the input.
    source
        Human-readable description of where the data came from, recorded in run
        manifests for provenance.
    """

    sequences: list[str]
    labels: list[str]
    ids: list[str]
    source: str = field(default="unknown")

    def __post_init__(self) -> None:
        """Validate that the parallel lists line up.

        Raises
        ------
        ValueError
            If the three lists have different lengths.
        """
        lengths = {len(self.sequences), len(self.labels), len(self.ids)}
        if len(lengths) > 1:
            raise ValueError(
                "sequences, labels and ids must be the same length, got "
                f"{len(self.sequences)}, {len(self.labels)}, {len(self.ids)}."
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pfam_records(
        cls,
        records_by_family: dict[str, list[SeqRecord]],
        min_length: int = 50,
        max_per_family: int | None = None,
        random_state: int = 0,
    ) -> ProteinDataset:
        """Build a dataset from parsed Pfam seed alignments.

        Parameters
        ----------
        records_by_family
            Mapping of accession to seed records, as returned by
            :func:`~prot2vec.data.pfam.parse_pfam_families`.
        min_length
            Discard sequences shorter than this after cleaning. Very short
            fragments carry almost no compositional signal and make k-mer
            features degenerate.
        max_per_family
            Cap each family at this many sequences. Pfam seed alignments vary
            in size by more than an order of magnitude, and an unbalanced
            benchmark rewards a method for simply predicting the largest
            family — capping makes k-NN accuracy interpretable against a known
            chance baseline.
        random_state
            Seed for the subsampling, so a capped dataset is reproducible.

        Returns
        -------
        ProteinDataset
            The assembled dataset.
        """
        sequences: list[str] = []
        labels: list[str] = []
        ids: list[str] = []
        rng = random.Random(random_state)

        for family, records in records_by_family.items():
            kept: list[tuple[str, str]] = []
            for record in records:
                cleaned = clean_sequence(str(record.seq))
                if len(cleaned) >= min_length:
                    kept.append((cleaned, str(record.id)))

            dropped = len(records) - len(kept)
            if dropped:
                logger.debug(
                    "%s: dropped %d/%d sequences shorter than %d residues",
                    family,
                    dropped,
                    len(records),
                    min_length,
                )

            if max_per_family is not None and len(kept) > max_per_family:
                logger.info("%s: subsampling %d of %d sequences", family, max_per_family, len(kept))
                kept = rng.sample(kept, max_per_family)

            for cleaned, record_id in kept:
                sequences.append(cleaned)
                labels.append(family)
                ids.append(record_id)

        return cls(
            sequences=sequences,
            labels=labels,
            ids=_unique_ids(ids),
            source=f"pfam:{','.join(sorted(records_by_family))}",
        )

    @classmethod
    def from_fasta(
        cls,
        path: str | Path,
        label_from: str = "first_token",
        min_length: int = 50,
        max_per_family: int | None = None,
        random_state: int = 0,
    ) -> ProteinDataset:
        """Load sequences from a FASTA file, deriving labels from the headers.

        Lets Prot2Vec benchmark embeddings on your own sequences rather than
        only on Pfam — any grouping works, as long as the header encodes it.

        Parameters
        ----------
        path
            FASTA file. Plain text or gzip, as decided by the file extension.
        label_from
            How to read a group label out of each header:

            ``"first_token"``
                Text before the first whitespace, e.g. ``>PF00069 sp|...`` →
                ``PF00069``. The default.
            ``"last_token"``
                Text after the last whitespace, for headers that append the
                class.
            ``"description"``
                The whole description after the identifier.
            ``"pipe:N"``
                Field ``N`` of a pipe-delimited header, zero-indexed — so
                ``"pipe:2"`` reads ``sp`` from ``>sp|P12345|KINASE_HUMAN``.
            ``"none"``
                One single group; useful for unlabelled exploration where you
                only want the projection and figures.
        min_length
            Discard sequences shorter than this after cleaning.
        max_per_family
            Cap each group at this many sequences.
        random_state
            Seed for the subsampling.

        Returns
        -------
        ProteinDataset
            The assembled dataset.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If ``label_from`` is not recognised, or no sequence survives
            cleaning and length filtering.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {path}")

        grouped: dict[str, list[tuple[str, str]]] = {}
        total = 0
        for record in _iter_fasta(path):
            total += 1
            cleaned = clean_sequence(str(record.seq))
            if len(cleaned) < min_length:
                continue
            label = _label_from_record(record, label_from)
            grouped.setdefault(label, []).append((cleaned, _record_id(record, label_from)))

        if not grouped:
            raise ValueError(
                f"No usable sequences in {path}: read {total} record(s), none of "
                f"which had at least {min_length} standard residues. Lower "
                "min_seq_length, or check the file is protein rather than nucleotide."
            )

        rng = random.Random(random_state)
        sequences: list[str] = []
        labels: list[str] = []
        ids: list[str] = []
        for label in sorted(grouped):
            kept = grouped[label]
            if max_per_family is not None and len(kept) > max_per_family:
                kept = rng.sample(kept, max_per_family)
            for cleaned, record_id in kept:
                sequences.append(cleaned)
                labels.append(label)
                ids.append(record_id)

        logger.info(
            "Loaded %d/%d sequences from %s across %d group(s)",
            len(sequences),
            total,
            path.name,
            len(grouped),
        )
        return cls(
            sequences=sequences,
            labels=labels,
            ids=_unique_ids(ids),
            source=f"fasta:{path.name}",
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of sequences."""
        return len(self.sequences)

    def __repr__(self) -> str:
        """Render as ``ProteinDataset(n=90, families={...})``."""
        return f"ProteinDataset(n={len(self)}, families={self.family_counts})"

    @property
    def families(self) -> list[str]:
        """Sorted unique group labels."""
        return sorted(set(self.labels))

    @property
    def family_counts(self) -> dict[str, int]:
        """Sequence count per group, in sorted label order."""
        counts = Counter(self.labels)
        return {family: counts[family] for family in sorted(counts)}

    @property
    def sequence_lengths(self) -> list[int]:
        """Length of each cleaned sequence."""
        return [len(s) for s in self.sequences]

    def summary(self) -> dict[str, object]:
        """Describe the dataset for logs, manifests and CLI output.

        Returns
        -------
        dict
            Sequence and family counts, length statistics, the majority-class
            fraction (the accuracy a classifier gets for free, which is the
            only honest baseline for the reported k-NN scores), and the source
            description.
        """
        lengths = self.sequence_lengths
        counts = self.family_counts
        return {
            "n_sequences": len(self),
            "n_families": len(counts),
            "family_counts": counts,
            "length_min": min(lengths) if lengths else 0,
            "length_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "length_max": max(lengths) if lengths else 0,
            "majority_class_fraction": (max(counts.values()) / len(self)) if counts else 0.0,
            "source": self.source,
        }

    def to_fasta(self, path: str | Path) -> Path:
        """Write the cleaned sequences to a FASTA file.

        Useful for handing the exact benchmarked input to another tool, and for
        making a run reproducible independently of the Pfam release.

        Headers are written as ``>{label} {sequence_id}`` so that the file
        reads back correctly through :meth:`from_fasta` with its default
        ``label_from="first_token"`` — an export its own loader cannot parse is
        worse than no export.

        Parameters
        ----------
        path
            Destination file. Parent directories are created.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record_id, label, sequence in zip(
                self.ids, self.labels, self.sequences, strict=True
            ):
                handle.write(f">{label} {record_id}\n")
                for start in range(0, len(sequence), 60):
                    handle.write(sequence[start : start + 60] + "\n")
        return path


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------


def _unique_ids(ids: list[str]) -> list[str]:
    """Disambiguate repeated identifiers while keeping the original text.

    Exported tables use ``sequence_id`` as the join key between the sequence
    index, the embedding matrices and the projections, so duplicates would make
    those files ambiguous. FASTA files repeat identifiers routinely -- and with
    ``label_from="first_token"`` every record in a group shares one -- so a
    ``#2``, ``#3`` suffix is appended to later occurrences.
    """
    seen: dict[str, int] = {}
    unique: list[str] = []
    for identifier in ids:
        count = seen.get(identifier, 0) + 1
        seen[identifier] = count
        unique.append(identifier if count == 1 else f"{identifier}#{count}")
    return unique


def _iter_fasta(path: Path) -> list[SeqRecord]:
    """Parse a FASTA file, transparently handling gzip."""
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt") as handle:
            return list(SeqIO.parse(handle, "fasta"))
    with path.open() as handle:
        return list(SeqIO.parse(handle, "fasta"))


def _record_id(record: SeqRecord, label_from: str) -> str:
    """Pick the most informative identifier left over after labelling.

    ``record.id`` is always the header's first token, which is exactly what
    ``label_from="first_token"`` consumes as the group label. In that case the
    remainder of the description is the part that still identifies the
    individual sequence.
    """
    if label_from == "first_token":
        parts = (record.description or "").split(maxsplit=1)
        if len(parts) > 1:
            return parts[1].strip()
    return str(record.id)


def _label_from_record(record: SeqRecord, label_from: str) -> str:
    """Extract a group label from a FASTA record header.

    Raises
    ------
    ValueError
        If ``label_from`` is not a recognised strategy.
    """
    fallback = str(record.id)
    description = str(record.description or fallback)

    if label_from == "first_token":
        tokens = description.split()
        return tokens[0] if tokens else fallback
    if label_from == "last_token":
        tokens = description.split()
        return tokens[-1] if tokens else fallback
    if label_from == "description":
        parts = description.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else fallback
    if label_from == "none":
        return "all"
    if label_from.startswith("pipe:"):
        try:
            index = int(label_from.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"Malformed label_from {label_from!r}; expected e.g. 'pipe:1'."
            ) from exc
        fields = description.split("|")
        if index >= len(fields):
            raise ValueError(
                f"Header {description!r} has {len(fields)} pipe-delimited field(s), "
                f"so {label_from!r} is out of range."
            )
        return fields[index].strip()

    raise ValueError(
        f"Unknown label_from {label_from!r}. Choose from 'first_token', "
        "'last_token', 'description', 'pipe:N' or 'none'."
    )
