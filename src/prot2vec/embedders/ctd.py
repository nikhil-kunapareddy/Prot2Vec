"""Composition / Transition / Distribution descriptors (Dubchak et al.)."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from ..data.alphabets import PROTEIN, Alphabet, get_alphabet, require_protein
from .base import SequenceEmbedder

#: The seven standard CTD properties, each partitioning the 20 residues into
#: three groups. These are the groupings used by PROFEAT and iFeature, so the
#: resulting features are directly comparable with the large body of published
#: work built on them.
PROPERTY_GROUPS: dict[str, tuple[str, str, str]] = {
    # polar / neutral / hydrophobic
    "hydrophobicity": ("RKEDQN", "GASTPHY", "CLVIMFW"),
    # small / medium / large
    "vdw_volume": ("GASTPDC", "NVEQIL", "MHKFRYW"),
    # low / medium / high
    "polarity": ("LIFWCMVY", "PATGS", "HQRKNED"),
    # low / medium / high
    "polarizability": ("GASDT", "CPNVEQIL", "KMHFRYW"),
    # positive / neutral / negative
    "charge": ("KR", "ANCQGHILMFPSTWYV", "DE"),
    # helix / strand / coil
    "secondary_structure": ("EALMQKRH", "VIYCWFT", "GNPSD"),
    # buried / exposed / intermediate
    "solvent_accessibility": ("ALFCGIVW", "RKQEND", "MSPTHY"),
}

#: Quantiles at which the distribution features are sampled, as percentages.
#: ``0`` means the first occurrence of the group.
DISTRIBUTION_QUANTILES = (0, 25, 50, 75, 100)

#: Features contributed per property: 3 composition + 3 transition + 15
#: distribution.
FEATURES_PER_PROPERTY = 3 + 3 + 3 * len(DISTRIBUTION_QUANTILES)


def _validate_partitions() -> None:
    """Check every property partitions the 20 residues exactly once.

    A residue missing from a property, or appearing in two of its groups, would
    silently skew that property's composition features rather than raising, so
    the tables are checked once at import.

    Raises
    ------
    ValueError
        If any property is not a clean partition of the protein alphabet.
    """
    expected = set(PROTEIN.tokens)
    for property_name, groups in PROPERTY_GROUPS.items():
        combined = "".join(groups)
        if len(combined) != len(expected) or set(combined) != expected:
            raise ValueError(
                f"CTD property {property_name!r} does not partition the 20 "
                f"residues: {len(combined)} entries, "
                f"missing {sorted(expected - set(combined))}, "
                f"duplicated {sorted({c for c in combined if combined.count(c) > 1})}."
            )


_validate_partitions()


class CTDEmbedder(SequenceEmbedder):
    """Encode residue-class composition, adjacency and positional spread.

    CTD asks three questions of each chemical property, and the third is what
    makes it distinctive among the descriptors here:

    **Composition** — how much of each class is present. This is what amino
    acid composition already measures, coarsened into three bins.

    **Transition** — how often the sequence switches between classes. A protein
    with alternating polar and hydrophobic residues and one with the same
    residues in two blocks have identical composition and very different
    transition profiles. This is where membrane topology and amphipathic
    helices show up.

    **Distribution** — *where* along the sequence each class first appears and
    accumulates, as a fraction of length. Because the positions are normalised,
    this captures "the hydrophobic residues are concentrated in the C-terminal
    third" in a way that survives differences in sequence length — unlike
    positional one-hot encoding, which needs the sequences to line up.

    Seven properties x 21 features = 147 dimensions.

    Parameters
    ----------
    alphabet
        Must be protein; the property groupings are amino acid classes.
    """

    def __init__(self, alphabet: str | Alphabet = PROTEIN) -> None:
        resolved = get_alphabet(alphabet)
        require_protein(resolved, "CTDEmbedder")
        self.alphabet = resolved.name
        self._spec = resolved

    @property
    def name(self) -> str:
        """Identifier: ``ctd``."""
        return "ctd"

    @property
    def embedding_dim(self) -> int:
        """Width of the output: 147 for the seven standard properties."""
        return len(PROPERTY_GROUPS) * FEATURES_PER_PROPERTY

    def feature_names(self) -> list[str]:
        """Column labels in output order."""
        names: list[str] = []
        for property_name in PROPERTY_GROUPS:
            names += [f"{property_name}_C_g{g + 1}" for g in range(3)]
            names += [f"{property_name}_T_g{a + 1}g{b + 1}" for a, b in ((0, 1), (0, 2), (1, 2))]
            for g in range(3):
                names += [f"{property_name}_D_g{g + 1}_{q}" for q in DISTRIBUTION_QUANTILES]
        return names

    @staticmethod
    def _distribution(positions: list[int], length: int) -> list[float]:
        """Positions of the quantile-th occurrence of a class, as fractions."""
        if not positions:
            return [0.0] * len(DISTRIBUTION_QUANTILES)
        count = len(positions)
        features: list[float] = []
        for quantile in DISTRIBUTION_QUANTILES:
            rank = 0 if quantile == 0 else int(np.ceil(count * quantile / 100.0)) - 1
            rank = max(0, min(rank, count - 1))
            # positions are 1-based, so this is a fraction of the sequence.
            features.append(positions[rank] / length)
        return features

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        """Compute CTD features for every sequence.

        Parameters
        ----------
        sequences
            Protein sequences as one-letter amino acid strings. Non-standard
            residues are removed first, so they neither occupy a position nor
            create a spurious class transition.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(sequences), 147)``, ``float32``. A sequence
            with no standard residues yields a zero row.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")

        # Precompute residue -> group index per property.
        lookups = [
            {residue: index for index, group in enumerate(groups) for residue in group}
            for groups in PROPERTY_GROUPS.values()
        ]
        tokens = self._spec.token_set
        vectors = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)

        for row, sequence in enumerate(sequences):
            residues = [c for c in sequence.upper() if c in tokens]
            length = len(residues)
            if length == 0:
                continue

            features: list[float] = []
            for lookup in lookups:
                classes = [lookup[residue] for residue in residues]

                # Composition.
                counts = [classes.count(g) for g in range(3)]
                features += [count / length for count in counts]

                # Transition: adjacent positions spanning two different classes.
                transitions = dict.fromkeys(((0, 1), (0, 2), (1, 2)), 0)
                for first, second in pairwise(classes):
                    if first != second:
                        transitions[(min(first, second), max(first, second))] += 1
                denominator = length - 1 if length > 1 else 1
                features += [transitions[pair] / denominator for pair in ((0, 1), (0, 2), (1, 2))]

                # Distribution.
                for g in range(3):
                    positions = [i + 1 for i, c in enumerate(classes) if c == g]
                    features += self._distribution(positions, length)

            vectors[row] = features

        self._validate_rows(vectors, sequences)
        return vectors
