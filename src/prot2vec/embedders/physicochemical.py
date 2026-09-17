"""Physicochemical property profiles: a small, interpretable descriptor."""

from __future__ import annotations

import numpy as np

from ..data.alphabets import PROTEIN, Alphabet, get_alphabet, require_protein
from .base import SequenceEmbedder

#: Kyte & Doolittle (1982) hydropathy. The mean of this scale over a sequence
#: is GRAVY, the standard grand average of hydropathicity.
HYDROPATHY: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

#: Residue molecular weight in daltons.
MOLECULAR_WEIGHT: dict[str, float] = {
    "A": 89.09,
    "R": 174.20,
    "N": 132.12,
    "D": 133.10,
    "C": 121.16,
    "Q": 146.15,
    "E": 147.13,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "L": 131.17,
    "K": 146.19,
    "M": 149.21,
    "F": 165.19,
    "P": 115.13,
    "S": 105.09,
    "T": 119.12,
    "W": 204.23,
    "Y": 181.19,
    "V": 117.15,
}

#: Net side-chain charge near pH 7. Histidine is partially protonated.
CHARGE: dict[str, float] = {
    "A": 0.0,
    "R": 1.0,
    "N": 0.0,
    "D": -1.0,
    "C": 0.0,
    "Q": 0.0,
    "E": -1.0,
    "G": 0.0,
    "H": 0.1,
    "I": 0.0,
    "L": 0.0,
    "K": 1.0,
    "M": 0.0,
    "F": 0.0,
    "P": 0.0,
    "S": 0.0,
    "T": 0.0,
    "W": 0.0,
    "Y": 0.0,
    "V": 0.0,
}

#: Grantham (1974) polarity.
POLARITY: dict[str, float] = {
    "A": 8.1,
    "R": 10.5,
    "N": 11.6,
    "D": 13.0,
    "C": 5.5,
    "Q": 10.5,
    "E": 12.3,
    "G": 9.0,
    "H": 10.4,
    "I": 5.2,
    "L": 4.9,
    "K": 11.3,
    "M": 5.7,
    "F": 5.2,
    "P": 8.0,
    "S": 9.2,
    "T": 8.6,
    "W": 5.4,
    "Y": 6.2,
    "V": 5.9,
}

#: Zamyatnin (1972) residue volume in cubic angstroms.
VOLUME: dict[str, float] = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}

#: The scales applied, in output order.
SCALES: dict[str, dict[str, float]] = {
    "hydropathy": HYDROPATHY,
    "molecular_weight": MOLECULAR_WEIGHT,
    "charge": CHARGE,
    "polarity": POLARITY,
    "volume": VOLUME,
}

#: Summary statistics computed per scale, in output order.
STATISTICS = ("mean", "std", "min", "max")

#: Residues with aromatic side chains.
AROMATIC = frozenset("FWY")

#: Residues carrying a full charge near pH 7.
CHARGED = frozenset("DEKR")


class PhysicochemicalEmbedder(SequenceEmbedder):
    """Summarise a sequence by the chemistry of its residues.

    Every other representation here is either compositional bookkeeping or an
    opaque learned vector. This one is neither: each of its 23 features is a
    physical quantity with units, so a separation it achieves comes with an
    explanation. If two families split on ``hydropathy_mean``, one is more
    hydrophobic than the other, and that is a statement a reviewer can check.

    Its low dimensionality is the point. With 23 features it cannot overfit a
    small seed alignment the way an 8,000-column k-mer matrix can, which makes
    it the right sanity check when a high-dimensional method reports a
    suspiciously perfect score.

    The features are, for each of the five scales in :data:`SCALES`, the mean,
    standard deviation, minimum and maximum over the residues; plus the log
    sequence length, the aromatic fraction and the charged fraction.

    Parameters
    ----------
    alphabet
        Must be protein. Nucleotide alphabets are rejected, because these
        scales describe amino acid side chains.

    Notes
    -----
    Scales are standardised across the amino acid vocabulary before use, so
    that molecular weight (tens to hundreds of daltons) and charge (-1 to 1)
    contribute comparably rather than the largest-magnitude scale dominating
    every distance.
    """

    def __init__(self, alphabet: str | Alphabet = PROTEIN) -> None:
        resolved = get_alphabet(alphabet)
        require_protein(resolved, "PhysicochemicalEmbedder")
        self.alphabet = resolved.name
        self._spec = resolved
        self._scales = self._standardised_scales()

    @staticmethod
    def _standardised_scales() -> dict[str, dict[str, float]]:
        """Z-score each scale over the 20 residues so no scale dominates."""
        standardised: dict[str, dict[str, float]] = {}
        for scale_name, values in SCALES.items():
            array = np.fromiter((values[residue] for residue in PROTEIN.tokens), dtype=np.float64)
            spread = float(array.std())
            centre = float(array.mean())
            if spread == 0.0:  # pragma: no cover - no constant scale is shipped
                spread = 1.0
            standardised[scale_name] = {
                residue: (values[residue] - centre) / spread for residue in PROTEIN.tokens
            }
        return standardised

    @property
    def name(self) -> str:
        """Identifier: ``physicochemical``."""
        return "physicochemical"

    @property
    def embedding_dim(self) -> int:
        """Width of the output: four statistics per scale, plus three globals."""
        return len(SCALES) * len(STATISTICS) + 3

    def feature_names(self) -> list[str]:
        """Column labels in output order, e.g. ``["hydropathy_mean", ...]``."""
        names = [f"{scale}_{statistic}" for scale in SCALES for statistic in STATISTICS]
        return [*names, "log_length", "aromatic_fraction", "charged_fraction"]

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        """Compute the property profile of every sequence.

        Parameters
        ----------
        sequences
            Protein sequences as one-letter amino acid strings. Non-standard
            residues are ignored.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(sequences), 23)``, ``float32``. A sequence
            with no standard residues yields a zero row.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")

        vectors = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)
        tokens = self._spec.token_set

        for row, sequence in enumerate(sequences):
            residues = [c for c in sequence.upper() if c in tokens]
            if not residues:
                continue

            features: list[float] = []
            for scale in self._scales.values():
                profile = np.fromiter((scale[residue] for residue in residues), dtype=np.float64)
                features += [
                    float(profile.mean()),
                    float(profile.std()),
                    float(profile.min()),
                    float(profile.max()),
                ]

            total = len(residues)
            features.append(float(np.log1p(total)))
            features.append(sum(r in AROMATIC for r in residues) / total)
            features.append(sum(r in CHARGED for r in residues) / total)
            vectors[row] = features

        self._validate_rows(vectors, sequences)
        return vectors
