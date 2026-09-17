"""Sequence alphabets, so the toolkit is not silently protein-only.

Cleaning is destructive: anything outside the alphabet is discarded. That is
correct for a Pfam seed alignment full of gaps, and catastrophic if the
alphabet is wrong — feeding DNA through the 20-amino-acid alphabet silently
keeps only the A, C, G and T characters that happen to also be amino acid
codes, deletes every other base, and returns a dataset that looks fine and
means nothing. Naming the alphabet explicitly makes that failure impossible to
reach by accident.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Alphabet:
    """A fixed, ordered set of sequence tokens.

    Parameters
    ----------
    name
        Short identifier used in configs and result metadata.
    tokens
        The canonical tokens, in the order that fixes the meaning of every
        column in a composition-style embedding.
    description
        One-line human-readable description.
    """

    name: str
    tokens: str
    description: str

    def __len__(self) -> int:
        """Return the number of tokens."""
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        """Return whether ``token`` is part of this alphabet."""
        return token in self.tokens

    def __str__(self) -> str:
        """Render as ``protein (20 tokens)``."""
        return f"{self.name} ({len(self)} tokens)"

    @property
    def token_set(self) -> frozenset[str]:
        """The tokens as a set, for fast membership tests."""
        return frozenset(self.tokens)

    @property
    def index(self) -> dict[str, int]:
        """Map each token to its canonical column position."""
        return {token: position for position, token in enumerate(self.tokens)}


#: The 20 standard amino acids. Excludes gaps, the ambiguity codes
#: ``B``/``Z``/``J``/``X``, and the non-canonical residues ``U``
#: (selenocysteine) and ``O`` (pyrrolysine), because embedders disagree on how
#: to tokenise them and the choice would differ between methods being compared.
PROTEIN = Alphabet("protein", "ACDEFGHIKLMNPQRSTVWY", "20 standard amino acids")

#: DNA bases. ``N`` and the other IUPAC ambiguity codes are dropped.
DNA = Alphabet("dna", "ACGT", "DNA nucleotides")

#: RNA bases.
RNA = Alphabet("rna", "ACGU", "RNA nucleotides")

#: Alphabets addressable by name from a config file.
ALPHABETS: dict[str, Alphabet] = {a.name: a for a in (PROTEIN, DNA, RNA)}


def get_alphabet(alphabet: str | Alphabet = PROTEIN) -> Alphabet:
    """Resolve an alphabet from a name, or pass an :class:`Alphabet` through.

    Parameters
    ----------
    alphabet
        ``"protein"``, ``"dna"``, ``"rna"``, or an :class:`Alphabet`.

    Returns
    -------
    Alphabet
        The resolved alphabet.

    Raises
    ------
    ValueError
        If the name is not recognised.

    Examples
    --------
    >>> get_alphabet("dna").tokens
    'ACGT'
    """
    if isinstance(alphabet, Alphabet):
        return alphabet
    try:
        return ALPHABETS[alphabet]
    except KeyError:
        raise ValueError(
            f"Unknown alphabet {alphabet!r}. Choose from {sorted(ALPHABETS)}."
        ) from None


def clean_sequence(sequence: str, alphabet: str | Alphabet = PROTEIN) -> str:
    """Strip every character that is not a token of ``alphabet``.

    Parameters
    ----------
    sequence
        Raw sequence, possibly gapped and in mixed case.
    alphabet
        Alphabet to keep. Defaults to protein.

    Returns
    -------
    str
        Uppercase sequence containing only alphabet tokens.

    Examples
    --------
    >>> clean_sequence("mkt-AY..iX")
    'MKTAYI'
    >>> clean_sequence("acgt-NNNacgt", alphabet="dna")
    'ACGTACGT'
    """
    tokens = get_alphabet(alphabet).token_set
    return "".join(c for c in sequence.upper() if c in tokens)


def require_protein(alphabet: Alphabet, descriptor: str) -> None:
    """Raise unless ``alphabet`` is protein.

    Physicochemical scales, CTD property groupings and protein language models
    are defined over amino acids and have no meaning for nucleotides. Refusing
    is better than returning a vector of zeros that looks like a result.

    Raises
    ------
    ValueError
        If ``alphabet`` is not the protein alphabet.
    """
    if alphabet.name != PROTEIN.name:
        raise ValueError(
            f"{descriptor} is defined for amino acids only, but the dataset "
            f"alphabet is {alphabet.name!r}. Use a composition, k-mer or "
            "one-hot representation for nucleotide sequences."
        )


__all__ = [
    "ALPHABETS",
    "DNA",
    "PROTEIN",
    "RNA",
    "Alphabet",
    "clean_sequence",
    "get_alphabet",
    "require_protein",
]
