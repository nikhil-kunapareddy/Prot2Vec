"""Sequence embedders. All implement :class:`SequenceEmbedder`.

Three families are available, and they answer different questions:

**Interpretable descriptors** — :class:`CompositionEmbedder`,
:class:`DipeptideEmbedder`, :class:`PhysicochemicalEmbedder`,
:class:`CTDEmbedder`. Every feature has a meaning, they need no pretraining,
and they run in milliseconds. Use them as the floor that a learned
representation has to beat, and as a check when a high-dimensional method
reports a suspiciously perfect score.

**Corpus-derived** — :class:`KmerEmbedder` (TF-IDF over k-mers) and
:class:`OneHotEmbedder`. No pretraining either, but the features are defined by
the dataset or by position rather than by chemistry.

**Pretrained models** — :class:`ESMEmbedder` (via fair-esm),
:class:`HuggingFaceEmbedder` (ProtBERT, ProtT5, Ankh, ProstT5, the ESM-2 ports,
nucleotide models), and :class:`LLMEmbedder`, which is a control rather than a
protein method.

Embedders with heavy or optional dependencies import them lazily, so they are
safe to import without the ``[esm]`` / ``[hf]`` / ``[llm]`` extras installed —
the ``ImportError`` is raised only when you actually embed.
"""

from .base import SequenceEmbedder
from .composition import CompositionEmbedder
from .ctd import CTDEmbedder
from .dipeptide import DipeptideEmbedder
from .esm import ESMEmbedder
from .huggingface import HuggingFaceEmbedder
from .kmer import KmerEmbedder
from .llm import LLMEmbedder
from .onehot import OneHotEmbedder
from .physicochemical import PhysicochemicalEmbedder

__all__ = [
    "CTDEmbedder",
    "CompositionEmbedder",
    "DipeptideEmbedder",
    "ESMEmbedder",
    "HuggingFaceEmbedder",
    "KmerEmbedder",
    "LLMEmbedder",
    "OneHotEmbedder",
    "PhysicochemicalEmbedder",
    "SequenceEmbedder",
]
