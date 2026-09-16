"""Sequence embedders. All implement :class:`SequenceEmbedder`.

``ESMEmbedder`` and ``LLMEmbedder`` import their heavy/optional dependencies
lazily, so they are safe to import without the ``[esm]`` / ``[llm]`` extras
installed — the ``ImportError`` is raised only when you actually embed.
"""

from .base import SequenceEmbedder
from .composition import CompositionEmbedder
from .esm import ESMEmbedder
from .kmer import KmerEmbedder
from .llm import LLMEmbedder

__all__ = [
    "CompositionEmbedder",
    "ESMEmbedder",
    "KmerEmbedder",
    "LLMEmbedder",
    "SequenceEmbedder",
]
