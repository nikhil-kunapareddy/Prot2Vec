"""The single interface every embedder implements."""

from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

from .._matrix import EmbeddingMatrix


class SequenceEmbedder(ABC):
    """Turn protein sequences into vectors.

    Implementations must honour two rules, because the benchmark pairs every
    output row with a family label positionally:

    1. **One row out per sequence in.** Never silently drop a sequence — raise
       instead, so the caller can filter deliberately.
    2. **Order preserved.** Row ``i`` of the result describes ``sequences[i]``.

    Returning a :mod:`scipy.sparse` matrix is allowed for very wide
    representations such as k-mer counts; reducers and the pipeline accept both
    dense and sparse input.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in filenames, logs and result tables."""
        ...

    @abstractmethod
    def fit_transform(self, sequences: list[str]) -> EmbeddingMatrix:
        """Embed ``sequences`` and return a matrix of shape ``(len(sequences), d)``.

        Parameters
        ----------
        sequences
            Protein sequences as uppercase one-letter amino acid strings.

        Returns
        -------
        EmbeddingMatrix
            One row per input sequence, in input order. Dense, or
            :mod:`scipy.sparse` for very wide representations.
        """
        ...

    def __repr__(self) -> str:
        """Render as ``ClassName(name='...')``."""
        return f"{type(self).__name__}(name={self.name!r})"

    @property
    def params(self) -> dict[str, Any]:
        """Configuration that determines this embedder's output.

        Derived from the constructor signature rather than from
        ``vars(self)``: an embedder legitimately holds runtime state too — a
        resolved device, a loaded model, a call counter — and folding that into
        the identity of a configuration would change the cache key mid-run.
        Parameters stored under a leading underscore are still found, so
        ``device`` kept as ``self._device`` is reported.

        Override this if an implementation derives its behaviour from something
        the constructor does not name.
        """
        signature = inspect.signature(type(self).__init__)
        found: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.name == "self" or parameter.kind in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                continue
            for attribute in (parameter.name, f"_{parameter.name}"):
                if hasattr(self, attribute):
                    found[parameter.name] = getattr(self, attribute)
                    break
        return found

    @property
    def cache_key(self) -> str:
        """Stable identifier for caching, covering every setting that changes output.

        :attr:`name` alone is not enough: two ``LLMEmbedder`` instances that
        differ only in ``output_dim`` share a name, and reusing one's cached
        vectors for the other would silently mix representations within a
        single results table.
        """
        settings = sorted(
            (key, value)
            for key, value in self.params.items()
            if isinstance(value, str | int | float | bool | type(None))
        )
        if not settings:
            return self.name
        rendered = ",".join(f"{key}={value!r}" for key, value in settings)
        digest = hashlib.sha1(rendered.encode(), usedforsecurity=False).hexdigest()[:8]
        return f"{self.name}-{digest}"

    def _validate_rows(self, result: EmbeddingMatrix, sequences: list[str]) -> None:
        """Assert the row-count contract, raising a message that names the cause.

        Raises
        ------
        RuntimeError
            If ``result`` does not have exactly one row per input sequence.
        """
        if result.shape[0] != len(sequences):
            raise RuntimeError(
                f"{self.name} returned {result.shape[0]} rows for "
                f"{len(sequences)} sequences. Embedders must emit one row per "
                "input sequence so that rows stay aligned with family labels."
            )
