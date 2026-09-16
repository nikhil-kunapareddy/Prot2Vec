"""General-purpose LLM embedding APIs applied to raw amino acid strings.

This is a deliberately naive baseline, and that is the point. A text embedding
model has never been told that ``MKTAYIAKQRQ`` is a protein; it tokenises the
letters the way it would tokenise any other string. Including it in the
benchmark quantifies how much of a "protein embedding" result is really
protein-specific signal and how much any competent sequence model would
recover from character statistics alone.

Interpret its scores as a floor, not as a protein method.

Requires the optional ``[llm]`` extra::

    pip install "prot2vec[llm]"
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from .base import SequenceEmbedder

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Default embedding model per provider.
PROVIDER_DEFAULTS: dict[str, str] = {
    "google": "gemini-embedding-001",
}

#: Environment variable holding each provider's API key.
PROVIDER_KEY_ENV: dict[str, str] = {
    "google": "GOOGLE_API_KEY",
}

#: Substrings that mark a retryable (rate-limit or transient server) failure.
_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "resource_exhausted",
    "rate limit",
    "ratelimit",
    "quota",
    "deadline",
    "unavailable",
    "timeout",
)


def _is_retryable(exc: Exception) -> bool:
    """Return ``True`` for rate limits and transient server errors.

    Matches on the rendered exception because the Google SDK surfaces HTTP
    status through a small number of generic exception classes, so the class
    name alone does not distinguish "slow down" from "your key is invalid".
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


class LLMEmbedder(SequenceEmbedder):
    """Embed sequences by sending them as plain text to an embedding API.

    Parameters
    ----------
    provider
        Currently ``"google"``.
    model
        Model name; defaults to the provider's entry in
        :data:`PROVIDER_DEFAULTS`.
    batch_size
        Sequences per API request. Requests are genuinely batched, so this
        directly controls how many round trips a run costs.
    api_key_env
        Environment variable holding the API key. Defaults to the provider's
        conventional name.
    max_len
        Characters of each sequence to send. Truncation is unavoidable for long
        proteins and is applied from the N-terminus.
    output_dim
        Request a specific embedding width. ``gemini-embedding-001`` returns
        3072 dimensions by default and supports truncation to smaller widths;
        truncated vectors are re-normalised, which the API expects callers to
        do.
    retry_delay
        Initial back-off in seconds; doubles on each retry.
    max_retries
        Attempts per request before giving up.

    Notes
    -----
    Runs cost real API quota and are not free of side effects, so keep
    ``cache_embeddings`` enabled: a repeated benchmark then re-reads the cached
    vectors instead of re-billing the request.
    """

    def __init__(
        self,
        provider: str = "google",
        model: str | None = None,
        batch_size: int = 32,
        api_key_env: str | None = None,
        max_len: int = 512,
        output_dim: int | None = None,
        retry_delay: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        if provider not in PROVIDER_DEFAULTS:
            raise ValueError(
                f"Unknown provider {provider!r}. Choose from {list(PROVIDER_DEFAULTS)}."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if output_dim is not None and output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}.")

        self.provider = provider
        self.model = model or PROVIDER_DEFAULTS[provider]
        self.batch_size = batch_size
        self.max_len = max_len
        self.output_dim = output_dim
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.api_key_env = api_key_env or PROVIDER_KEY_ENV[provider]

    @property
    def name(self) -> str:
        """Identifier such as ``llm_google_gemini-embedding-001``."""
        return f"llm_{self.provider}_{self.model}"

    @property
    def params(self) -> dict[str, Any]:
        """Configuration that changes the vectors, excluding retry behaviour.

        ``retry_delay`` and ``max_retries`` affect only how failures are
        handled, never the embeddings. Leaving them out of the cache key
        matters here more than anywhere else: a changed retry setting would
        otherwise invalidate the cache and re-bill every request.
        """
        return {
            key: value
            for key, value in super().params.items()
            if key not in ("retry_delay", "max_retries")
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        """Read the API key from the environment.

        Raises
        ------
        RuntimeError
            If the variable is unset or empty.
        """
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"No API key found in ${self.api_key_env}. Add it to a .env file "
                f"at the repository root as {self.api_key_env}=... , or export it "
                "in your shell. Keys are never read from the config file."
            )
        return key

    def _with_retry(self, call: Callable[[], T], what: str) -> T:
        """Call ``call``, retrying retryable failures with exponential back-off.

        Raises
        ------
        Exception
            The last error, once ``max_retries`` attempts are exhausted.
        """
        delay = self.retry_delay
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return call()
            except Exception as exc:
                last_error = exc
                if not _is_retryable(exc) or attempt == self.max_retries:
                    raise
                logger.warning(
                    "%s failed (%s); retrying in %.1fs (attempt %d/%d)",
                    what,
                    type(exc).__name__,
                    delay,
                    attempt,
                    self.max_retries,
                )
                time.sleep(delay)
                delay *= 2

        raise RuntimeError(f"{what} failed after {self.max_retries} attempts") from last_error

    def _batches(self, sequences: list[str]) -> list[list[str]]:
        """Split ``sequences`` into truncated batches of at most ``batch_size``."""
        truncated = [s[: self.max_len] for s in sequences]
        return [
            truncated[i : i + self.batch_size] for i in range(0, len(truncated), self.batch_size)
        ]

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def _embed_google(self, sequences: list[str]) -> np.ndarray:
        """Embed via the Google AI embedding API."""
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "The Google embedding API needs the google-genai SDK. "
                'Install it with: pip install "prot2vec[llm]"'
            ) from exc

        client = genai.Client(api_key=self._resolve_api_key())
        config: Any = (
            types.EmbedContentConfig(output_dimensionality=self.output_dim)
            if self.output_dim is not None
            else None
        )

        vectors: list[list[float]] = []
        batches = self._batches(sequences)
        for i, batch in enumerate(batches, start=1):

            def _call(batch: list[str] = batch) -> list[list[float]]:
                response = client.models.embed_content(
                    model=self.model, contents=batch, config=config
                )
                return [list(e.values) for e in response.embeddings]

            batch_vectors = self._with_retry(_call, f"embed batch {i}/{len(batches)}")
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"API returned {len(batch_vectors)} embeddings for a batch of "
                    f"{len(batch)}. Retry, or lower batch_size."
                )
            vectors.extend(batch_vectors)
            logger.debug("Embedded %d/%d sequences", len(vectors), len(sequences))

        result = np.asarray(vectors, dtype=np.float32)
        if self.output_dim is not None:
            # Gemini normalises only its full-width output; truncated vectors
            # must be re-normalised before cosine distances mean anything.
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.where(norms == 0, 1.0, norms)
        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        """Embed ``sequences`` through the configured provider.

        Parameters
        ----------
        sequences
            Protein sequences as one-letter amino acid strings.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(sequences), d)``, ``float32``.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        RuntimeError
            If the API key is missing, or the provider returns a batch of the
            wrong size.
        ImportError
            If the ``[llm]`` extra is not installed.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")

        logger.info(
            "LLMEmbedder(%s, %s): embedding %d sequences in %d request(s)",
            self.provider,
            self.model,
            len(sequences),
            -(-len(sequences) // self.batch_size),
        )

        if self.provider == "google":
            result = self._embed_google(sequences)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"Unknown provider: {self.provider!r}")

        self._validate_rows(result, sequences)
        return result
