"""LLM-based sequence embedder (Google Gemini).

Sends protein sequences as plain-text strings to a general-purpose embedding
model. Useful for benchmarking whether non-biological LLM embedders incidentally
capture protein-family structure.

Supported providers
-------------------
google  — ``gemini-embedding-001``  via Google AI API (env: GOOGLE_API_KEY)

Install optional dependencies
------------------------------
    pip install "prot2vec[llm]"
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from .base import SequenceEmbedder

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, str] = {
    "google": "gemini-embedding-001",
}


class LLMEmbedder(SequenceEmbedder):
    """Embed protein sequences using a general-purpose LLM embedding model.

    Args:
        provider:       Currently supported: ``"google"``.
        model:          Model name. Defaults to ``gemini-embedding-001``.
        batch_size:     Sequences per API call.
        api_key_env:    Environment variable holding the API key.
        max_len:        Truncate sequences to this many characters.
        retry_delay:    Seconds between retries on rate-limit errors.
        max_retries:    Max retry attempts per request.
    """

    def __init__(
        self,
        provider: str = "google",
        model: str | None = None,
        batch_size: int = 64,
        api_key_env: str | None = None,
        max_len: int = 512,
        retry_delay: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        if provider not in _DEFAULTS:
            raise ValueError(
                f"Unknown provider {provider!r}. Choose from: {list(_DEFAULTS)}"
            )
        self.provider = provider
        self.model = model or _DEFAULTS[provider]
        self.batch_size = batch_size
        self.max_len = max_len
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.api_key_env = api_key_env or "GOOGLE_API_KEY"

    @property
    def name(self) -> str:
        return f"llm_{self.provider}_{self.model}"

    def _truncate(self, sequences: list[str]) -> list[str]:
        return [s[: self.max_len] for s in sequences]

    def _resolve_api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise EnvironmentError(
                f"API key not found. Set the {self.api_key_env!r} environment variable."
            )
        return key

    @staticmethod
    def _with_retry(fn, max_retries: int, retry_delay: float):
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as exc:
                if "RateLimit" in type(exc).__name__ and attempt < max_retries:
                    logger.warning(
                        "Rate limit hit — retrying in %.1fs (attempt %d/%d)",
                        retry_delay, attempt, max_retries,
                    )
                    time.sleep(retry_delay)
                else:
                    raise

    def _embed_google(self, sequences: list[str]) -> np.ndarray:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai not installed. Run: pip install 'prot2vec[llm]'"
            ) from exc

        client = genai.Client(api_key=self._resolve_api_key())
        all_embeds: list[list[float]] = []

        for i, seq in enumerate(sequences):
            def _call(s=seq):
                result = client.models.embed_content(model=self.model, contents=s)
                return result.embeddings[0].values

            all_embeds.append(self._with_retry(_call, self.max_retries, self.retry_delay))
            if (i + 1) % 10 == 0:
                logger.debug("Google: embedded %d/%d", i + 1, len(sequences))

        return np.array(all_embeds, dtype=np.float32)

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        logger.info(
            "LLMEmbedder(%s, %s): embedding %d sequences",
            self.provider, self.model, len(sequences),
        )
        seqs = self._truncate(sequences)

        if self.provider == "google":
            return self._embed_google(seqs)
        raise ValueError(f"Unknown provider: {self.provider!r}")
