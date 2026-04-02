"""ESM-2 protein language model embedder.

Requires the optional [esm] dependencies:
    pip install "prot2vec[esm]"
"""
from __future__ import annotations

import logging

import numpy as np

from .base import SequenceEmbedder

logger = logging.getLogger(__name__)

_MODEL_CHECKPOINTS: dict[str, str] = {
    "esm2_t6_8M": "esm2_t6_8M_UR50D",
    "esm2_t12_35M": "esm2_t12_35M_UR50D",
    "esm2_t30_150M": "esm2_t30_150M_UR50D",
    "esm2_t33_650M": "esm2_t33_650M_UR50D",
}


class ESMEmbedder(SequenceEmbedder):
    """Mean-pooled ESM-2 token embeddings with MPS / CUDA / CPU auto-detection.

    Args:
        model_key:  One of the keys in _MODEL_CHECKPOINTS. Default is the 35M variant.
        device:     Force a specific device string. Auto-detected if None.
        batch_size: Sequences per forward pass. Reduce if you hit OOM.
        max_len:    Sequences longer than this are centre-cropped.
    """

    def __init__(
        self,
        model_key: str = "esm2_t12_35M",
        device: str | None = None,
        batch_size: int = 16,
        max_len: int = 512,
    ) -> None:
        if model_key not in _MODEL_CHECKPOINTS:
            valid = list(_MODEL_CHECKPOINTS)
            raise ValueError(f"Unknown model_key {model_key!r}. Choose from {valid}")
        self.model_key = model_key
        self.batch_size = batch_size
        self.max_len = max_len
        self._device = device
        self._model = None
        self._alphabet = None

    @property
    def name(self) -> str:
        return f"esm_{self.model_key}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            import torch
            import esm as esm_lib
        except ImportError as e:
            raise ImportError(
                "ESM dependencies not installed. Run: pip install 'prot2vec[esm]'"
            ) from e

        if self._device is None:
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

        checkpoint = _MODEL_CHECKPOINTS[self.model_key]
        loader = getattr(esm_lib.pretrained, checkpoint)
        model, alphabet = loader()
        self._model = model.eval().to(self._device)
        self._alphabet = alphabet
        logger.info(f"Loaded {checkpoint} on {self._device}")

    def _clean_seq(self, seq: str) -> str:
        """Remove non-standard tokens and centre-crop to max_len."""
        valid = set(self._alphabet.standard_toks)
        cleaned = "".join(c for c in seq if c in valid)
        if len(cleaned) > self.max_len:
            start = (len(cleaned) - self.max_len) // 2
            cleaned = cleaned[start : start + self.max_len]
        return cleaned

    def _embed_batch(self, batch: list[tuple[str, str]]) -> list[np.ndarray]:
        import torch

        _, strs, toks = self._alphabet.get_batch_converter()(batch)
        toks = toks.to(self._device)
        layer = self._model.num_layers

        with torch.no_grad():
            try:
                with torch.autocast(device_type=self._device, dtype=torch.float16):
                    out = self._model(toks, repr_layers=[layer], return_contacts=False)
            except (RuntimeError, TypeError):
                # Fallback to fp32 if autocast isn't supported on this device
                out = self._model(toks, repr_layers=[layer], return_contacts=False)

        reps = out["representations"][layer].detach().cpu().numpy()
        # Mean-pool over sequence tokens (skip BOS/EOS)
        return [reps[j, 1 : 1 + len(s), :].mean(axis=0) for j, s in enumerate(strs)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        if self._model is None:
            self._load_model()

        prepared: list[tuple[str, str]] = []
        for i, seq in enumerate(sequences):
            cleaned = self._clean_seq(seq)
            if len(cleaned) >= 10:
                prepared.append((str(i), cleaned))

        embeds: list[np.ndarray] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            embeds.extend(self._embed_batch(batch))
            logger.debug(f"Embedded {min(start + self.batch_size, len(prepared))}/{len(prepared)}")

        return np.vstack(embeds).astype(np.float32)
