"""ESM-2 protein language model embeddings.

ESM-2 is the reference point this benchmark exists to measure against: a
transformer pretrained on UniRef that has seen no Pfam labels, yet whose
mean-pooled residue representations usually separate families far better than
any hand-designed feature.

Requires the optional ``[esm]`` extra::

    pip install "prot2vec[esm]"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import SequenceEmbedder

logger = logging.getLogger(__name__)

#: Public ESM-2 checkpoints, smallest first. Embedding width grows with depth.
MODEL_CHECKPOINTS: dict[str, str] = {
    "esm2_t6_8M": "esm2_t6_8M_UR50D",
    "esm2_t12_35M": "esm2_t12_35M_UR50D",
    "esm2_t30_150M": "esm2_t30_150M_UR50D",
    "esm2_t33_650M": "esm2_t33_650M_UR50D",
}

#: Embedding width per checkpoint, for documentation and sanity checks.
MODEL_DIMS: dict[str, int] = {
    "esm2_t6_8M": 320,
    "esm2_t12_35M": 480,
    "esm2_t30_150M": 640,
    "esm2_t33_650M": 1280,
}


class ESMEmbedder(SequenceEmbedder):
    """Mean-pooled ESM-2 residue embeddings with automatic device selection.

    Parameters
    ----------
    model_key
        One of :data:`MODEL_CHECKPOINTS`. Weights are downloaded by
        ``fair-esm`` on first use and cached under ``~/.cache/torch/hub``.
    device
        Force ``"cpu"``, ``"cuda"`` or ``"mps"``. Auto-detected when ``None``.
    batch_size
        Sequences per forward pass. Lower this first if you hit out-of-memory.
    max_len
        Residues to keep. Longer sequences are centre-cropped, which retains
        the domain core that Pfam families are defined on rather than the
        terminal extensions that vary most within a family.

    Notes
    -----
    Attention is quadratic in sequence length, so runtime is dominated by
    ``max_len`` rather than by sequence count. On CPU prefer ``esm2_t6_8M``.
    """

    def __init__(
        self,
        model_key: str = "esm2_t12_35M",
        device: str | None = None,
        batch_size: int = 16,
        max_len: int = 512,
    ) -> None:
        if model_key not in MODEL_CHECKPOINTS:
            raise ValueError(
                f"Unknown model_key {model_key!r}. Choose from {list(MODEL_CHECKPOINTS)}."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}.")

        self.model_key = model_key
        self.batch_size = batch_size
        self.max_len = max_len
        self._device = device
        self._model: Any = None
        self._alphabet: Any = None
        self._use_autocast = False

    @property
    def name(self) -> str:
        """Identifier such as ``esm_esm2_t12_35M``."""
        return f"esm_{self.model_key}"

    @property
    def embedding_dim(self) -> int:
        """Width of the vectors this checkpoint produces."""
        return MODEL_DIMS[self.model_key]

    @property
    def device(self) -> str | None:
        """Device in use, or ``None`` before the model is loaded."""
        return self._device

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Resolve the device, then download and load the checkpoint."""
        try:
            import esm as esm_lib
            import torch
        except ImportError as exc:
            raise ImportError(
                "ESM-2 support needs PyTorch and fair-esm. "
                'Install them with: pip install "prot2vec[esm]"'
            ) from exc

        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        # fp16 autocast helps on GPU and is a pessimisation (or unsupported) on
        # CPU, so decide once here instead of guessing per batch.
        self._use_autocast = self._device in ("cuda", "mps")

        checkpoint = MODEL_CHECKPOINTS[self.model_key]
        logger.info("Loading %s on %s ...", checkpoint, self._device)
        model, alphabet = getattr(esm_lib.pretrained, checkpoint)()
        self._model = model.eval().to(self._device)
        self._alphabet = alphabet
        logger.info("Loaded %s (%d-d) on %s", checkpoint, self.embedding_dim, self._device)

    # ------------------------------------------------------------------
    # Sequence preparation
    # ------------------------------------------------------------------

    def _clean_seq(self, seq: str) -> str:
        """Drop residues outside the model alphabet and centre-crop to ``max_len``."""
        valid = set(self._alphabet.standard_toks)
        cleaned = "".join(c for c in seq.upper() if c in valid)
        if len(cleaned) > self.max_len:
            start = (len(cleaned) - self.max_len) // 2
            cleaned = cleaned[start : start + self.max_len]
        return cleaned

    def _embed_batch(self, batch: list[tuple[str, str]]) -> list[np.ndarray]:
        """Embed one batch and mean-pool over residues, skipping BOS/EOS."""
        import torch

        _, strs, tokens = self._alphabet.get_batch_converter()(batch)
        tokens = tokens.to(self._device)
        layer = self._model.num_layers

        with torch.inference_mode():
            if self._use_autocast:
                with torch.autocast(device_type=str(self._device), dtype=torch.float16):
                    out = self._model(tokens, repr_layers=[layer], return_contacts=False)
            else:
                out = self._model(tokens, repr_layers=[layer], return_contacts=False)

        reps = out["representations"][layer].float().cpu().numpy()
        return [reps[j, 1 : 1 + len(s), :].mean(axis=0) for j, s in enumerate(strs)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        """Embed ``sequences``, returning one mean-pooled vector each.

        Parameters
        ----------
        sequences
            Protein sequences as one-letter amino acid strings.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(sequences), embedding_dim)``, ``float32``.

        Raises
        ------
        ValueError
            If ``sequences`` is empty, or if any sequence contains no residues
            the model alphabet recognises. Dropping such rows would silently
            misalign every later row against its family label, so this fails
            loudly instead — filter with ``min_seq_length`` if you expect them.
        ImportError
            If the ``[esm]`` extra is not installed.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")
        if self._model is None:
            self._load_model()

        prepared: list[tuple[str, str]] = []
        empty: list[int] = []
        for i, seq in enumerate(sequences):
            cleaned = self._clean_seq(seq)
            if not cleaned:
                empty.append(i)
            prepared.append((str(i), cleaned))

        if empty:
            preview = empty[:10]
            raise ValueError(
                f"{len(empty)} sequence(s) contain no residues ESM-2 recognises "
                f"(indices {preview}{' ...' if len(empty) > len(preview) else ''}). "
                "Filter them out before embedding — for example with a higher "
                "`min_seq_length` in your config."
            )

        embeddings: list[np.ndarray] = []
        total = len(prepared)
        for start in range(0, total, self.batch_size):
            batch = prepared[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))
            logger.debug("Embedded %d/%d sequences", min(start + self.batch_size, total), total)

        result = np.vstack(embeddings).astype(np.float32)
        self._validate_rows(result, sequences)
        return result
