"""Protein language models from the Hugging Face Hub.

One class covers most of the field: ProtBERT and ProtT5 from Rostlab, Ankh,
ProstT5, the ESM-2 ports, and nucleotide models such as DNABERT. They differ in
tokeniser conventions and architecture rather than in anything Prot2Vec cares
about, so those differences are handled here instead of in five near-identical
classes.

Requires the optional ``[hf]`` extra::

    pip install "prot2vec[hf]"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import SequenceEmbedder

logger = logging.getLogger(__name__)

#: Short names for widely used checkpoints. Any Hub model id also works.
PRESETS: dict[str, str] = {
    # Rostlab -- the most cited protein language models.
    "protbert": "Rostlab/prot_bert",
    "protbert_bfd": "Rostlab/prot_bert_bfd",
    "prott5": "Rostlab/prot_t5_xl_uniref50",
    "prott5_bfd": "Rostlab/prot_t5_xl_bfd",
    "prostt5": "Rostlab/ProstT5",
    # Ankh -- encoder-decoder models trained with a protein-specific objective.
    "ankh_base": "ElnaggarLab/ankh-base",
    "ankh_large": "ElnaggarLab/ankh-large",
    # ESM-2 via Hugging Face rather than fair-esm.
    "esm2_8m": "facebook/esm2_t6_8M_UR50D",
    "esm2_35m": "facebook/esm2_t12_35M_UR50D",
    "esm2_150m": "facebook/esm2_t30_150M_UR50D",
    "esm2_650m": "facebook/esm2_t33_650M_UR50D",
    # Nucleotide models, for use with the dna/rna alphabets.
    "dnabert2": "zhihan1996/DNABERT-2-117M",
    "nucleotide_transformer": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
}

#: Checkpoints whose tokenisers expect residues separated by spaces.
_SPACE_SEPARATED_PREFIXES = ("Rostlab/", "ElnaggarLab/")

#: Approximate parameter counts, to warn before a very large download.
_LARGE_MODEL_MARKERS = ("xl", "650m", "large", "500m", "3b")

POOLING_STRATEGIES = ("mean", "cls", "max")


class HuggingFaceEmbedder(SequenceEmbedder):
    """Mean-pooled hidden states from any Hugging Face sequence encoder.

    Parameters
    ----------
    model_name
        A key of :data:`PRESETS` (e.g. ``"protbert"``) or any Hub model id
        (e.g. ``"Rostlab/prot_bert"``).
    pooling
        How to collapse per-token states into one vector per sequence:

        ``"mean"``
            Average over real residue positions, excluding padding and special
            tokens. The standard choice and the default.
        ``"cls"``
            The first token's state. Meaningful only for models pretrained with
            a sentence-level objective; for masked-LM-only protein models it is
            usually worse than the mean.
        ``"max"``
            Element-wise maximum over residues, which emphasises the strongest
            motif-like activations.
    layer
        Hidden layer to pool, ``-1`` being the last. Intermediate layers often
        transfer better than the final one, which is specialised towards the
        pretraining objective.
    batch_size
        Sequences per forward pass. Lower this first on out-of-memory.
    max_len
        Tokens to keep; longer sequences are truncated by the tokeniser.
    device
        Force ``"cpu"``, ``"cuda"`` or ``"mps"``. Auto-detected when ``None``.
    trust_remote_code
        Allow the checkpoint to execute its own modelling code. Required by
        some nucleotide models, including DNABERT-2. Off by default because it
        runs code downloaded from the Hub — enable it only for checkpoints you
        trust.

    Notes
    -----
    Weights are cached by ``huggingface_hub`` under ``~/.cache/huggingface``.
    The ProtT5 and Ankh checkpoints are several gigabytes; prefer ``esm2_35m``
    or ``protbert`` when iterating.
    """

    def __init__(
        self,
        model_name: str = "protbert",
        pooling: str = "mean",
        layer: int = -1,
        batch_size: int = 8,
        max_len: int = 1024,
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(
                f"Unknown pooling {pooling!r}. Choose from {list(POOLING_STRATEGIES)}."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if max_len < 2:
            raise ValueError(f"max_len must be >= 2, got {max_len}.")

        self.model_name = model_name
        self.pooling = pooling
        self.layer = layer
        self.batch_size = batch_size
        self.max_len = max_len
        self.trust_remote_code = trust_remote_code
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._space_separated = False

    @property
    def repo_id(self) -> str:
        """The resolved Hub model id."""
        return PRESETS.get(self.model_name, self.model_name)

    @property
    def name(self) -> str:
        """Identifier such as ``hf_prot_bert_mean``."""
        slug = self.repo_id.split("/")[-1].replace("-", "_")
        layer = "" if self.layer == -1 else f"_L{self.layer}"
        return f"hf_{slug}_{self.pooling}{layer}"

    @property
    def device(self) -> str | None:
        """Device in use, or ``None`` before the model is loaded."""
        return self._device

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Download and load the tokeniser and encoder."""
        try:
            import torch
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Hugging Face support needs transformers and PyTorch. "
                'Install them with: pip install "prot2vec[hf]"'
            ) from exc

        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        repo = self.repo_id
        if any(marker in repo.lower() for marker in _LARGE_MODEL_MARKERS):
            logger.warning("%s is a large checkpoint; the first run downloads several GB.", repo)

        self._space_separated = repo.startswith(_SPACE_SEPARATED_PREFIXES)
        common = {"trust_remote_code": self.trust_remote_code}

        self._tokenizer = AutoTokenizer.from_pretrained(repo, do_lower_case=False, **common)

        config = AutoConfig.from_pretrained(repo, **common)
        if getattr(config, "is_encoder_decoder", False) or config.model_type in ("t5",):
            # ProtT5, ProstT5 and Ankh ship as encoder-decoders; only the
            # encoder produces residue representations, and loading the decoder
            # would double the memory for nothing.
            from transformers import T5EncoderModel

            model = T5EncoderModel.from_pretrained(repo, **common)
        else:
            model = AutoModel.from_pretrained(repo, **common)

        self._model = model.eval().to(self._device)
        logger.info("Loaded %s on %s", repo, self._device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _prepare(self, sequences: list[str]) -> list[str]:
        """Apply the tokeniser conventions the checkpoint expects."""
        prepared = [s.upper() for s in sequences]
        if self._space_separated:
            prepared = [" ".join(s) for s in prepared]
        if "prostt5" in self.repo_id.lower():
            # ProstT5 is bilingual over sequence and structure alphabets and
            # needs to be told which one it is receiving.
            prepared = [f"<AA2fold> {s}" for s in prepared]
        return prepared

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        """Encode one batch and pool to one vector per sequence."""
        import torch

        encoded = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        special = encoded.pop("special_tokens_mask")
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.inference_mode():
            output = self._model(**encoded, output_hidden_states=True)

        states = output.hidden_states[self.layer]

        if self.pooling == "cls":
            return states[:, 0, :].float().cpu().numpy()

        # Pool over real residues only: padding and the [CLS]/[SEP]/</s> tokens
        # carry no residue information, and averaging them in shifts every
        # vector toward a length-dependent constant.
        mask = encoded["attention_mask"].bool() & ~special.bool().to(self._device)
        mask = mask.unsqueeze(-1)

        if self.pooling == "max":
            neutral = torch.finfo(states.dtype).min
            return states.masked_fill(~mask, neutral).max(dim=1).values.float().cpu().numpy()

        summed = (states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return (summed / counts).float().cpu().numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        """Embed ``sequences`` with the configured checkpoint.

        Parameters
        ----------
        sequences
            Sequences as one-letter token strings.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(len(sequences), hidden_size)``, ``float32``.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        ImportError
            If the ``[hf]`` extra is not installed.
        """
        if not sequences:
            raise ValueError("No sequences to embed.")
        if self._model is None:
            self._load_model()

        prepared = self._prepare(sequences)
        chunks: list[np.ndarray] = []
        for start in range(0, len(prepared), self.batch_size):
            chunks.append(self._embed_batch(prepared[start : start + self.batch_size]))
            logger.debug(
                "Embedded %d/%d sequences",
                min(start + self.batch_size, len(prepared)),
                len(prepared),
            )

        result = np.vstack(chunks).astype(np.float32)
        self._validate_rows(result, sequences)
        return result
