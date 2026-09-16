"""Prot2Vec — benchmark protein sequence embeddings on protein family separability.

Prot2Vec answers a practical question for computational biologists: *for my
sequences, which vector representation actually separates protein families?*
It runs a matrix of embedders (amino acid composition, k-mer TF-IDF, ESM-2
protein language models, general-purpose LLM embedding APIs) against a matrix
of dimensionality reducers, and scores every pair with the same metrics.

Quick start
-----------
>>> from prot2vec import (ProteinDataset, CompositionEmbedder, PCAReducer,
...                       RunConfig, run)
>>> dataset = ProteinDataset.from_fasta("my_proteins.fasta")
>>> results = run(RunConfig(dataset=dataset,
...                         embedders=[CompositionEmbedder()],
...                         reducers=[PCAReducer()]))

See https://github.com/nikhil-kunapareddy/Prot2Vec for full documentation.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .data.dataset import ProteinDataset
from .data.pfam import download_pfam_seed, parse_pfam_families
from .embedders.base import SequenceEmbedder
from .embedders.composition import CompositionEmbedder
from .embedders.esm import ESMEmbedder
from .embedders.kmer import KmerEmbedder
from .embedders.llm import LLMEmbedder
from .evaluation.metrics import (
    clustering_agreement,
    compute_silhouette,
    compute_trustworthiness,
    evaluate,
    knn_cv_accuracy,
    retrieval_precision_at_k,
)
from .pipeline import RunConfig, run
from .reduction.reducers import DimReducer, PCAReducer, TSNEReducer, UMAPReducer

__all__ = [
    "__version__",
    # data
    "ProteinDataset",
    "download_pfam_seed",
    "parse_pfam_families",
    # embedders
    "SequenceEmbedder",
    "CompositionEmbedder",
    "KmerEmbedder",
    "ESMEmbedder",
    "LLMEmbedder",
    # reduction
    "DimReducer",
    "PCAReducer",
    "UMAPReducer",
    "TSNEReducer",
    # evaluation + orchestration
    "evaluate",
    "compute_trustworthiness",
    "compute_silhouette",
    "knn_cv_accuracy",
    "retrieval_precision_at_k",
    "clustering_agreement",
    "RunConfig",
    "run",
]
