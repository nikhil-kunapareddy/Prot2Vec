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

__version__ = "0.3.0"

from .data.alphabets import DNA, PROTEIN, RNA, Alphabet, clean_sequence, get_alphabet
from .data.dataset import ProteinDataset, SequenceDataset
from .data.pfam import download_pfam_seed, parse_pfam_families
from .embedders.base import SequenceEmbedder
from .embedders.composition import CompositionEmbedder
from .embedders.ctd import CTDEmbedder
from .embedders.dipeptide import DipeptideEmbedder
from .embedders.esm import ESMEmbedder
from .embedders.huggingface import HuggingFaceEmbedder
from .embedders.kmer import KmerEmbedder
from .embedders.llm import LLMEmbedder
from .embedders.onehot import OneHotEmbedder
from .embedders.physicochemical import PhysicochemicalEmbedder
from .evaluation import (
    METRIC_GROUPS,
    available_metrics,
    clustering_agreement,
    clustering_report,
    composition_distance_correlation,
    compute_continuity,
    compute_silhouette,
    compute_trustworthiness,
    distance_correlation,
    evaluate,
    internal_cluster_indices,
    knn_classification_report,
    knn_cv_accuracy,
    length_distance_correlation,
    length_only_knn_accuracy,
    local_continuity_meta_criterion,
    mean_average_precision,
    neighborhood_preservation,
    r_precision,
    retrieval_precision_at_k,
    same_class_auroc,
)
from .pipeline import RunConfig, run
from .reduction import (
    DimReducer,
    IsomapReducer,
    KernelPCAReducer,
    LLEReducer,
    MDSReducer,
    NMFReducer,
    PaCMAPReducer,
    PCAReducer,
    PHATEReducer,
    RandomProjectionReducer,
    SpectralReducer,
    TruncatedSVDReducer,
    TSNEReducer,
    UMAPReducer,
)

__all__ = [
    "__version__",
    # alphabets
    "Alphabet",
    "PROTEIN",
    "DNA",
    "RNA",
    "get_alphabet",
    "clean_sequence",
    # data
    "ProteinDataset",
    "SequenceDataset",
    "download_pfam_seed",
    "parse_pfam_families",
    # embedders
    "SequenceEmbedder",
    "CompositionEmbedder",
    "DipeptideEmbedder",
    "PhysicochemicalEmbedder",
    "CTDEmbedder",
    "KmerEmbedder",
    "OneHotEmbedder",
    "ESMEmbedder",
    "HuggingFaceEmbedder",
    "LLMEmbedder",
    # reduction
    "DimReducer",
    "PCAReducer",
    "TruncatedSVDReducer",
    "NMFReducer",
    "RandomProjectionReducer",
    "UMAPReducer",
    "TSNEReducer",
    "IsomapReducer",
    "MDSReducer",
    "SpectralReducer",
    "LLEReducer",
    "KernelPCAReducer",
    "PHATEReducer",
    "PaCMAPReducer",
    # evaluation
    "evaluate",
    "METRIC_GROUPS",
    "available_metrics",
    "compute_trustworthiness",
    "compute_continuity",
    "neighborhood_preservation",
    "local_continuity_meta_criterion",
    "distance_correlation",
    "knn_cv_accuracy",
    "knn_classification_report",
    "retrieval_precision_at_k",
    "mean_average_precision",
    "r_precision",
    "same_class_auroc",
    "compute_silhouette",
    "clustering_agreement",
    "clustering_report",
    "internal_cluster_indices",
    "length_distance_correlation",
    "composition_distance_correlation",
    "length_only_knn_accuracy",
    # orchestration
    "RunConfig",
    "run",
]
