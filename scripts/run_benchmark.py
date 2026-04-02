"""CLI script: run a full embedding benchmark from a YAML config."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from prot2vec.data.dataset import ProteinDataset
from prot2vec.data.pfam import download_pfam_seed, parse_pfam_families
from prot2vec.embedders.base import SequenceEmbedder
from prot2vec.pipeline import RunConfig, run
from prot2vec.reduction.reducers import DimReducer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_embedder(cfg: dict) -> SequenceEmbedder:
    name = cfg["name"]
    if name == "composition":
        from prot2vec.embedders.composition import CompositionEmbedder

        return CompositionEmbedder()
    if name == "kmer":
        from prot2vec.embedders.kmer import KmerEmbedder

        return KmerEmbedder(k=cfg.get("k", 3))
    if name == "esm2":
        from prot2vec.embedders.esm import ESMEmbedder

        return ESMEmbedder(
            model_key=cfg.get("model_key", "esm2_t12_35M"),
            batch_size=cfg.get("batch_size", 16),
            max_len=cfg.get("max_len", 512),
        )
    raise ValueError(f"Unknown embedder: {name!r}")


def _build_reducer(cfg: dict) -> DimReducer:
    name = cfg["name"]
    if name == "pca":
        from prot2vec.reduction.reducers import PCAReducer

        return PCAReducer(n_components=cfg.get("n_components", 2))
    if name == "umap":
        from prot2vec.reduction.reducers import UMAPReducer

        return UMAPReducer(
            n_neighbors=cfg.get("n_neighbors", 15),
            min_dist=cfg.get("min_dist", 0.1),
            metric=cfg.get("metric", "cosine"),
        )
    if name == "tsne":
        from prot2vec.reduction.reducers import TSNEReducer

        return TSNEReducer(perplexity=cfg.get("perplexity", 30))
    raise ValueError(f"Unknown reducer: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prot2vec embedding benchmark.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to experiment YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"Config not found: {config_path}")

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    pfam_cfg = cfg["pfam"]
    seed_path = download_pfam_seed(
        version=pfam_cfg["version"],
        cache_dir=cfg.get("data", {}).get("cache_dir", "data/raw"),
    )
    records = parse_pfam_families(pfam_ids=pfam_cfg["families"], seed_path=seed_path)
    dataset = ProteinDataset.from_pfam_records(
        records,
        min_length=cfg.get("data", {}).get("min_seq_length", 50),
    )
    logger.info(dataset)

    embedders = [_build_embedder(e) for e in cfg["embedders"]]
    reducer = _build_reducer(cfg["reducer"])

    run_cfg = RunConfig(
        dataset=dataset,
        embedders=embedders,
        reducer=reducer,
        results_dir=Path(cfg.get("results_dir", "results")),
        cache_embeddings=cfg.get("cache_embeddings", True),
        save_figures=cfg.get("save_figures", True),
    )
    results = run(run_cfg)

    print("\n=== Benchmark Results ===")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
