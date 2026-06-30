"""Orchestrates embed → reduce → evaluate for a full benchmark run."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data.dataset import ProteinDataset
from .embedders.base import SequenceEmbedder
from .evaluation.metrics import evaluate
from .reduction.reducers import DimReducer
from .visualization.plots import metrics_bar_chart, scatter_2d

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    dataset: ProteinDataset
    embedders: list[SequenceEmbedder]
    reducer: DimReducer
    results_dir: Path = field(default_factory=lambda: Path("results"))
    cache_embeddings: bool = True
    save_figures: bool = True
    on_progress: Callable[[str, dict], None] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(results_dir: Path, embedder_name: str, dataset: ProteinDataset) -> Path:
    key = hashlib.md5("".join(dataset.sequences).encode()).hexdigest()[:8]
    return results_dir / "embeddings" / f"{embedder_name}_{key}.npy"


def _load_or_compute(
    embedder: SequenceEmbedder,
    dataset: ProteinDataset,
    cache: Path | None,
) -> np.ndarray:
    if cache and cache.exists():
        logger.info(f"Loading cached embeddings: {cache}")
        return np.load(cache, allow_pickle=False)

    logger.info(f"Computing embeddings with {embedder.name} ...")
    X = embedder.fit_transform(dataset.sequences)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        np.save(cache, X_dense)
        logger.info(f"Cached to {cache}")

    return X


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(config: RunConfig) -> pd.DataFrame:
    """Run the full benchmark. Returns a DataFrame with one row per embedder."""
    records = []

    def _emit(event: str, **kwargs) -> None:
        if config.on_progress:
            config.on_progress(event, kwargs)

    for embedder in config.embedders:
        cache = (
            _cache_path(config.results_dir, embedder.name, config.dataset)
            if config.cache_embeddings
            else None
        )
        _emit("embed_start", name=embedder.name, cached=cache and cache.exists())
        X_high = _load_or_compute(embedder, config.dataset, cache)
        _emit("embed_done", name=embedder.name)

        # Keep a dense copy for metrics (trustworthiness needs dense)
        X_high_dense = X_high.toarray() if hasattr(X_high, "toarray") else X_high

        _emit("reduce_start", name=embedder.name, reducer=config.reducer.name)
        logger.info(f"Reducing with {config.reducer.name} ...")
        X_low = config.reducer.fit_transform(X_high)
        _emit("reduce_done", name=embedder.name)

        metrics = evaluate(X_high_dense, X_low, config.dataset.labels)
        row = {"method": f"{embedder.name}+{config.reducer.name}", **metrics}
        records.append(row)
        _emit("evaluate_done", name=embedder.name, metrics=metrics)
        logger.info(row)

        if config.save_figures:
            fig_path = config.results_dir / "figures" / f"{embedder.name}_{config.reducer.name}.png"
            scatter_2d(
                X_low,
                config.dataset.labels,
                title=f"{embedder.name} + {config.reducer.name}",
                save_path=fig_path,
                dim_labels=(f"{config.reducer.name.upper()}-1", f"{config.reducer.name.upper()}-2"),
            )

    results_df = pd.DataFrame(records)

    # Persist metrics
    metrics_csv = config.results_dir / "metrics" / "benchmark.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(metrics_csv, index=False)
    logger.info(f"Metrics saved to {metrics_csv}")

    if config.save_figures:
        summary_path = config.results_dir / "figures" / "metrics_summary.png"
        metrics_bar_chart(results_df, save_path=summary_path)

    return results_df
