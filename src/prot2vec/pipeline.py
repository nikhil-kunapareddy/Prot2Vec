"""Orchestration: embed once, reduce many ways, score every pair.

A run is a matrix. Each embedder is computed a single time and then handed to
every reducer, because embedding is the expensive half — an ESM-2 forward pass
or a billed embedding API call — while reducing the same matrix three ways
costs seconds. Comparing PCA against UMAP against t-SNE on identical vectors is
therefore nearly free, and it is the comparison that reveals when a striking
projection is an artefact of the reducer rather than a property of the
representation.

Everything a run produces is written next to the metrics: the exact sequences
benchmarked, the embeddings themselves, the 2-D coordinates behind each figure,
and a manifest recording the versions and settings that produced them. The
point is that a number in ``benchmark.csv`` can be traced back to the inputs
that made it months later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse

from . import __version__
from ._matrix import EmbeddingMatrix, as_csr, as_dense, is_sparse
from .data.dataset import ProteinDataset
from .embedders.base import SequenceEmbedder
from .evaluation.metrics import evaluate
from .reduction.reducers import DimReducer

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Everything one benchmark run needs.

    Attributes
    ----------
    dataset
        Sequences and family labels to benchmark.
    embedders
        Representations to compare. Each is computed once and scored against
        every reducer.
    reducers
        Dimensionality reducers to apply. A single reducer may be passed
        directly instead of a one-element list.
    results_dir
        Root for ``embeddings/``, ``projections/``, ``metrics/`` and
        ``figures/`` output.
    cache_embeddings
        Reuse embeddings already on disk for the same embedder settings and
        dataset. Keep this on for API-backed embedders, where a re-run
        otherwise costs real quota.
    save_figures
        Write per-pair scatter plots and a summary bar chart.
    export_embeddings
        Write the sequence index, the embedding matrices and the 2-D
        projections in plain, id-aligned formats for use outside Prot2Vec.
    metric_params
        Extra keyword arguments forwarded to
        :func:`prot2vec.evaluation.evaluate`, e.g.
        ``{"metric_groups": ["classification", "retrieval"]}``. The dataset's
        sequences and alphabet are supplied automatically.
    on_progress
        Optional callback invoked as ``on_progress(event, payload)`` so a CLI
        can render progress without the pipeline depending on the CLI.
    """

    dataset: ProteinDataset
    embedders: list[SequenceEmbedder]
    reducers: list[DimReducer] | DimReducer
    results_dir: Path = field(default_factory=lambda: Path("results"))
    cache_embeddings: bool = True
    save_figures: bool = True
    export_embeddings: bool = True
    metric_params: dict[str, Any] = field(default_factory=dict)
    on_progress: Callable[[str, dict[str, Any]], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Normalise ``reducers`` and reject a run that cannot produce results.

        Raises
        ------
        ValueError
            If no embedder or no reducer is configured, if the dataset is
            empty, or if fewer than two families are present — every metric
            here measures *between-family* structure, so one family cannot be
            scored.
        """
        if isinstance(self.reducers, DimReducer):
            self.reducers = [self.reducers]
        self.reducers = list(self.reducers)
        self.results_dir = Path(self.results_dir)

        if not self.embedders:
            raise ValueError("At least one embedder is required.")
        if not self.reducers:
            raise ValueError("At least one reducer is required.")
        if len(self.dataset) == 0:
            raise ValueError("The dataset is empty — nothing to benchmark.")
        if len(self.dataset.families) < 2:
            raise ValueError(
                f"Only one family ({self.dataset.families[0]}) is present. Family "
                "separability needs at least two."
            )

    @property
    def reducer_list(self) -> list[DimReducer]:
        """The reducers as a list, whatever form they were supplied in."""
        if isinstance(self.reducers, DimReducer):
            return [self.reducers]
        return list(self.reducers)

    @property
    def n_steps(self) -> int:
        """Total progress steps: one embed per embedder, plus reduce and score per pair."""
        return len(self.embedders) * (1 + 2 * len(self.reducer_list))


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def _dataset_fingerprint(dataset: ProteinDataset) -> str:
    """Hash the exact sequences a cache entry was computed from."""
    digest = hashlib.sha256()
    for sequence in dataset.sequences:
        digest.update(sequence.encode())
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


def _cache_path(results_dir: Path, embedder: SequenceEmbedder, dataset: ProteinDataset) -> Path:
    """Build the cache path stem for one (embedder settings, dataset) combination.

    The suffix is left off; :func:`_save_embedding` appends ``.npy`` or
    ``.npz`` depending on whether the embedding is dense or sparse.
    """
    return results_dir / "embeddings" / f"{embedder.cache_key}_{_dataset_fingerprint(dataset)}"


def _save_embedding(stem: Path, X: EmbeddingMatrix) -> Path:
    """Persist ``X``, preserving sparsity so reloads are byte-equivalent."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    if is_sparse(X):
        path = stem.with_suffix(".npz")
        scipy.sparse.save_npz(path, as_csr(X))
    else:
        path = stem.with_suffix(".npy")
        np.save(path, np.asarray(X))
    return path


def _load_embedding(stem: Path) -> EmbeddingMatrix | None:
    """Load a cached embedding for ``stem``, or return ``None`` if absent."""
    sparse_path = stem.with_suffix(".npz")
    if sparse_path.exists():
        return scipy.sparse.load_npz(sparse_path)
    dense_path = stem.with_suffix(".npy")
    if dense_path.exists():
        return np.load(dense_path, allow_pickle=False)
    return None


def _load_or_compute(
    embedder: SequenceEmbedder,
    dataset: ProteinDataset,
    cache_stem: Path | None,
) -> EmbeddingMatrix:
    """Return ``embedder``'s vectors for ``dataset``, reusing the cache if present.

    A sparse embedding is cached in sparse form. Densifying it on write would
    mean the first run reduced a sparse matrix and the second reduced a dense
    one — UMAP and t-SNE do not necessarily agree across that change, and a
    benchmark whose numbers move when you re-run it is not a benchmark.
    """
    if cache_stem is not None:
        cached = _load_embedding(cache_stem)
        if cached is not None:
            logger.info("Loaded cached embeddings for %s", embedder.name)
            if cached.shape[0] != len(dataset):
                raise RuntimeError(
                    f"Cached embedding for {embedder.name} has {cached.shape[0]} rows "
                    f"but the dataset has {len(dataset)}. Delete "
                    f"{cache_stem.parent} to force a re-embed."
                )
            return cached

    logger.info("Computing embeddings with %s ...", embedder.name)
    X = embedder.fit_transform(dataset.sequences)

    if X.shape[0] != len(dataset):
        raise RuntimeError(
            f"{embedder.name} returned {X.shape[0]} rows for {len(dataset)} "
            "sequences; rows would not line up with family labels."
        )

    if cache_stem is not None:
        path = _save_embedding(cache_stem, X)
        logger.info("Cached embeddings to %s", path)

    return X


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def _write_sequence_index(dataset: ProteinDataset, results_dir: Path) -> Path:
    """Write the benchmarked sequences in row order, as the key to every export.

    Row ``i`` of this file corresponds to row ``i`` of every embedding matrix
    and projection, which is what makes the numeric exports usable elsewhere.
    """
    path = results_dir / "embeddings" / "sequences.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "row": range(len(dataset)),
            "sequence_id": dataset.ids,
            "family": dataset.labels,
            "length": dataset.sequence_lengths,
            "sequence": dataset.sequences,
        }
    )
    frame.to_csv(path, sep="\t", index=False)
    return path


def _export_embedding(embedder: SequenceEmbedder, X: EmbeddingMatrix, results_dir: Path) -> Path:
    """Write one embedding matrix as a plain dense ``.npy`` for external use."""
    path = results_dir / "embeddings" / f"{embedder.name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, as_dense(X).astype(np.float32))
    return path


def _export_projection(
    dataset: ProteinDataset,
    X_low: np.ndarray,
    embedder: SequenceEmbedder,
    reducer: DimReducer,
    results_dir: Path,
) -> Path:
    """Write the 2-D coordinates behind a figure as a tidy TSV.

    Shipping the coordinates, not just the PNG, means the figure can be
    rebuilt in ggplot2 or plotly without re-running the embedding.
    """
    path = results_dir / "projections" / f"{embedder.name}__{reducer.name}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {f"dim{i + 1}": X_low[:, i] for i in range(X_low.shape[1])}
    frame = pd.DataFrame({"sequence_id": dataset.ids, "family": dataset.labels, **columns})
    frame.to_csv(path, sep="\t", index=False)
    return path


def _dependency_versions() -> dict[str, str]:
    """Collect versions of the libraries that can change a numeric result."""
    from importlib.metadata import PackageNotFoundError, version

    packages = [
        "biopython",
        "fair-esm",
        "google-genai",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "umap-learn",
    ]
    found: dict[str, str] = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            continue
    return found


def _git_commit() -> str | None:
    """Return the current git commit, or ``None`` outside a repository."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _describe(obj: SequenceEmbedder | DimReducer) -> dict[str, Any]:
    """Record an embedder's or reducer's public settings for the manifest."""

    def scalars(mapping: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in mapping.items()
            if isinstance(value, str | int | float | bool | type(None))
        }

    params = getattr(obj, "params", None)
    settings = scalars(params) if isinstance(params, dict) else {}
    return {"name": obj.name, "class": type(obj).__name__, "settings": settings}


def _write_manifest(
    config: RunConfig,
    results: pd.DataFrame,
    elapsed: float,
    results_dir: Path,
    skipped: list[dict[str, str]] | None = None,
) -> Path:
    """Write a JSON record of what was run, with what, and on which versions."""
    manifest = {
        "prot2vec_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 3),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "dataset": config.dataset.summary(),
        "embedders": [_describe(e) for e in config.embedders],
        "reducers": [_describe(r) for r in config.reducer_list],
        "metric_params": config.metric_params,
        "dependencies": _dependency_versions(),
        "results": results.to_dict(orient="records"),
        "skipped_pairs": skipped or [],
    }
    path = results_dir / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(config: RunConfig) -> pd.DataFrame:
    """Run the benchmark matrix and return one row per (embedder, reducer) pair.

    Parameters
    ----------
    config
        The run description. See :class:`RunConfig`.

    Returns
    -------
    pandas.DataFrame
        Columns ``embedder``, ``reducer``, ``method`` and one per metric,
        sorted in the order the pairs were evaluated. Also written to
        ``<results_dir>/metrics/benchmark.csv``.

    Raises
    ------
    RuntimeError
        If an embedder returns a row count that does not match the dataset, or
        if every pair failed. Individual pair failures are recorded under
        ``skipped_pairs`` in the run manifest and do not stop the run.
    """
    reducers = config.reducer_list
    results_dir = config.results_dir
    started = time.time()
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    def emit(event: str, **payload: Any) -> None:
        if config.on_progress:
            config.on_progress(event, payload)

    if config.export_embeddings:
        index_path = _write_sequence_index(config.dataset, results_dir)
        logger.info("Wrote sequence index to %s", index_path)

    for embedder in config.embedders:
        cache = (
            _cache_path(results_dir, embedder, config.dataset) if config.cache_embeddings else None
        )
        emit(
            "embed_start",
            name=embedder.name,
            cached=cache is not None and _load_embedding(cache) is not None,
        )
        X_high = _load_or_compute(embedder, config.dataset, cache)
        emit("embed_done", name=embedder.name)

        if config.export_embeddings:
            _export_embedding(embedder, X_high, results_dir)

        # Metrics need dense input; do the conversion once per embedder rather
        # than once per reducer.
        X_high_dense = as_dense(X_high)

        for reducer in reducers:
            method = f"{embedder.name}+{reducer.name}"
            emit("reduce_start", name=embedder.name, reducer=reducer.name, method=method)
            logger.info("Reducing %s with %s ...", embedder.name, reducer.name)

            # A matrix run must survive one incompatible cell. NMF rejects
            # signed input, PHATE may not be installed, a manifold method can
            # fail on a disconnected neighbour graph -- none of which is a
            # reason to discard the other pairs that already succeeded.
            try:
                X_low = reducer.fit_transform(X_high)
            except (ImportError, ValueError, RuntimeError, ArithmeticError) as exc:
                logger.warning("Skipping %s: %s", method, exc)
                skipped.append({"method": method, "stage": "reduce", "reason": str(exc)})
                emit(
                    "pair_skipped",
                    name=embedder.name,
                    reducer=reducer.name,
                    reason=str(exc),
                    stage="reduce",
                )
                continue
            emit("reduce_done", name=embedder.name, reducer=reducer.name)

            # Sequences and alphabet are supplied so the confound group can
            # run: "is this embedding just encoding sequence length?" is
            # unanswerable from the vectors alone.
            try:
                metrics = evaluate(
                    X_high_dense,
                    X_low,
                    config.dataset.labels,
                    sequences=config.dataset.sequences,
                    alphabet=config.dataset.alphabet,
                    **config.metric_params,
                )
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                logger.warning("Skipping %s: %s", method, exc)
                skipped.append({"method": method, "stage": "evaluate", "reason": str(exc)})
                emit(
                    "pair_skipped",
                    name=embedder.name,
                    reducer=reducer.name,
                    reason=str(exc),
                    stage="evaluate",
                )
                continue
            records.append(
                {
                    "embedder": embedder.name,
                    "reducer": reducer.name,
                    "method": method,
                    **metrics,
                }
            )
            emit("evaluate_done", name=embedder.name, reducer=reducer.name, metrics=metrics)
            logger.info("%s: %s", method, metrics)

            if config.export_embeddings:
                _export_projection(config.dataset, X_low, embedder, reducer, results_dir)

            if config.save_figures:
                from .visualization.plots import scatter_2d

                scatter_2d(
                    X_low,
                    config.dataset.labels,
                    title=method,
                    save_path=results_dir / "figures" / f"{embedder.name}_{reducer.name}.png",
                    dim_labels=(f"{reducer.name.upper()}-1", f"{reducer.name.upper()}-2"),
                )

    if not records:
        details = "; ".join(f"{s['method']}: {s['reason']}" for s in skipped)
        raise RuntimeError(f"Every (embedder, reducer) pair failed. {details}")
    if skipped:
        logger.warning(
            "%d of %d pairs were skipped: %s",
            len(skipped),
            len(records) + len(skipped),
            ", ".join(s["method"] for s in skipped),
        )

    results_df = pd.DataFrame(records)

    metrics_csv = results_dir / "metrics" / "benchmark.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(metrics_csv, index=False)
    logger.info("Metrics saved to %s", metrics_csv)

    if config.save_figures:
        from .visualization.plots import metrics_bar_chart

        primary = (
            "knn_accuracy_highdim_mean"
            if "knn_accuracy_highdim_mean" in results_df.columns
            else "knn_accuracy_mean"
        )
        metrics_bar_chart(
            results_df,
            metric=primary,
            save_path=results_dir / "figures" / "metrics_summary.png",
        )

    manifest_path = _write_manifest(
        config, results_df, time.time() - started, results_dir, skipped=skipped
    )
    logger.info("Run manifest written to %s", manifest_path)

    return results_df
