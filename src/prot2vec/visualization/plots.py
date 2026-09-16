"""Figure helpers for embedding comparisons.

These functions never call :func:`matplotlib.pyplot.show`. Prot2Vec runs
unattended in CI and on headless compute nodes, where an interactive ``show()``
either blocks or warns, and every un-closed figure leaks memory across a long
benchmark matrix. Each helper returns its :class:`~matplotlib.figure.Figure` so
notebook users keep full control, and closes it when it was only ever written
to disk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def _style() -> None:
    """Apply the shared Prot2Vec figure style."""
    sns.set_context("talk")
    sns.set_style("whitegrid")


def _save(fig: Figure, save_path: Path | str | None) -> None:
    """Write ``fig`` to ``save_path``, creating parent directories as needed."""
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def scatter_2d(
    Z: np.ndarray,
    labels: list[str],
    title: str,
    save_path: Path | str | None = None,
    dim_labels: tuple[str, str] = ("Dim 1", "Dim 2"),
    close: bool | None = None,
) -> Figure:
    """Plot a 2-D projection coloured by protein family.

    Parameters
    ----------
    Z
        Projected coordinates, shape ``(n_samples, 2)``.
    labels
        Family label per sample; determines colour and legend order.
    title
        Figure title.
    save_path
        Written as a 150-dpi PNG when given.
    dim_labels
        Axis labels, e.g. ``("UMAP-1", "UMAP-2")``.
    close
        Close the figure before returning. Defaults to ``True`` when the figure
        was saved (batch use) and ``False`` otherwise (interactive use).

    Returns
    -------
    matplotlib.figure.Figure
        The figure, whether or not it was closed.

    Raises
    ------
    ValueError
        If ``Z`` is not two-dimensional or disagrees with ``labels`` in length.
    """
    Z = np.asarray(Z)
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError(f"scatter_2d needs an (n, 2) array, got shape {Z.shape}.")
    if Z.shape[0] != len(labels):
        raise ValueError(f"Z has {Z.shape[0]} rows but {len(labels)} labels were given.")

    _style()
    families = sorted(set(labels))
    palette = sns.color_palette("tab10", n_colors=max(len(families), 3))
    labels_arr = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    for family, color in zip(families, palette, strict=False):
        mask = labels_arr == family
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            label=f"{family}  (n={int(mask.sum())})",
            alpha=0.75,
            s=40,
            edgecolor="k",
            linewidth=0.3,
            color=color,
        )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(dim_labels[0], fontsize=12)
    ax.set_ylabel(dim_labels[1], fontsize=12)
    ax.legend(title="Pfam family", loc="best", fontsize=9, framealpha=0.85)
    fig.tight_layout()

    _save(fig, save_path)
    if close if close is not None else save_path is not None:
        plt.close(fig)
    return fig


def metrics_bar_chart(
    results: pd.DataFrame,
    metric: str = "knn_accuracy_mean",
    title: str | None = None,
    save_path: Path | str | None = None,
    close: bool | None = None,
) -> Figure:
    """Compare one scalar metric across every benchmarked method.

    Parameters
    ----------
    results
        Benchmark results with a ``method`` column plus ``metric``.
    metric
        Column to plot. When a matching ``*_std`` column exists it is drawn as
        error bars.
    title
        Figure title; derived from ``metric`` when omitted.
    save_path
        Written as a 150-dpi PNG when given.
    close
        See :func:`scatter_2d`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, whether or not it was closed.

    Raises
    ------
    ValueError
        If ``metric`` is not a column of ``results``.
    """
    if metric not in results.columns:
        raise ValueError(f"{metric!r} is not in the results columns: {sorted(results.columns)}")

    _style()
    pretty = metric.removesuffix("_mean").replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(max(6.0, len(results) * 1.6), 5.0), dpi=150)
    x = np.arange(len(results))
    ax.bar(x, results[metric], color=sns.color_palette("tab10", max(len(results), 3)))

    std_column = metric.replace("_mean", "_std")
    if std_column != metric and std_column in results.columns:
        ax.errorbar(
            x, results[metric], yerr=results[std_column], fmt="none", color="black", capsize=4
        )

    ax.set_xticks(x)
    ax.set_xticklabels(results["method"], rotation=30, ha="right")
    ax.set_ylabel(pretty)
    ax.set_ylim(0, 1.05)
    ax.set_title(title or f"{pretty} by method", fontsize=14, weight="bold")
    fig.tight_layout()

    _save(fig, save_path)
    if close if close is not None else save_path is not None:
        plt.close(fig)
    return fig
