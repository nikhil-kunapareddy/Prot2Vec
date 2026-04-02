"""Visualization utilities for embedding comparisons."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def scatter_2d(
    Z: np.ndarray,
    labels: list[str],
    title: str,
    save_path: Path | None = None,
    dim_labels: tuple[str, str] = ("Dim 1", "Dim 2"),
) -> None:
    """2D scatter plot coloured by family label."""
    sns.set_context("talk")
    sns.set_style("whitegrid")

    families = sorted(set(labels))
    palette = sns.color_palette("tab10", n_colors=len(families))
    labels_arr = np.array(labels)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    for fam, color in zip(families, palette):
        mask = labels_arr == fam
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            label=fam, alpha=0.7, s=40,
            edgecolor="k", linewidth=0.3, color=color,
        )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(dim_labels[0], fontsize=12)
    ax.set_ylabel(dim_labels[1], fontsize=12)
    ax.legend(title="Pfam Family", loc="best", fontsize=10, framealpha=0.8)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def metrics_bar_chart(
    results: pd.DataFrame,
    metric: str = "knn_accuracy_mean",
    title: str = "5-NN CV Accuracy by Method",
    save_path: Path | None = None,
) -> None:
    """Bar chart comparing a scalar metric across all methods."""
    sns.set_context("talk")
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.5), 5), dpi=150)
    x = np.arange(len(results))
    ax.bar(x, results[metric], color=sns.color_palette("tab10", len(results)))

    if metric == "knn_accuracy_mean" and "knn_accuracy_std" in results.columns:
        ax.errorbar(x, results[metric], yerr=results["knn_accuracy_std"],
                    fmt="none", color="black", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(results["method"], rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, weight="bold")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
