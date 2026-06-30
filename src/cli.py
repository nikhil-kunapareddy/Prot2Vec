"""Command-line entry points: benchmark runner and Pfam data downloader."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import yaml
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Load API keys from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .data.dataset import ProteinDataset
from .data.pfam import download_pfam_seed, parse_pfam_families
from .embedders.base import SequenceEmbedder
from .pipeline import RunConfig, run
from .reduction.reducers import DimReducer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

console = Console()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def print_banner() -> None:
    title = Text()
    title.append("PROT", style="bold cyan")
    title.append("2", style="bold yellow")
    title.append("VEC", style="bold cyan")
    title.append("\n")
    title.append("Protein Sequence Embedding Benchmark", style="dim white")
    title.append("  ·  ", style="dim")
    title.append("v0.1.0", style="dim cyan")

    console.print()
    console.print(Panel(Align.center(title), border_style="cyan", padding=(1, 8)))
    console.print()


def print_config_summary(cfg: dict, dataset: ProteinDataset) -> None:
    console.print(Rule("[bold cyan]Experiment Configuration[/bold cyan]", style="cyan"))
    console.print()

    # Families
    families = cfg["pfam"]["families"]
    fam_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    fam_table.add_column(style="dim")
    fam_table.add_column(style="white")
    fam_table.add_row("Pfam version", cfg["pfam"]["version"])
    fam_table.add_row("Families", "  ".join(f"[cyan]{f}[/cyan]" for f in families))
    fam_table.add_row("Sequences", f"[yellow]{len(dataset.sequences)}[/yellow] total")
    for fam, count in dataset.family_counts.items():
        fam_table.add_row("", f"  {fam}  →  {count} sequences")
    console.print(fam_table)

    # Embedders
    embedder_names = [e["name"] for e in cfg["embedders"]]
    emb_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    emb_table.add_column(style="dim")
    emb_table.add_column(style="white")
    emb_table.add_row("Embedders", "  ".join(f"[magenta]{n}[/magenta]" for n in embedder_names))
    emb_table.add_row("Reducer", f"[green]{cfg['reducer']['name']}[/green]")
    emb_table.add_row("Output", str(cfg.get("results_dir", "results")))
    console.print(emb_table)
    console.print()


def print_results_table(results) -> None:
    console.print(Rule("[bold cyan]Benchmark Results[/bold cyan]", style="cyan"))
    console.print()

    table = Table(box=box.ROUNDED, border_style="cyan", show_lines=True)
    table.add_column("Method", style="bold white", no_wrap=True)
    table.add_column("Trustworthiness", justify="center")
    table.add_column("kNN Accuracy", justify="center")

    def score_style(val: float) -> str:
        if val >= 0.9:
            return "bold green"
        if val >= 0.7:
            return "yellow"
        return "red"

    for _, row in results.iterrows():
        trust = row["trustworthiness"]
        knn = row["knn_accuracy_mean"]
        knn_std = row["knn_accuracy_std"]
        table.add_row(
            row["method"],
            f"[{score_style(trust)}]{trust:.4f}[/]",
            f"[{score_style(knn)}]{knn:.4f}[/] [dim]±{knn_std:.4f}[/dim]",
        )

    console.print(table)
    console.print()


def print_footer(elapsed: float, results_dir: Path) -> None:
    console.print(Rule(style="cyan"))
    footer = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    footer.add_column(style="dim")
    footer.add_column(style="white")
    footer.add_row("Completed in", f"[cyan]{elapsed:.1f}s[/cyan]")
    footer.add_row("Metrics saved to", str(results_dir / "metrics" / "benchmark.csv"))
    footer.add_row("Figures saved to", str(results_dir / "figures" / ""))
    console.print(footer)
    console.print()


# ---------------------------------------------------------------------------
# Embedder / reducer builders
# ---------------------------------------------------------------------------

def _build_embedder(cfg: dict) -> SequenceEmbedder:
    name = cfg["name"]
    if name == "composition":
        from .embedders.composition import CompositionEmbedder
        return CompositionEmbedder()
    if name == "kmer":
        from .embedders.kmer import KmerEmbedder
        return KmerEmbedder(k=cfg.get("k", 3))
    if name == "esm2":
        from .embedders.esm import ESMEmbedder
        return ESMEmbedder(
            model_key=cfg.get("model_key", "esm2_t12_35M"),
            batch_size=cfg.get("batch_size", 16),
            max_len=cfg.get("max_len", 512),
        )
    if name == "llm":
        from .embedders.llm import LLMEmbedder
        return LLMEmbedder(
            provider=cfg.get("provider", "google"),
            model=cfg.get("model", None),
            batch_size=cfg.get("batch_size", 64),
            api_key_env=cfg.get("api_key_env", None),
            max_len=cfg.get("max_len", 512),
        )
    raise ValueError(f"Unknown embedder: {name!r}")


def _build_reducer(cfg: dict) -> DimReducer:
    name = cfg["name"]
    if name == "pca":
        from .reduction.reducers import PCAReducer
        return PCAReducer(n_components=cfg.get("n_components", 2))
    if name == "umap":
        from .reduction.reducers import UMAPReducer
        return UMAPReducer(
            n_neighbors=cfg.get("n_neighbors", 15),
            min_dist=cfg.get("min_dist", 0.1),
            metric=cfg.get("metric", "cosine"),
        )
    if name == "tsne":
        from .reduction.reducers import TSNEReducer
        return TSNEReducer(perplexity=cfg.get("perplexity", 30))
    raise ValueError(f"Unknown reducer: {name!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prot2Vec — Protein Sequence Embedding Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  prot2vec-benchmark --config configs/experiments/quick.yaml\n"
            "  prot2vec-benchmark --config configs/experiments/llm.yaml\n"
            "  prot2vec-benchmark --config configs/experiments/full.yaml\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        metavar="PATH",
        help="Path to experiment YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[bold red]Error:[/bold red] Config not found: {config_path}")
        raise SystemExit(1)

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    print_banner()

    # --- Data ---
    pfam_cfg = cfg["pfam"]
    with console.status("[cyan]Downloading / loading Pfam data...[/cyan]", spinner="dots"):
        seed_path = download_pfam_seed(
            version=pfam_cfg["version"],
            cache_dir=cfg.get("data", {}).get("cache_dir", "data/raw"),
        )
        records = parse_pfam_families(pfam_ids=pfam_cfg["families"], seed_path=seed_path)
        dataset = ProteinDataset.from_pfam_records(
            records,
            min_length=cfg.get("data", {}).get("min_seq_length", 50),
        )

    print_config_summary(cfg, dataset)

    embedders = [_build_embedder(e) for e in cfg["embedders"]]
    reducer = _build_reducer(cfg["reducer"])
    results_dir = Path(cfg.get("results_dir", "results"))
    n_steps = len(embedders) * 3  # embed + reduce + evaluate per embedder

    # --- Benchmark with live progress ---
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36, style="cyan", complete_style="bold cyan"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Starting...", total=n_steps)

        def on_progress(event: str, payload: dict) -> None:
            name = payload.get("name", "")
            short = name if len(name) <= 28 else name[:25] + "..."
            if event == "embed_start":
                label = "[dim](cached)[/dim]" if payload.get("cached") else ""
                progress.update(task, description=f"Embedding  [magenta]{short}[/magenta] {label}")
            elif event == "embed_done":
                progress.advance(task)
            elif event == "reduce_start":
                progress.update(task, description=f"Reducing   [magenta]{short}[/magenta]")
            elif event == "reduce_done":
                progress.advance(task)
            elif event == "evaluate_done":
                progress.update(task, description=f"Evaluating [magenta]{short}[/magenta]")
                progress.advance(task)

        run_cfg = RunConfig(
            dataset=dataset,
            embedders=embedders,
            reducer=reducer,
            results_dir=results_dir,
            cache_embeddings=cfg.get("cache_embeddings", True),
            save_figures=cfg.get("save_figures", True),
            on_progress=on_progress,
        )
        results = run(run_cfg)
        progress.update(task, description="[bold green]Done[/bold green]")

    elapsed = time.time() - start_time

    print_results_table(results)
    print_footer(elapsed, results_dir)


# ---------------------------------------------------------------------------
# Data download entry point
# ---------------------------------------------------------------------------

def download() -> None:
    """Download and cache the Pfam seed alignment (prot2vec-download)."""
    parser = argparse.ArgumentParser(
        description="Download Pfam-A.seed.gz from EBI and cache locally."
    )
    parser.add_argument("--version", default="35.0", help="Pfam release version (default: 35.0)")
    parser.add_argument(
        "--cache-dir", default="data/raw", help="Directory to save the file (default: data/raw)"
    )
    args = parser.parse_args()

    path = download_pfam_seed(version=args.version, cache_dir=args.cache_dir)
    console.print(f"[green]Pfam seed available at:[/green] {path}")


if __name__ == "__main__":
    main()
