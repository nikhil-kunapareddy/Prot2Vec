"""Command-line entry points: the benchmark runner and the Pfam downloader."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, NoReturn

import yaml
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# API keys live in .env, never in the config files.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv ships with the [llm] extra
    pass

from . import __version__
from .data.dataset import ProteinDataset
from .data.families import describe
from .data.pfam import download_pfam_seed, parse_pfam_families
from .embedders.base import SequenceEmbedder
from .pipeline import RunConfig, run
from .reduction.reducers import DimReducer

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_banner() -> None:
    """Print the Prot2Vec banner."""
    title = Text()
    title.append("PROT", style="bold cyan")
    title.append("2", style="bold yellow")
    title.append("VEC", style="bold cyan")
    title.append("\n")
    title.append("Protein Sequence Embedding Benchmark", style="dim white")
    title.append("  ·  ", style="dim")
    title.append(f"v{__version__}", style="dim cyan")

    console.print()
    console.print(Panel(Align.center(title), border_style="cyan", padding=(1, 8)))
    console.print()


def print_config_summary(cfg: dict[str, Any], dataset: ProteinDataset) -> None:
    """Print the resolved experiment configuration and dataset composition."""
    console.print(Rule("[bold cyan]Experiment Configuration[/bold cyan]", style="cyan"))
    console.print()

    summary = dataset.summary()
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="white")

    pfam_cfg = cfg.get("pfam") or {}
    if pfam_cfg:
        table.add_row("Pfam version", str(pfam_cfg.get("version", "-")))
    table.add_row("Source", dataset.source)
    table.add_row("Sequences", f"[yellow]{len(dataset)}[/yellow] total")
    for family, count in dataset.family_counts.items():
        table.add_row("", f"  [cyan]{describe(family)}[/cyan]  →  {count} sequences")
    table.add_row(
        "Length",
        f"min {summary['length_min']}  ·  median {summary['length_median']}"
        f"  ·  max {summary['length_max']} residues",
    )
    # The majority-class fraction is the accuracy a classifier gets for free,
    # so every kNN number below should be read against it.
    table.add_row(
        "Chance baseline",
        f"[yellow]{summary['majority_class_fraction']:.3f}[/yellow] "
        "[dim](majority-class fraction)[/dim]",
    )
    table.add_row(
        "Embedders",
        "  ".join(f"[magenta]{e['name']}[/magenta]" for e in cfg["embedders"]),
    )
    table.add_row(
        "Reducers",
        "  ".join(f"[green]{r['name']}[/green]" for r in _reducer_configs(cfg)),
    )
    table.add_row("Output", str(cfg.get("results_dir", "results")))
    console.print(table)
    console.print()


def _score_style(value: float) -> str:
    """Map a 0-1 score to a colour."""
    if value >= 0.9:
        return "bold green"
    if value >= 0.7:
        return "yellow"
    return "red"


def print_results_table(results: Any, baseline: float | None = None) -> None:
    """Print the benchmark results, one row per (embedder, reducer) pair.

    Only the headline metrics are shown; ``benchmark.csv`` carries every
    column. The best row by full-dimensional k-NN accuracy is marked, since
    that is the metric most runs are actually trying to compare.

    Parameters
    ----------
    results
        Benchmark results DataFrame as returned by :func:`prot2vec.run`.
    baseline
        Majority-class fraction, shown as a reference so the k-NN and
        retrieval scores can be read against what chance already achieves.
    """
    console.print(Rule("[bold cyan]Benchmark Results[/bold cyan]", style="cyan"))
    console.print()

    # (column header, results column, justify) for the columns present.
    candidates = [
        ("kNN-full", "knn_accuracy_highdim_mean", "knn_accuracy_highdim_std"),
        ("P@k", "precision_at_k", None),
        ("Silh", "silhouette", None),
        ("ARI", "adjusted_rand", None),
        ("Trust", "trustworthiness", None),
        ("kNN-2D", "knn_accuracy_mean", None),
    ]
    columns = [c for c in candidates if c[1] in results.columns]

    best_row = None
    for _, column, _ in columns:
        if column == "knn_accuracy_highdim_mean":
            best_row = results[column].idxmax()
            break

    table = Table(box=box.ROUNDED, border_style="cyan")
    table.add_column("Method", style="bold white", no_wrap=True)
    for header, _, _ in columns:
        table.add_column(header, justify="center")

    for index, row in results.iterrows():
        marker = " [bold yellow]*[/bold yellow]" if index == best_row else ""
        cells = [f"{row['method']}{marker}"]
        for _, column, std_column in columns:
            value = row[column]
            # Silhouette and ARI are meaningful below zero, so they are not
            # colour-graded on the same 0-1 scale as the accuracies.
            style = _score_style(value) if column not in ("silhouette", "adjusted_rand") else ""
            rendered = f"[{style}]{value:.3f}[/]" if style else f"{value:.3f}"
            if std_column and std_column in results.columns:
                rendered += f" [dim]±{row[std_column]:.3f}[/dim]"
            cells.append(rendered)
        table.add_row(*cells)

    console.print(table)
    if baseline is not None:
        console.print(
            f"  [dim]Chance baseline (majority class): {baseline:.3f} — k-NN and P@k "
            "at or below this are no better than guessing.[/dim]"
        )
    console.print(
        "  [dim]* best by full-dimensional k-NN.  Full metric set: metrics/benchmark.csv[/dim]"
    )
    console.print()


def print_footer(elapsed: float, results_dir: Path) -> None:
    """Print timings and output locations."""
    console.print(Rule(style="cyan"))
    footer = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    footer.add_column(style="dim")
    footer.add_column(style="white")
    footer.add_row("Completed in", f"[cyan]{elapsed:.1f}s[/cyan]")
    footer.add_row("Metrics", str(results_dir / "metrics" / "benchmark.csv"))
    footer.add_row("Figures", str(results_dir / "figures"))
    footer.add_row("Embeddings", str(results_dir / "embeddings"))
    footer.add_row("Projections", str(results_dir / "projections"))
    footer.add_row("Manifest", str(results_dir / "run_manifest.json"))
    console.print(footer)
    console.print()


def _fail(message: str, hint: str | None = None) -> NoReturn:
    """Print an error panel and exit non-zero.

    A benchmark fails for mundane, fixable reasons — a missing API key, a
    broken UMAP install, a family that is not in the pinned release — and a
    stack trace buries the one line that says which. The traceback is still
    available at ``--log-level debug``.
    """
    body = Text(message)
    if hint:
        body.append("\n\n")
        body.append(hint, style="dim")
    console.print(Panel(body, title="[bold red]Error[/bold red]", border_style="red"))
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Config -> objects
# ---------------------------------------------------------------------------


def _build_embedder(cfg: dict[str, Any]) -> SequenceEmbedder:
    """Instantiate one embedder from its config block.

    Raises
    ------
    ValueError
        If ``cfg["name"]`` is not a known embedder.
    """
    name = cfg["name"]
    if name == "composition":
        from .embedders.composition import CompositionEmbedder

        return CompositionEmbedder()
    if name == "kmer":
        from .embedders.kmer import KmerEmbedder

        return KmerEmbedder(k=cfg.get("k", 3), min_df=cfg.get("min_df", 1))
    if name == "esm2":
        from .embedders.esm import ESMEmbedder

        return ESMEmbedder(
            model_key=cfg.get("model_key", "esm2_t12_35M"),
            device=cfg.get("device"),
            batch_size=cfg.get("batch_size", 16),
            max_len=cfg.get("max_len", 512),
        )
    if name == "llm":
        from .embedders.llm import LLMEmbedder

        return LLMEmbedder(
            provider=cfg.get("provider", "google"),
            model=cfg.get("model"),
            batch_size=cfg.get("batch_size", 32),
            api_key_env=cfg.get("api_key_env"),
            max_len=cfg.get("max_len", 512),
            output_dim=cfg.get("output_dim"),
        )
    raise ValueError(f"Unknown embedder {name!r}. Choose from: composition, kmer, esm2, llm.")


def _reducer_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the reducer blocks, accepting either ``reducer`` or ``reducers``.

    A single ``reducer:`` mapping stays valid because most experiments compare
    embedders at a fixed projection; ``reducers:`` takes a list for the case
    where the projection itself is what you are comparing.
    """
    if cfg.get("reducers"):
        reducers = cfg["reducers"]
        return list(reducers) if isinstance(reducers, list) else [reducers]
    return [cfg["reducer"]]


def _build_reducer(cfg: dict[str, Any]) -> DimReducer:
    """Instantiate one reducer from its config block.

    Raises
    ------
    ValueError
        If ``cfg["name"]`` is not a known reducer.
    """
    name = cfg["name"]
    if name == "pca":
        from .reduction.reducers import PCAReducer

        return PCAReducer(n_components=cfg.get("n_components", 2))
    if name == "umap":
        from .reduction.reducers import UMAPReducer

        return UMAPReducer(
            n_components=cfg.get("n_components", 2),
            n_neighbors=cfg.get("n_neighbors", 15),
            min_dist=cfg.get("min_dist", 0.1),
            metric=cfg.get("metric", "cosine"),
        )
    if name == "tsne":
        from .reduction.reducers import TSNEReducer

        return TSNEReducer(
            n_components=cfg.get("n_components", 2),
            perplexity=cfg.get("perplexity", 30),
        )
    raise ValueError(f"Unknown reducer {name!r}. Choose from: pca, umap, tsne.")


def _load_config(path: Path) -> dict[str, Any]:
    """Read and validate an experiment YAML file.

    Raises
    ------
    SystemExit
        With a rendered error panel if the file is missing, unparseable, or
        lacks a required section.
    """
    if not path.exists():
        _fail(
            f"Config not found: {path}",
            "Ready-made experiments live in configs/experiments/ — try "
            "configs/experiments/quick.yaml.",
        )
    try:
        cfg = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        _fail(f"Could not parse {path}:\n{exc}")

    if not isinstance(cfg, dict):
        _fail(f"{path} must contain a YAML mapping at the top level.")

    if not cfg.get("embedders"):
        _fail(
            f"{path} is missing the required 'embedders' section.",
            "configs/default.yaml documents every option.",
        )
    if not cfg.get("reducer") and not cfg.get("reducers"):
        _fail(
            f"{path} must define a 'reducer' mapping or a 'reducers' list.",
            "configs/default.yaml documents every option.",
        )
    if not cfg.get("pfam") and not cfg.get("fasta"):
        _fail(
            f"{path} must define either a 'pfam' or a 'fasta' data source.",
            "See configs/default.yaml for both forms.",
        )
    return cfg


def _load_dataset(cfg: dict[str, Any]) -> ProteinDataset:
    """Build the dataset described by ``cfg``, from FASTA or from Pfam."""
    data_cfg = cfg.get("data") or {}
    min_length = data_cfg.get("min_seq_length", 50)
    max_per_family = data_cfg.get("max_per_family")
    random_state = data_cfg.get("random_state", 0)

    fasta_cfg = cfg.get("fasta")
    if fasta_cfg:
        path = fasta_cfg["path"] if isinstance(fasta_cfg, dict) else fasta_cfg
        label_from = (
            fasta_cfg.get("label_from", "first_token")
            if isinstance(fasta_cfg, dict)
            else "first_token"
        )
        with console.status(f"[cyan]Reading {path}...[/cyan]", spinner="dots"):
            return ProteinDataset.from_fasta(
                path,
                label_from=label_from,
                min_length=min_length,
                max_per_family=max_per_family,
                random_state=random_state,
            )

    pfam_cfg = cfg["pfam"]
    with console.status("[cyan]Downloading / loading Pfam data...[/cyan]", spinner="dots"):
        seed_path = download_pfam_seed(
            version=str(pfam_cfg["version"]),
            cache_dir=data_cfg.get("cache_dir", "data/raw"),
        )
        records = parse_pfam_families(
            pfam_ids=[str(f) for f in pfam_cfg["families"]], seed_path=seed_path
        )
    return ProteinDataset.from_pfam_records(
        records,
        min_length=min_length,
        max_per_family=max_per_family,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# prot2vec-benchmark
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the benchmark argument parser."""
    parser = argparse.ArgumentParser(
        prog="prot2vec-benchmark",
        description="Prot2Vec — benchmark protein sequence embeddings on family separability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  prot2vec-benchmark --config configs/experiments/quick.yaml\n"
            "  prot2vec-benchmark --config configs/experiments/full.yaml\n"
            "  prot2vec-benchmark --config configs/experiments/llm.yaml\n"
            "\n"
            "Experiments are defined in YAML, not in flags — copy a file from\n"
            "configs/experiments/ and edit it to describe a new comparison.\n"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        metavar="PATH",
        help="experiment YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="logging verbosity (default: warning)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore cached embeddings and recompute from scratch",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip writing figures")
    parser.add_argument("--version", action="version", version=f"prot2vec {__version__}")
    return parser


def main() -> None:
    """Run a benchmark described by an experiment config (``prot2vec-benchmark``)."""
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s"
    )
    debug = args.log_level == "debug"

    cfg = _load_config(Path(args.config))
    print_banner()

    try:
        dataset = _load_dataset(cfg)
    except Exception as exc:
        if debug:
            raise
        _fail(f"{type(exc).__name__}: {exc}")

    if len(dataset) == 0:
        _fail(
            "No sequences survived loading.",
            "Lower 'min_seq_length', or check the requested families exist in this Pfam release.",
        )
    if len(dataset.families) < 2:
        _fail(
            f"Only one group ({dataset.families[0]}) has sequences, so family "
            "separability cannot be measured.",
            "Request at least two families with populated seed alignments.",
        )

    print_config_summary(cfg, dataset)

    try:
        embedders = [_build_embedder(e) for e in cfg["embedders"]]
        reducers = [_build_reducer(r) for r in _reducer_configs(cfg)]
    except (ValueError, KeyError) as exc:
        _fail(str(exc), "configs/default.yaml documents every option.")

    results_dir = Path(cfg.get("results_dir", "results"))
    metric_params = dict(cfg.get("metrics") or {})
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36, style="cyan", complete_style="bold cyan"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        run_cfg = RunConfig(
            dataset=dataset,
            embedders=embedders,
            reducers=reducers,
            results_dir=results_dir,
            cache_embeddings=cfg.get("cache_embeddings", True) and not args.no_cache,
            save_figures=cfg.get("save_figures", True) and not args.no_figures,
            export_embeddings=cfg.get("export_embeddings", True),
            metric_params=metric_params,
        )
        task = progress.add_task("Starting...", total=run_cfg.n_steps)

        def on_progress(event: str, payload: dict[str, Any]) -> None:
            name = str(payload.get("name", ""))
            reducer_name = str(payload.get("reducer", ""))
            short = name if len(name) <= 24 else name[:21] + "..."
            if event == "embed_start":
                label = "[dim](cached)[/dim]" if payload.get("cached") else ""
                progress.update(task, description=f"Embedding  [magenta]{short}[/magenta] {label}")
            elif event == "embed_done":
                progress.advance(task)
            elif event == "reduce_start":
                progress.update(
                    task,
                    description=(
                        f"Reducing   [magenta]{short}[/magenta] [green]{reducer_name}[/green]"
                    ),
                )
            elif event == "reduce_done":
                progress.advance(task)
            elif event == "evaluate_done":
                progress.update(
                    task,
                    description=(
                        f"Scoring    [magenta]{short}[/magenta] [green]{reducer_name}[/green]"
                    ),
                )
                progress.advance(task)

        run_cfg.on_progress = on_progress
        try:
            results = run(run_cfg)
        except ImportError as exc:
            if debug:
                raise
            _fail(str(exc))
        except (RuntimeError, ValueError) as exc:
            if debug:
                raise
            _fail(f"{type(exc).__name__}: {exc}", "Re-run with --log-level debug for a traceback.")
        progress.update(task, description="[bold green]Done[/bold green]")

    elapsed = time.time() - start_time
    baseline = float(dataset.summary()["majority_class_fraction"])  # type: ignore[arg-type]
    print_results_table(results, baseline=baseline)
    print_footer(elapsed, results_dir)


# ---------------------------------------------------------------------------
# prot2vec-download
# ---------------------------------------------------------------------------


def download() -> None:
    """Download and cache a Pfam seed alignment (``prot2vec-download``)."""
    parser = argparse.ArgumentParser(
        prog="prot2vec-download",
        description="Download Pfam-A.seed.gz from the EBI mirror and cache it locally.",
    )
    parser.add_argument("--version", default="35.0", help="Pfam release (default: 35.0)")
    parser.add_argument(
        "--cache-dir", default="data/raw", help="destination directory (default: data/raw)"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download even if a valid cache exists"
    )
    args = parser.parse_args()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Pfam {task.fields[release]}"),
        BarColumn(bar_width=36, style="cyan", complete_style="bold cyan"),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("download", total=None, release=args.version)

        def on_progress(done: int, total: int | None) -> None:
            progress.update(task, completed=done, total=total)

        try:
            path = download_pfam_seed(
                version=args.version,
                cache_dir=args.cache_dir,
                force=args.force,
                on_progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary: render, don't traceback
            _fail(f"{type(exc).__name__}: {exc}")

    size_mb = path.stat().st_size / 1e6
    console.print(f"[green]Pfam seed ready:[/green] {path} [dim]({size_mb:.0f} MB)[/dim]")


if __name__ == "__main__":
    main()
