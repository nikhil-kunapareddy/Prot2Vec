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
from .evaluation import resolve_groups
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
    table.add_row("Alphabet", f"[cyan]{dataset.alphabet}[/cyan]")
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

    Deliberately narrow. A full run can produce sixty rows and thirty-one
    metrics, which no terminal renders legibly, so five columns are shown and
    ``benchmark.csv`` carries everything. The five chosen are the ones that
    answer different questions: two k-NN scores plus trustworthiness separate
    "the representation is bad" from "the projection lost it", and the
    retrieval pair says whether a neighbour lookup would actually work.

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

    candidates = [
        ("kNN-hi", "knn_accuracy_highdim_mean"),
        ("kNN-2D", "knn_accuracy_mean"),
        ("P@k", "precision_at_k"),
        ("AUROC", "same_class_auroc"),
        ("Trust", "trustworthiness"),
    ]
    columns = [c for c in candidates if c[1] in results.columns]

    best_row = None
    for _, column in columns:
        if column == "knn_accuracy_highdim_mean":
            best_row = results[column].idxmax()
            break

    table = Table(box=box.ROUNDED, border_style="cyan")
    table.add_column("Method", style="bold white", no_wrap=True)
    for header, _ in columns:
        table.add_column(header, justify="right")

    for index, row in results.iterrows():
        method = str(row["method"])
        if len(method) > 24:
            method = method[:21] + "..."
        marker = " [bold yellow]*[/bold yellow]" if index == best_row else ""
        cells = [f"{method}{marker}"]
        for _, column in columns:
            value = float(row[column])
            cells.append(f"[{_score_style(value)}]{value:.3f}[/]")
        table.add_row(*cells)

    console.print(table)
    if baseline is not None:
        console.print(
            f"  [dim]Chance baseline (majority class): {baseline:.3f} - k-NN and P@k "
            "at or below this are no better than guessing.[/dim]"
        )
    console.print(
        "  [dim]* best by full-dimensional k-NN.  All metrics: metrics/benchmark.csv[/dim]"
    )
    _warn_about_confounds(results)
    console.print()


def _warn_about_confounds(results: Any) -> None:
    """Flag results that sequence length or composition already explains.

    The easiest way to get a good-looking embedding result is to accidentally
    measure something trivial, and the columns that reveal it are buried among
    thirty others. Surfacing them next to the table means the caveat arrives
    with the result rather than after publication.
    """
    warnings: list[str] = []

    if "length_only_knn_accuracy" in results.columns:
        length_only = float(results["length_only_knn_accuracy"].iloc[0])
        column = (
            "knn_accuracy_highdim_mean"
            if "knn_accuracy_highdim_mean" in results.columns
            else "knn_accuracy_mean"
        )
        if column in results.columns:
            best = float(results[column].max())
            if length_only >= best - 0.05:
                warnings.append(
                    f"sequence length alone scores {length_only:.3f} k-NN accuracy "
                    f"against {best:.3f} for the best embedding - these groups may "
                    "differ mainly in length"
                )

    if "length_distance_rho" in results.columns:
        worst = results.loc[results["length_distance_rho"].abs().idxmax()]
        rho = float(worst["length_distance_rho"])
        if abs(rho) >= 0.5:
            warnings.append(
                f"{worst['method']} has a length/distance rho of {rho:+.2f} - much "
                "of what it encodes is sequence length"
            )

    if "composition_distance_rho" in results.columns and "embedder" in results.columns:
        # Composition-based embedders correlate with composition by
        # construction; the finding is only interesting for learned ones.
        learned = results[~results["embedder"].isin(("composition", "dipeptide"))]
        if not learned.empty:
            worst = learned.loc[learned["composition_distance_rho"].abs().idxmax()]
            rho = float(worst["composition_distance_rho"])
            if abs(rho) >= 0.9:
                warnings.append(
                    f"{worst['method']} has a composition/distance rho of {rho:+.2f}"
                    " - it may be re-encoding residue frequencies"
                )

    for message in warnings:
        console.print(f"  [yellow]![/yellow] [dim]{message}[/dim]")


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


def _build_embedder(cfg: dict[str, Any], alphabet: str = "protein") -> SequenceEmbedder:
    """Instantiate one embedder from its config block.

    Parameters
    ----------
    cfg
        The embedder's config mapping; ``cfg["name"]`` selects the class.
    alphabet
        Dataset alphabet, passed to the embedders that are alphabet-aware so
        that a DNA run does not silently use amino acid columns.

    Raises
    ------
    ValueError
        If ``cfg["name"]`` is not a known embedder.
    """
    name = cfg["name"]

    if name == "composition":
        from .embedders.composition import CompositionEmbedder

        return CompositionEmbedder()
    if name == "dipeptide":
        from .embedders.dipeptide import DipeptideEmbedder

        return DipeptideEmbedder(alphabet=cfg.get("alphabet", alphabet))
    if name == "physicochemical":
        from .embedders.physicochemical import PhysicochemicalEmbedder

        # Passed explicitly so a nucleotide run is refused rather than silently
        # scored with amino acid scales.
        return PhysicochemicalEmbedder(alphabet=cfg.get("alphabet", alphabet))
    if name == "ctd":
        from .embedders.ctd import CTDEmbedder

        return CTDEmbedder(alphabet=cfg.get("alphabet", alphabet))
    if name == "kmer":
        from .embedders.kmer import KmerEmbedder

        return KmerEmbedder(k=cfg.get("k", 3), min_df=cfg.get("min_df", 1))
    if name == "onehot":
        from .embedders.onehot import OneHotEmbedder

        return OneHotEmbedder(
            max_len=cfg.get("max_len", 256),
            alphabet=cfg.get("alphabet", alphabet),
            truncate=cfg.get("truncate", "center"),
        )
    if name == "esm2":
        from .embedders.esm import ESMEmbedder

        return ESMEmbedder(
            model_key=cfg.get("model_key", "esm2_t12_35M"),
            device=cfg.get("device"),
            batch_size=cfg.get("batch_size", 16),
            max_len=cfg.get("max_len", 512),
        )
    if name in ("hf", "huggingface"):
        from .embedders.huggingface import HuggingFaceEmbedder

        return HuggingFaceEmbedder(
            model_name=cfg.get("model", "protbert"),
            pooling=cfg.get("pooling", "mean"),
            layer=cfg.get("layer", -1),
            batch_size=cfg.get("batch_size", 8),
            max_len=cfg.get("max_len", 1024),
            device=cfg.get("device"),
            trust_remote_code=cfg.get("trust_remote_code", False),
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

    raise ValueError(
        f"Unknown embedder {name!r}. Choose from: composition, dipeptide, "
        "physicochemical, ctd, kmer, onehot, esm2, hf, llm."
    )


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
    components = cfg.get("n_components", 2)

    if name == "pca":
        from .reduction.reducers import PCAReducer

        return PCAReducer(n_components=components)
    if name == "svd":
        from .reduction.reducers import TruncatedSVDReducer

        return TruncatedSVDReducer(n_components=components)
    if name == "nmf":
        from .reduction.reducers import NMFReducer

        return NMFReducer(n_components=components, max_iter=cfg.get("max_iter", 500))
    if name == "random_projection":
        from .reduction.reducers import RandomProjectionReducer

        return RandomProjectionReducer(n_components=components)
    if name == "umap":
        from .reduction.reducers import UMAPReducer

        return UMAPReducer(
            n_components=components,
            n_neighbors=cfg.get("n_neighbors", 15),
            min_dist=cfg.get("min_dist", 0.1),
            metric=cfg.get("metric", "cosine"),
        )
    if name == "tsne":
        from .reduction.reducers import TSNEReducer

        return TSNEReducer(n_components=components, perplexity=cfg.get("perplexity", 30))
    if name == "isomap":
        from .reduction.reducers import IsomapReducer

        return IsomapReducer(n_components=components, n_neighbors=cfg.get("n_neighbors", 10))
    if name == "mds":
        from .reduction.reducers import MDSReducer

        return MDSReducer(n_components=components, n_init=cfg.get("n_init", 4))
    if name == "spectral":
        from .reduction.reducers import SpectralReducer

        return SpectralReducer(n_components=components, n_neighbors=cfg.get("n_neighbors", 10))
    if name == "lle":
        from .reduction.reducers import LLEReducer

        return LLEReducer(n_components=components, n_neighbors=cfg.get("n_neighbors", 10))
    if name == "kernel_pca":
        from .reduction.reducers import KernelPCAReducer

        return KernelPCAReducer(
            n_components=components,
            kernel=cfg.get("kernel", "cosine"),
            gamma=cfg.get("gamma"),
        )
    if name == "phate":
        from .reduction.reducers import PHATEReducer

        return PHATEReducer(
            n_components=components,
            knn=cfg.get("knn", 5),
            decay=cfg.get("decay", 40),
        )
    if name == "pacmap":
        from .reduction.reducers import PaCMAPReducer

        return PaCMAPReducer(
            n_components=components,
            n_neighbors=cfg.get("n_neighbors", 10),
            mn_ratio=cfg.get("mn_ratio", 0.5),
            fp_ratio=cfg.get("fp_ratio", 2.0),
        )

    raise ValueError(
        f"Unknown reducer {name!r}. Choose from: pca, svd, nmf, "
        "random_projection, umap, tsne, isomap, mds, spectral, lle, "
        "kernel_pca, phate, pacmap."
    )


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
    alphabet = data_cfg.get("alphabet", "protein")

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
                alphabet=alphabet,
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
        alphabet=alphabet,
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
        embedders = [_build_embedder(e, dataset.alphabet) for e in cfg["embedders"]]
        reducers = [_build_reducer(r) for r in _reducer_configs(cfg)]
    except (ValueError, KeyError) as exc:
        _fail(str(exc), "configs/default.yaml documents every option.")

    results_dir = Path(cfg.get("results_dir", "results"))
    metric_params = dict(cfg.get("metrics") or {})
    # `groups:` reads better in YAML than `metric_groups:`, but the keyword on
    # evaluate() has to be unambiguous about what it groups.
    if "groups" in metric_params:
        metric_params["metric_groups"] = metric_params.pop("groups")
    try:
        resolve_groups(metric_params.get("metric_groups"))
    except ValueError as exc:
        _fail(str(exc), "See the `metrics:` block in configs/default.yaml.")
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
        skipped_pairs: list[tuple[str, str]] = []

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
            elif event == "pair_skipped":
                # A reduce-stage skip never reached reduce_done, so two steps
                # are outstanding; an evaluate-stage skip leaves one.
                progress.advance(task, 2 if payload.get("stage") == "reduce" else 1)
                skipped_pairs.append((f"{name}+{reducer_name}", str(payload.get("reason", ""))))

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
    print_skipped_pairs(skipped_pairs)
    print_footer(elapsed, results_dir)


def print_skipped_pairs(skipped: list[tuple[str, str]]) -> None:
    """Report pairs that could not be scored, and why.

    Some combinations are simply invalid -- NMF cannot factorise a signed
    embedding, a manifold method can fail on a disconnected neighbour graph.
    The run continues without them, but a quietly missing row in a benchmark
    table invites the wrong conclusion, so each omission is named here.
    """
    if not skipped:
        return
    console.print(f"  [yellow]{len(skipped)} pair(s) skipped:[/yellow]")
    for method, reason in skipped:
        first_line = reason.strip().splitlines()[0]
        trimmed = first_line if len(first_line) <= 96 else first_line[:93] + "..."
        console.print(f"    [dim]{method}[/dim] - {trimmed}")
    console.print()


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
