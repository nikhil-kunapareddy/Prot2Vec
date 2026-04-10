# Prot2Vec

```
╭─────────────────────────────────────────────╮
│                                             │
│         PROT2VEC                            │
│         Protein Sequence Embedding          │
│         Benchmark  ·  v0.1.0               │
│                                             │
╰─────────────────────────────────────────────╯
```

A benchmarking toolkit for protein sequence vectorization techniques. Compares embedding methods — amino acid composition, k-mer TF-IDF, transformer-based protein language models (ESM-2), and general-purpose LLM embedders (Google Gemini) — on their ability to capture structural and functional similarity across Pfam protein families.

## Overview

Proteins are fundamental to biological systems, but representing them in a way that computational models can interpret is non-trivial. This project evaluates multiple embedding strategies across core bioinformatics tasks and measures them with quantitative metrics:

- **Trustworthiness** — how well local neighborhoods in high-dimensional space are preserved in 2D projections
- **5-NN cross-validated accuracy** — how separable protein families are in embedding space

## Methods Compared

| Method | Type | Dimensionality |
|---|---|---|
| Amino acid composition | Frequency vector | 20-d |
| k-mer TF-IDF (`k=3`) | Sparse bag-of-ngrams | ~8000-d |
| ESM-2 (35M parameters) | Protein language model | 480-d |
| Google Gemini (`gemini-embedding-001`) | General-purpose LLM | 768-d |

Dimensionality reduction is applied via **PCA**, **UMAP**, or **t-SNE** before evaluation and visualization.

## Repository Structure

```
prot2vec/
├── prot2vec/               # Core package
│   ├── data/               # Pfam download + parsing, ProteinDataset
│   ├── embedders/          # Composition, k-mer, ESM-2, LLM embedders
│   ├── reduction/          # PCA, UMAP, t-SNE wrappers
│   ├── evaluation/         # Trustworthiness + kNN accuracy metrics
│   ├── visualization/      # Scatter plots + metric bar charts
│   └── pipeline.py         # Orchestrates embed → reduce → evaluate
├── configs/                # YAML experiment configs
│   ├── default.yaml
│   └── experiments/
│       ├── quick.yaml      # 2 families, no ESM (fast iteration)
│       ├── full.yaml       # 5 families, all embedders
│       └── llm.yaml        # Google Gemini LLM embedder benchmark
├── scripts/
│   ├── download_data.py    # Pre-fetch Pfam seed alignment
│   └── run_benchmark.py    # CLI entry point
├── notebooks/
└── tests/
```

## Installation

Requires Python ≥ 3.11.

```bash
git clone https://github.com/<your-username>/prot2vec.git
cd prot2vec

python -m venv .venv
source .venv/bin/activate

# Base install (composition + k-mer only)
pip install -e ".[dev]"

# With ESM-2 support (requires PyTorch)
pip install -e ".[esm,dev]"

# With LLM embedder support (Google Gemini)
pip install -e ".[llm,dev]"

# Or install all dependencies at once
pip install -r requirements.txt
```

## Quick Start

### Run from CLI

```bash
# Download Pfam seed alignment (~500 MB, cached after first run)
python scripts/download_data.py --version 35.0 --cache-dir data/raw

# Quick benchmark: composition + k-mer on 2 families (~30 seconds)
python scripts/run_benchmark.py --config configs/experiments/quick.yaml

# Full benchmark: all embedders on 5 families
python scripts/run_benchmark.py --config configs/experiments/full.yaml

# LLM benchmark: Google Gemini vs. protein-specific methods
python scripts/run_benchmark.py --config configs/experiments/llm.yaml
```

On launch you will see a live CLI UI:

```
 ╭──────────────────────────────────────╮
 │                                      │
 │      PROT2VEC                        │
 │      Protein Sequence Embedding      │
 │      Benchmark  ·  v0.1.0           │
 │                                      │
 ╰──────────────────────────────────────╯

 ──────────────── Experiment Configuration ────────────────

   Pfam version   35.0
   Families       PF00069   PF00072
   Sequences      90 total
                    PF00069  →  38 sequences
                    PF00072  →  52 sequences
   Embedders      composition   kmer   llm
   Reducer        umap
   Output         results

 ⠸ Embedding  llm_google_gemini-embedding-001  ━━━━━━━╺━━━━  5/9  0:00:12

 ──────────────────── Benchmark Results ───────────────────

 ╭─────────────────────────────────────────────────────────╮
 │ Method                  │ Trustworthiness │ kNN Accuracy │
 ├─────────────────────────┼─────────────────┼──────────────┤
 │ composition+umap        │    0.8980       │ 0.9444 ±0.05 │
 │ kmer_k3+umap            │    0.7986       │ 1.0000 ±0.00 │
 │ llm_google_gemini+umap  │    0.8426       │ 0.5667 ±0.10 │
 ╰─────────────────────────┴─────────────────┴──────────────╯

 ──────────────────────────────────────────────────────────
   Completed in    14.3s
   Metrics saved   results/metrics/benchmark.csv
   Figures saved   results/figures/
```

Results are written to `results/metrics/benchmark.csv` and figures to `results/figures/`.

### Use as a library

```python
from prot2vec.data.pfam import download_pfam_seed, parse_pfam_families
from prot2vec.data.dataset import ProteinDataset
from prot2vec.embedders.composition import CompositionEmbedder
from prot2vec.embedders.kmer import KmerEmbedder
from prot2vec.embedders.llm import LLMEmbedder
from prot2vec.reduction.reducers import UMAPReducer
from prot2vec.pipeline import RunConfig, run

seed_path = download_pfam_seed(version="35.0", cache_dir="data/raw")
records = parse_pfam_families(["PF00069", "PF00072"], seed_path)
dataset = ProteinDataset.from_pfam_records(records)

config = RunConfig(
    dataset=dataset,
    embedders=[
        CompositionEmbedder(),
        KmerEmbedder(k=3),
        LLMEmbedder(provider="google"),
    ],
    reducer=UMAPReducer(n_neighbors=15, metric="cosine"),
)
results = run(config)
print(results)
```

## Configuration

Experiments are defined as YAML files. The full set of options:

```yaml
pfam:
  version: "35.0"
  families: [PF00069, PF00072]

data:
  cache_dir: data/raw
  min_seq_length: 50

embedders:
  - name: composition
  - name: kmer
    k: 3
  - name: esm2
    model_key: esm2_t12_35M   # esm2_t6_8M | esm2_t12_35M | esm2_t30_150M
    batch_size: 16
    max_len: 512
  - name: llm
    provider: google            # currently supported: google
    model: gemini-embedding-001
    batch_size: 64
    max_len: 512

reducer:
  name: umap                  # pca | umap | tsne
  n_neighbors: 15
  min_dist: 0.1
  metric: cosine

results_dir: results
cache_embeddings: true        # skip re-embedding if .npy cache exists
save_figures: true
```

## LLM Embedder Setup

The LLM embedder sends protein sequences as plain text to a general-purpose embedding API, benchmarking whether non-biological models can incidentally capture protein family structure.

**Google Gemini** (`gemini-embedding-001`, 768-d):
- Get a free API key at [aistudio.google.com](https://aistudio.google.com) (1,500 requests/day free)
- Add to a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

API keys are loaded automatically from `.env` at runtime.

## Running Tests

```bash
pytest
```

Tests cover the data layer, all embedders, and all evaluation metrics. ESM-2 is tested separately (requires `[esm]` install).

## Hardware Notes

ESM-2 inference auto-detects the best available device:

- **Apple Silicon (MPS)**: detected automatically, fp16 autocast applied
- **NVIDIA GPU (CUDA)**: detected automatically
- **CPU fallback**: works on all hardware; use `esm2_t6_8M` for speed

## Roadmap

- [x] Amino acid composition baseline
- [x] k-mer TF-IDF embeddings
- [x] ESM-2 protein language model
- [x] General-purpose LLM embedder (Google Gemini)
- [x] Rich CLI UI with banner, progress bars, and results table
- [ ] ProtTrans / Ankh embedder support
- [ ] Silhouette score metric
- [ ] Benchmark on full Pfam database
- [ ] Nearest-neighbour retrieval evaluation

## Data

Protein families are sourced from [Pfam](https://www.ebi.ac.uk/training/online/courses/pfam-creating-protein-families/) (Pfam 35.0, EBI FTP). The seed alignment file is downloaded on first run and cached locally.

## License

[MIT](LICENSE)
