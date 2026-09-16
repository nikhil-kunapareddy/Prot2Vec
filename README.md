# Prot2Vec

**Which vector representation actually separates your protein families?**

[![CI](https://github.com/nikhil-kunapareddy/Prot2Vec/actions/workflows/ci.yml/badge.svg)](https://github.com/nikhil-kunapareddy/Prot2Vec/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)

Prot2Vec is a benchmark harness for protein sequence embeddings. You give it
sequences with group labels; it runs a matrix of embedding methods against a
matrix of dimensionality reducers and scores every pair with the same metrics,
so the comparison is actually apples to apples.

It exists because "we embedded our proteins and the families clustered nicely"
is not a result until you know what a *trivial* representation would have
scored, whether the clusters survive outside the 2-D picture, and whether the
projection invented them.

```bash
pip install -e ".[dev]"
prot2vec-benchmark --config configs/experiments/quick.yaml
```

---

## Contents

- [What you get](#what-you-get)
- [Install](#install)
- [Quick start](#quick-start)
- [Benchmark your own sequences](#benchmark-your-own-sequences)
- [The methods](#the-methods)
- [The metrics, and how to read them](#the-metrics-and-how-to-read-them)
- [What a run produces](#what-a-run-produces)
- [Configuration](#configuration)
- [Library use](#library-use)
- [Extending Prot2Vec](#extending-prot2vec)
- [Hardware and cost](#hardware-and-cost)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [Citation](#citation)

---

## What you get

| | |
|---|---|
| **Four representations** | amino acid composition, k-mer TF-IDF, ESM-2 protein language models, general-purpose LLM embedding APIs |
| **Three projections** | PCA, UMAP, t-SNE — compared on identical vectors |
| **Seven metrics** | family separability in full and reduced space, homology-retrieval precision, silhouette, unsupervised cluster agreement, projection trustworthiness |
| **Two data sources** | curated Pfam seed alignments, or your own FASTA |
| **Honest baselines** | every score is printed next to the majority-class fraction, so "0.62 accuracy" can be read as what it is |
| **Usable outputs** | embeddings, 2-D coordinates and the exact input sequences, all id-aligned as plain `.npy`/`.tsv`, plus a provenance manifest |

Embeddings are computed **once per method** and reused across every reducer, so
adding a second or third projection to a run costs seconds rather than another
ESM-2 pass.

## Install

Python ≥ 3.11.

```bash
git clone https://github.com/nikhil-kunapareddy/Prot2Vec.git
cd Prot2Vec

python -m venv .venv && source .venv/bin/activate

pip install -e ".[dev]"              # composition + k-mer + all metrics
pip install -e ".[esm,dev]"          # + ESM-2 (pulls in PyTorch, ~2 GB)
pip install -e ".[llm,dev]"          # + Google Gemini embedding API
pip install -e ".[esm,llm,dev]"      # everything
```

The base install has no deep-learning dependency — composition, k-mer, all
three reducers and all metrics work without PyTorch.

## Quick start

```bash
# 1. Fetch the Pfam seed alignments (~500 MB, cached; resumable and verified)
prot2vec-download --version 35.0 --cache-dir data/raw

# 2. Two families, two cheap embedders — about a minute
prot2vec-benchmark --config configs/experiments/quick.yaml

# 3. Five families, including ESM-2
prot2vec-benchmark --config configs/experiments/full.yaml

# 4. One embedder, three projections — is the picture real?
prot2vec-benchmark --config configs/experiments/reducers.yaml
```

Example output (your numbers will differ):

```
╭──────────────────────────────────────────────────────────────────────────────╮
│               PROT2VEC                                                       │
│               Protein Sequence Embedding Benchmark  ·  v0.2.0                │
╰──────────────────────────────────────────────────────────────────────────────╯

─────────────────────────── Experiment Configuration ───────────────────────────

   Pfam version        35.0
   Source              pfam:PF00069,PF00072
   Sequences           90 total
                         PF00069 (Pkinase)       →  38 sequences
                         PF00072 (Response_reg)  →  52 sequences
   Length              min 51  ·  median 187  ·  max 476 residues
   Chance baseline     0.578 (majority-class fraction)
   Embedders           composition  kmer
   Reducers            pca  umap
   Output              results

  Scoring    kmer_k3 umap ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:14
────────────────────────────── Benchmark Results ───────────────────────────────

╭───────────────────┬──────────────┬───────┬───────┬───────┬───────┬────────╮
│ Method            │   kNN-full   │  P@k  │ Silh  │  ARI  │ Trust │ kNN-2D │
├───────────────────┼──────────────┼───────┼───────┼───────┼───────┼────────┤
│ composition+umap  │ 0.944 ±0.05  │ 0.921 │ 0.310 │ 0.712 │ 0.898 │ 0.944  │
│ kmer_k3+umap    * │ 1.000 ±0.00  │ 0.987 │ 0.208 │ 0.845 │ 0.799 │ 1.000  │
╰───────────────────┴──────────────┴───────┴───────┴───────┴───────┴────────╯
  Chance baseline (majority class): 0.578 — k-NN and P@k at or below this are
  no better than guessing.
  * best by full-dimensional k-NN.  Full metric set: metrics/benchmark.csv
```

## Benchmark your own sequences

Pfam is the built-in reference set, not a requirement. Point Prot2Vec at any
protein FASTA and tell it how to read a group label from each header:

```yaml
# configs/experiments/fasta.yaml
fasta:
  path: data/my_proteins.fasta
  label_from: first_token      # ">PF00069 sp|P12345|..."  -> PF00069

embedders:
  - name: composition
  - name: kmer
    k: 3

reducer:
  name: pca
```

`label_from` accepts:

| Value | Reads the label from | Example header → label |
|---|---|---|
| `first_token` | text before the first space | `>PF00069 kinase_1` → `PF00069` |
| `last_token` | text after the last space | `>seq1 CYTOPLASM` → `CYTOPLASM` |
| `description` | everything after the identifier | `>seq1 heat shock` → `heat shock` |
| `pipe:N` | field `N` of a `\|`-delimited header | `>sp\|P12345\|KIN` with `pipe:0` → `sp` |
| `none` | nothing — one single group | for unlabelled exploration |

The groups do not have to be protein families. Anything you want to test for
separability works: organism, subcellular localisation, EC class, experimental
condition, cluster assignment from another tool.

## The methods

| Method | Type | Dimensions | Needs |
|---|---|---|---|
| `composition` | residue frequency vector | 20 | — |
| `kmer` | TF-IDF over residue k-mers | up to 20ᵏ, sparse | — |
| `esm2` | mean-pooled protein language model | 320 / 480 / 640 / 1280 | `[esm]` |
| `llm` | general-purpose text embedding API | model-dependent | `[llm]` + API key |

**`composition`** discards residue order entirely. Any separation it achieves
comes from compositional bias alone — hydrophobic membrane domains separate
from soluble enzymes without any model learning anything about structure. It is
the floor a protein language model has to beat to have earned its GPU time.

**`kmer`** keeps local order, so short conserved motifs contribute. TF-IDF
down-weights k-mers that appear everywhere. Often a surprisingly strong
baseline on seed alignments, where family members share recognisable motifs.

**`esm2`** is the reference point: a transformer pretrained on UniRef that has
seen no Pfam labels. Residue representations are mean-pooled over the sequence.
Checkpoints range from 8M to 650M parameters (`esm2_t6_8M` through
`esm2_t33_650M`).

**`llm`** sends the raw residue letters to a text embedding model as if they
were prose. This is a **control, not a protein method** — read its score as a
floor, and as a measure of how much apparent "protein understanding" is really
just character statistics. It is included because the comparison is
informative, not because it is recommended.

## The metrics, and how to read them

Every metric is reported for every `(embedder, reducer)` pair. All except
`trustworthiness` and `kNN-2D` are computed on the **full-dimensional**
embedding, because that is the representation under test — scoring only the 2-D
projection grades the reducer instead.

| Metric | Question | Range |
|---|---|---|
| `kNN-full` | Are families recoverable from the embedding? | 0–1, vs. chance baseline |
| `P@k` | Do a protein's nearest neighbours share its family? | 0–1, vs. chance baseline |
| `Silh` | Are the families geometrically compact and separated? | −1 to 1 |
| `ARI` | Could families be recovered *without* labels? | ~0 for random, 1 for perfect |
| `Trust` | Did the projection preserve the real neighbourhoods? | 0–1 |
| `kNN-2D` | Are families separable in the picture you plotted? | 0–1 |

Three patterns worth knowing:

- **`kNN-full` high, `kNN-2D` low, `Trust` low.** The signal is in the
  embedding and the reducer is destroying it. Believe the embedding, not the
  figure — and try a different projection.
- **`kNN-full` barely above the chance baseline.** The representation does not
  encode family membership, however convincing the scatter plot looks. This is
  why the baseline is printed on every run.
- **`kNN-full` high but `ARI` low.** Families are separable but not by the
  dominant axes of variance. A supervised model will work; unsupervised
  clustering of unannotated sequences will not.

`P@k` is the one that translates most directly into practice: it is the
embedding-space analogue of a BLAST or HMMER hit list, and it answers "could I
annotate an unknown sequence from its neighbours?".

## What a run produces

```
results/
├── metrics/benchmark.csv              # every metric, every pair — the full record
├── embeddings/
│   ├── sequences.tsv                  # id, family, length, sequence — the join key
│   ├── composition.npy                # (n_sequences, d), row-aligned with the above
│   └── kmer_k3.npy
├── projections/
│   └── composition__pca.tsv           # sequence_id, family, dim1, dim2
├── figures/
│   ├── composition_pca.png
│   └── metrics_summary.png
└── run_manifest.json                  # versions, git commit, resolved settings
```

`sequences.tsv` row *i* corresponds to row *i* of every `.npy` and every
projection, so the numeric exports are usable outside Prot2Vec. The projection
TSVs exist so a figure can be rebuilt in ggplot2 or plotly without re-running
the embedding:

```r
library(ggplot2)
df <- read.delim("results/projections/esm_esm2_t12_35M__umap.tsv")
ggplot(df, aes(dim1, dim2, colour = family)) + geom_point()
```

`run_manifest.json` records the Prot2Vec version, Python version, platform, git
commit, dependency versions, dataset composition and every resolved embedder,
reducer and metric parameter — so a number in `benchmark.csv` can be traced
back to what produced it months later.

## Configuration

Experiments are **YAML files, not command-line flags** — an experiment should
be something you can commit, diff and cite. `configs/default.yaml` documents
every available option inline; the presets in `configs/experiments/` are
starting points:

| Preset | What it does |
|---|---|
| `quick.yaml` | 2 families, 2 cheap embedders — fast iteration |
| `full.yaml` | 5 families, includes ESM-2 |
| `reducers.yaml` | 1 embedder, 3 projections — tests whether the picture is real |
| `llm.yaml` | adds the general-purpose LLM control |
| `fasta.yaml` | your own sequences instead of Pfam |

The CLI takes only operational flags:

```
--config PATH      experiment YAML (default: configs/default.yaml)
--log-level LEVEL  debug | info | warning | error
--no-cache         ignore cached embeddings and recompute
--no-figures       skip writing figures
--version
```

## Library use

```python
from prot2vec import (
    ProteinDataset, CompositionEmbedder, KmerEmbedder,
    PCAReducer, UMAPReducer, RunConfig, run,
)

dataset = ProteinDataset.from_fasta("my_proteins.fasta", label_from="first_token")
print(dataset.summary())
# {'n_sequences': 240, 'n_families': 4, 'majority_class_fraction': 0.31, ...}

results = run(RunConfig(
    dataset=dataset,
    embedders=[CompositionEmbedder(), KmerEmbedder(k=3)],
    reducers=[PCAReducer(), UMAPReducer(n_neighbors=15)],
    results_dir="results",
))
print(results[["method", "knn_accuracy_highdim_mean", "precision_at_k"]])
```

Or use a single piece on its own:

```python
from prot2vec import ESMEmbedder, retrieval_precision_at_k

vectors = ESMEmbedder("esm2_t12_35M").fit_transform(dataset.sequences)
print(retrieval_precision_at_k(vectors, dataset.labels, k=5))
```

Every embedder guarantees **one output row per input sequence, in order**, so
vectors stay aligned with labels and identifiers.

Pfam data can be loaded directly too:

```python
from prot2vec import download_pfam_seed, parse_pfam_families, ProteinDataset

seed = download_pfam_seed(version="35.0", cache_dir="data/raw")
records = parse_pfam_families(["PF00069", "PF00072"], seed)
dataset = ProteinDataset.from_pfam_records(records, min_length=50, max_per_family=100)
```

`max_per_family` matters more than it looks: Pfam seed alignments differ in size
by more than an order of magnitude, and an unbalanced benchmark rewards a method
for simply predicting the largest family.

## Extending Prot2Vec

**A new embedder** — subclass `SequenceEmbedder`, implement `name` and
`fit_transform`, and register it in `_build_embedder` in `src/prot2vec/cli.py`:

```python
from prot2vec.embedders.base import SequenceEmbedder

class ProtBertEmbedder(SequenceEmbedder):
    def __init__(self, model_name: str = "Rostlab/prot_bert") -> None:
        self.model_name = model_name

    @property
    def name(self) -> str:
        return f"protbert_{self.model_name.split('/')[-1]}"

    def fit_transform(self, sequences: list[str]) -> np.ndarray:
        vectors = ...                              # one row per sequence, in order
        self._validate_rows(vectors, sequences)    # enforces the contract
        return vectors
```

Caching and manifest recording come for free: `params` is derived from your
constructor signature, so any argument you accept becomes part of the cache key
automatically. Add a unit test in `tests/test_embedders.py`.

**A new reducer** follows the same pattern via `DimReducer` and
`_build_reducer`. **Heavy or optional dependencies** go behind a pyproject
extra, imported lazily inside the method that needs them with an `ImportError`
that names the extra.

## Hardware and cost

ESM-2 picks its device automatically — CUDA, then Apple Silicon MPS, then CPU —
with fp16 autocast on GPU. Runtime is dominated by `max_len` rather than by
sequence count, since attention is quadratic in length.

| Setup | Suggested checkpoint |
|---|---|
| NVIDIA GPU | `esm2_t33_650M` |
| Apple Silicon | `esm2_t12_35M` or `esm2_t30_150M` |
| CPU only | `esm2_t6_8M`, and lower `max_len` |

The `llm` embedder makes billed API calls. Requests are batched, and
`cache_embeddings: true` (the default) means a re-run reads vectors from disk
instead of paying again. Retry settings are deliberately excluded from the
cache key so that tuning them does not invalidate the cache.

API keys are read from a `.env` file at the repository root, never from a config
file:

```
GOOGLE_API_KEY=your_key_here
```

## Reproducibility

- Pfam releases are pinned in the config, so a rerun sees the same sequences.
- Every stochastic step — fold shuffling, k-means init, UMAP, t-SNE,
  subsampling — takes an explicit seed, defaulting to `0`.
- Sparse embeddings are cached in sparse form, so a cached rerun feeds the
  reducer exactly what the first run did and produces identical numbers.
- `run_manifest.json` records the versions and settings behind every result.
- `ProteinDataset.to_fasta()` writes the exact benchmarked input, so a run can
  be reproduced independently of the Pfam release.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
pip install -e ".[esm,llm,dev]"
pre-commit install
pytest                              # 208 tests
ruff check src tests
mypy
```

## Citation

If Prot2Vec supports published work, please cite it via
[CITATION.cff](CITATION.cff), and cite the underlying resources directly:

- **Pfam** — Mistry et al. (2021), *Pfam: The protein families database in
  2021*, Nucleic Acids Research 49:D412–D419.
- **ESM-2** — Lin et al. (2023), *Evolutionary-scale prediction of atomic-level
  protein structure with a language model*, Science 379:1123–1130.
- **UMAP** — McInnes et al. (2018), arXiv:1802.03426.

## License

[MIT](LICENSE)
