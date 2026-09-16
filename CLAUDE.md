# Prot2Vec

Benchmark harness comparing protein sequence embeddings (amino acid composition, k-mer TF-IDF, ESM-2, general-purpose LLM APIs) on protein family separability, across PCA/UMAP/t-SNE projections.

## Layout

- `src/prot2vec/` — the package, src-layout, `import prot2vec` (`data/`, `embedders/`, `reduction/`, `evaluation/`, `visualization/`, `pipeline.py`, `cli.py`, `_matrix.py`)
- `configs/experiments/` — YAML presets (`quick.yaml`, `full.yaml`, `reducers.yaml`, `llm.yaml`, `fasta.yaml`)
- `src/prot2vec/cli.py` — entry points: `main` (`prot2vec`, `prot2vec-benchmark`) and `download` (`prot2vec-download`)
- `tests/` — 208 pytest tests, import via `from prot2vec...`; shared fixtures in `tests/conftest.py`
- root `conftest.py` — puts `src/` on `sys.path` so the suite runs from a fresh clone without installing

## Where to find things

- **README.md** — install, usage, the metrics and how to read them, output layout
- **CONTRIBUTING.md** — the invariants below, expanded, plus the PR checklist
- **configs/default.yaml** — canonical config with every option commented
- **src/prot2vec/pipeline.py** — orchestration of embed → reduce → evaluate; `RunConfig` shape, caching, exports, manifest
- **src/prot2vec/embedders/base.py** — `SequenceEmbedder` ABC: `name`, `fit_transform`, `params`, `cache_key`
- **src/prot2vec/_matrix.py** — `EmbeddingMatrix` alias and the single `as_dense` / `as_csr` / `is_sparse` helpers
- **data/raw/** — cached Pfam-A.seed.gz (~500 MB, downloaded once)

## Project-specific knowledge

- **Pfam IDs** (e.g. `PF00069` = Pkinase, `PF00072` = Response_reg) are EBI accessions; the seed alignment is the curated subset, not the full family. `src/prot2vec/data/families.py` holds the curated identity catalogue — check new accessions against InterPro before adding them.
- **Sequences are stripped to the 20 standard AAs** before embedding — gaps, `X`, `B`, `Z`, `U`, `O` are removed by `clean_sequence`.
- **Data can come from Pfam or from FASTA.** A config supplies either a `pfam:` or a `fasta:` block; `from_fasta` reads the group label out of the header via `label_from`.
- **Metrics name their space.** `trustworthiness` (embedding vs. projection) and `knn_accuracy_*` (2-D) grade the reducer; `knn_accuracy_highdim_*`, `precision_at_k`, `silhouette`, `adjusted_rand`, `normalized_mutual_info` grade the embedding. Reported per `(embedder, reducer)` pair, always next to the majority-class baseline.
- **A run is a matrix.** `reducers:` takes a list; each embedding is computed once and reused across every reducer.
- **ESM-2** auto-detects device (CUDA / MPS / CPU); fp16 autocast on GPU only. Use `esm2_t6_8M` on CPU.
- **The LLM embedder is a control, not a method** — it sends raw AA letters as plain text to a text embedding API to show how much apparent protein signal is just character statistics.
- **Caching** is keyed on `embedder.cache_key` (derived from the constructor signature) plus a hash of the sequences. Sparse embeddings are cached sparsely so reruns are bit-identical. Delete `results/embeddings/` to force a re-embed.
- **Every run writes provenance** to `results/run_manifest.json`: versions, git commit, resolved settings, dataset composition.

## Recurring mistakes to avoid

- **Don't commit `data/raw/` or `results/`** — both are gitignored; `pre-commit` also blocks files over 1 MB.
- **`GOOGLE_API_KEY` must be in `.env`** at repo root (loaded via `python-dotenv` in `src/prot2vec/cli.py`). Never in a config file, never hardcoded.
- **Never let an embedder return fewer rows than it was given.** Output rows are matched to family labels positionally, so a dropped sequence silently shifts every later row onto the wrong label. Raise instead, and call `_validate_rows`.
- **Don't fold runtime state into `params`.** It feeds `cache_key`; a value that changes during a run (a counter, a resolved device) would invalidate the cache mid-run. `params` comes from the constructor signature for exactly this reason.
- **Don't call `plt.show()` in library code** — it blocks headless and CI runs. Return the figure and close it after saving.
- **`umap` is broken in some environments** via `umap → parametric_umap → tensorflow → googleapiclient → pyparsing`, which raises `AttributeError`, not `ImportError`. `UMAPReducer` catches broadly on purpose; use `reducer: {name: pca}` to work around it.

## Preferred patterns

- **New embedder** → subclass `SequenceEmbedder` in `src/prot2vec/embedders/`, expose via `_build_embedder` in `src/prot2vec/cli.py`, document in `configs/default.yaml`, add a test in `tests/test_embedders.py`.
- **New reducer** → same pattern via `DimReducer` and `_build_reducer`, plus a `params` property for the manifest.
- **New metric** → raise `ValueError` when preconditions are unmet so `evaluate()` skips it with a warning instead of failing the run.
- **Optional heavy deps** go behind pyproject extras (`[esm]`, `[llm]`) with a lazy `ImportError` naming the extra.
- **Dense-or-sparse** matrices use the `EmbeddingMatrix` alias and the helpers in `_matrix.py` — don't write another `_as_dense`.
- **CLI output** uses `rich` (panels, progress bars, tables); failures render through `_fail` as a panel, not a traceback.
- **Experiments are YAML, not CLI flags** — add a file under `configs/experiments/`. The CLI takes only operational flags (`--log-level`, `--no-cache`, `--no-figures`).
- **Anything stochastic takes a `random_state`** defaulting to `0`.

## Commands

```bash
pip install -e ".[esm,llm,dev]" && pre-commit install
prot2vec-download --version 35.0 --cache-dir data/raw
prot2vec-benchmark --config configs/experiments/quick.yaml   # or: python -m prot2vec --config ...
pytest && ruff check src tests && ruff format --check src tests && mypy
```

Python ≥ 3.11.
