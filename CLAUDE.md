# Prot2Vec

Benchmark harness comparing biological sequence embeddings on group separability: 9 embedders x 13 reducers, scored with 31 metrics in 5 groups. Protein, DNA and RNA alphabets.

## Layout

- `src/prot2vec/` — the package, src-layout, `import prot2vec` (`data/`, `embedders/`, `reduction/`, `evaluation/`, `visualization/`, `pipeline.py`, `cli.py`, `_matrix.py`)
- `src/prot2vec/evaluation/` — a package: `projection.py`, `classification.py`, `retrieval.py`, `clustering.py`, `confound.py`, `suite.py`; `metrics.py` is a back-compat re-export shim
- `configs/experiments/` — YAML presets (`quick`, `full`, `descriptors`, `plm`, `reducers`, `llm`, `fasta`, `dna`)
- `src/prot2vec/cli.py` — entry points: `main` (`prot2vec`, `prot2vec-benchmark`) and `download` (`prot2vec-download`)
- `tests/` — 411 pytest tests, import via `from prot2vec...`; shared fixtures in `tests/conftest.py`
- root `conftest.py` — puts `src/` on `sys.path` so the suite runs from a fresh clone without installing

## Where to find things

- **README.md** — install, usage, the metrics and how to read them, output layout
- **CONTRIBUTING.md** — the invariants below, expanded, plus the PR checklist
- **configs/default.yaml** — canonical config with every option commented
- **src/prot2vec/pipeline.py** — orchestration of embed → reduce → evaluate; `RunConfig` shape, caching, exports, manifest
- **src/prot2vec/embedders/base.py** — `SequenceEmbedder` ABC: `name`, `fit_transform`, `params`, `cache_key`
- **src/prot2vec/_matrix.py** — `EmbeddingMatrix` alias and the single `as_dense` / `as_csr` / `is_sparse` helpers
- **src/prot2vec/data/alphabets.py** — `Alphabet`, `PROTEIN`/`DNA`/`RNA`, `clean_sequence`, `require_protein`
- **src/prot2vec/evaluation/suite.py** — `METRIC_GROUPS`, `available_metrics()`, and the `evaluate()` registry
- **data/raw/** — cached Pfam-A.seed.gz (~500 MB, downloaded once)

## Project-specific knowledge

- **Pfam IDs** (e.g. `PF00069` = Pkinase, `PF00072` = Response_reg) are EBI accessions; the seed alignment is the curated subset, not the full family. `src/prot2vec/data/families.py` holds the curated identity catalogue — check new accessions against InterPro before adding them.
- **Sequences are stripped to the configured alphabet** before embedding, by `clean_sequence(seq, alphabet)`. For protein that removes gaps, `X`, `B`, `Z`, `U`, `O`.
- **Always set `data.alphabet` for nucleotides.** `A`, `C`, `G`, `T` and `N` are all valid amino acid codes, so DNA cleaned against the protein alphabet is silently mangled rather than rejected. `physicochemical` and `ctd` call `require_protein()` and refuse nucleotide runs.
- **Data can come from Pfam or from FASTA.** A config supplies either a `pfam:` or a `fasta:` block; `from_fasta` reads the group label out of the header via `label_from`.
- **Metrics name their space.** `trustworthiness`, `continuity`, `neighborhood_preservation`, `lcmc`, `distance_correlation` and `knn_accuracy_*` (2-D) grade the *reducer*; everything else grades the *embedding*. Reported per `(embedder, reducer)` pair, always next to the majority-class baseline.
- **The `confound` group is the distinctive one.** `length_only_knn_accuracy` is the accuracy reachable from sequence length alone — read every other score against it. It needs the sequences, which `evaluate()` cannot get from the vectors, so the pipeline passes them.
- **Metric groups** are `projection`, `classification`, `retrieval`, `clustering`, `confound`; select with `metrics.groups` in YAML (mapped to `metric_groups=` on `evaluate()`).
- **A run is a matrix.** `reducers:` takes a list; each embedding is computed once and reused across every reducer.
- **One invalid pair must not abort the matrix.** NMF rejects signed input, manifold methods fail on disconnected graphs, optional backends may be absent. `run()` isolates per-pair failures, names them in the output and records them under `skipped_pairs` in the manifest; it raises only if every pair fails.
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
- **`umap` is broken in some environments** via `umap → parametric_umap → tensorflow → googleapiclient → pyparsing`, which raises `AttributeError`, not `ImportError`. `UMAPReducer` catches broadly on purpose; use `reducer: {name: pca}` or `svd` to work around it.
- **Don't write an unanchored `...` in coverage `exclude_lines`.** It matches `tuple[str, ...]` in a type annotation, and coverage then excludes that function's entire body — this silently hid `evaluate()` from the report. The pattern is anchored to `^\s*\.\.\.$`.
- **Don't add a second `_as_dense`.** The dense/sparse conversions live once in `_matrix.py`.
- **Never assert on a tie-break-dependent value.** Degenerate fixtures (`np.eye(n)`, all-identical points) make every pairwise distance equal, so neighbour rankings come down to the sort's tie-breaking — `compute_trustworthiness(np.eye(20), ...)` returns 0.27 on scikit-learn 1.8 and 0.95 on 1.9. Assert the guaranteed invariant (finite, in range) and test meaning on well-separated data.
- **The local anaconda env is far behind CI** (numpy 1.24/sklearn 1.8 vs numpy 2.4/sklearn 1.9/pandas 3.0), and its newer stubs catch real errors. Before pushing CI-visible changes, mirror it: `python3 -m venv /tmp/civenv && /tmp/civenv/bin/pip install -e ".[dev]"` then run ruff/mypy/pytest from that venv. `pip install -e .` works there even though it fails under anaconda.
- **Don't pin `python_version` in the mypy config** below the interpreter mypy runs on: numpy 2.4's stubs use PEP 695 `type` statements and mypy rejects them outright when told to assume 3.11.

## Preferred patterns

- **New embedder** → subclass `SequenceEmbedder` in `src/prot2vec/embedders/`, expose via `_build_embedder` in `src/prot2vec/cli.py`, document in `configs/default.yaml`, add a test in `tests/test_embedders.py`.
- **New reducer** → same pattern via `DimReducer` and `_build_reducer`, plus a `params` property for the manifest.
- **New metric** → put it in the `evaluation/` module for its group, register it in `suite.py`'s `evaluate()` entry list and in `available_metrics()`, and raise `ValueError` on unmet preconditions so it is skipped rather than fatal.
- **Alphabet-aware embedder** → take an `alphabet` argument, read tokens from `get_alphabet(...)`, and have `cli._build_embedder` forward the dataset alphabet.
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
