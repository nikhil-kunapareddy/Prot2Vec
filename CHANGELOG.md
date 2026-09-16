# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

A correctness and usability release. The import path changed, so this is a
breaking one.

### Breaking

- The package is now `prot2vec`, laid out under `src/prot2vec/`. Replace
  `from src... import` with `from prot2vec... import`, and `python -m src`
  with `python -m prot2vec`. Reinstall with `pip install -e .` to refresh the
  console scripts.
- `RunConfig.reducer` is now `RunConfig.reducers` and takes a list. Passing a
  single `DimReducer` still works and is normalised to a one-element list.
- `evaluate()` returns a larger metric dict. Existing keys are unchanged; code
  that asserted on the exact key set will need updating.

### Added

- **FASTA input.** `ProteinDataset.from_fasta()` reads plain or gzipped FASTA
  with five header-labelling strategies (`first_token`, `last_token`,
  `description`, `pipe:N`, `none`), and configs accept a `fasta:` block instead
  of `pfam:`. Pfam is no longer the only possible input.
- **New metrics.** `precision@k` retrieval (the homology-search analogue),
  silhouette, and k-means agreement (ARI / NMI). k-NN accuracy is now reported
  in the full-dimensional embedding as well as in the 2-D projection.
- **Multi-reducer runs.** `reducers:` takes a list, and each embedding is
  computed once and reused across every reducer.
- **Exports.** `sequences.tsv` (the id-aligned join key), `<embedder>.npy` per
  embedding, and `projections/*.tsv` with 2-D coordinates for re-plotting
  elsewhere.
- **Run manifest.** `run_manifest.json` records the Prot2Vec and Python
  versions, platform, git commit, dependency versions, dataset composition and
  every resolved parameter.
- **Chance baseline in output.** The CLI prints the majority-class fraction
  alongside every score.
- **Balanced sampling.** `max_per_family` caps each family, seeded for
  reproducibility.
- `prot2vec.data.families` — a curated Pfam accession catalogue with validation,
  so a malformed accession fails before a 500 MB download rather than after it.
- `--version`, `--log-level`, `--no-cache` and `--no-figures` CLI flags; a
  download progress bar; and rendered error panels instead of tracebacks.
- Type hints throughout with a `py.typed` marker; strict `mypy` configuration.
- 208 tests (up from 17) at 81% coverage, plus CI, pre-commit and issue
  templates.

### Fixed

- **`ESMEmbedder` silently dropped sequences** shorter than 10 residues after
  cleaning, while the label list kept them — shifting every later row against
  the wrong family. Embedders now guarantee one row per input sequence, raising
  instead of dropping, enforced in both the base class and the pipeline.
- **Cached sparse embeddings were densified on write**, so the first run
  reduced a sparse matrix and the second reduced a dense one, producing
  different UMAP and t-SNE output. Sparse embeddings are now cached sparsely and
  a cached rerun reproduces the original numbers exactly.
- **Embedding cache keys ignored embedder settings.** Two `LLMEmbedder`
  instances differing only in `output_dim` shared a cache entry. Keys are now
  derived from the constructor signature — and exclude runtime state, which
  previously could change a key mid-run.
- **`PF00072` was annotated as "GPCR"** in every config and in the README. It is
  `Response_reg`, the two-component response regulator receiver domain;
  `PF00001` is the rhodopsin-family GPCR domain.
- **The Pfam download was not atomic.** An interrupted transfer left a truncated
  file that every later run trusted as a valid cache. Downloads now write to a
  `.part` file, verify the gzip stream, and only then rename; 5xx and rate-limit
  responses are retried, while a wrong release number fails immediately with a
  pointer to the release listing.
- **`plt.show()` in library code** blocked headless and CI runs, and figures
  were never closed, leaking memory across a benchmark matrix. Figures are now
  returned and closed after saving.
- **`LLMEmbedder.batch_size` was documented but unused** — every sequence was
  sent as its own request. Requests are now genuinely batched, and rate-limit
  detection uses the response rather than matching on an exception class name.
- Metrics no longer crash on small or unbalanced families: fold counts and
  neighbourhood sizes are clamped to what the data supports, with a warning.
- `ProteinDataset.to_fasta()` output could not be read back by
  `from_fasta()` with its default settings.
- Duplicate sequence identifiers made the exported TSVs ambiguous as a join
  key; they are now disambiguated.
- `UMAPReducer` now reports a broken `umap-learn` install as an actionable
  `ImportError`. Because `umap` eagerly imports `parametric_umap` and therefore
  TensorFlow, a clash in that chain surfaces as an `AttributeError`, which
  previously escaped as an unhandled traceback.
- Amino acid composition is normalised by standard residues rather than raw
  sequence length, so stripped characters no longer dilute every frequency.
- `prot2vec.pipeline` no longer imports `matplotlib` at module load.

### Removed

- An unreachable branch in `knn_cv_accuracy`, and a permanently failing
  trustworthiness test whose premise was wrong: `np.eye(20)` makes all points
  mutually equidistant, so no 2-D projection can preserve their neighbourhoods.

## [0.1.0]

Initial release: amino acid composition, k-mer TF-IDF, ESM-2 and Google Gemini
embedders; PCA, UMAP and t-SNE reducers; trustworthiness and 5-NN accuracy
metrics; YAML-driven experiments and a rich CLI.

[Unreleased]: https://github.com/nikhil-kunapareddy/Prot2Vec/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/nikhil-kunapareddy/Prot2Vec/releases/tag/v0.2.0
[0.1.0]: https://github.com/nikhil-kunapareddy/Prot2Vec/releases/tag/v0.1.0
