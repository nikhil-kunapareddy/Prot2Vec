# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0]

Breadth release: nine representations, thirteen projections, thirty-one
metrics, and support for nucleotide as well as protein sequences.

### Added

- **Sequence alphabets.** `data.alphabet` accepts `protein`, `dna` or `rna`,
  and the alphabet-aware embedders switch vocabulary accordingly — dipeptide
  composition becomes 16 dinucleotide frequencies, one-hot becomes four
  channels per position. `SequenceDataset` is available as an alias for
  `ProteinDataset`, which is alphabet-aware rather than protein-specific.
  Setting this matters: `A`, `C`, `G`, `T` and `N` are all valid amino acid
  codes, so nucleotide sequences cleaned against the protein alphabet were
  silently mangled rather than rejected.
- **Five new embedders.** `dipeptide` (fixed-order adjacent-pair frequencies),
  `physicochemical` (23 features with physical units), `ctd`
  (composition/transition/distribution over seven property groups), `onehot`
  (positional, sparse), and `hf` — one class covering ProtBERT, ProtT5, Ankh,
  ProstT5, the ESM-2 Hugging Face ports, DNABERT-2 and the Nucleotide
  Transformer, with mean/CLS/max pooling and selectable hidden layer.
- **Ten new reducers.** `svd`, `nmf`, `random_projection`, `isomap`, `mds`,
  `spectral`, `lle`, `kernel_pca`, plus optional `phate` and `pacmap`.
  `random_projection` is a deliberate control: it fits nothing, so structure
  that survives it was robust to begin with.
- **Twenty-one new metrics**, organised into five selectable groups via
  `metrics.groups`:
  - `projection` adds continuity, neighbourhood preservation, LCMC and
    Shepard-diagram distance correlation.
  - `classification` adds macro F1, balanced accuracy, MCC, Cohen's kappa and
    one-vs-rest AUROC, because accuracy alone rewards ignoring small classes.
  - `retrieval` adds mean average precision, R-precision and same-class AUROC
    (the remote-homology framing, insensitive to class imbalance).
  - `clustering` adds homogeneity, completeness, V-measure, Fowlkes-Mallows,
    Davies-Bouldin and Calinski-Harabasz.
  - `confound` is new: `length_only_knn_accuracy` reports the accuracy
    reachable from sequence length alone, alongside length/distance and
    composition/distance correlations. The CLI warns when a result is
    explained by either.
- New extras: `[hf]`, `[phate]`, `[pacmap]` and `[all]`.
- New presets: `descriptors.yaml`, `plm.yaml`, `dna.yaml`; `reducers.yaml` now
  compares eight projections.
- 411 tests (up from 208) at 84% coverage.

### Changed

- The `evaluation` module became a package — `projection`, `classification`,
  `retrieval`, `clustering`, `confound` and `suite`. Every previous import path
  still works; `prot2vec.evaluation.metrics` re-exports the full surface.
- `evaluate()` takes `sequences`, `metric_groups` and `alphabet`. The pipeline
  supplies the first and last automatically, which is what makes the confound
  group possible — "is this just encoding length?" cannot be answered from the
  vectors alone.
- The results table shows five columns instead of eight and one line per row;
  a full run can produce sixty rows, which no terminal renders legibly. The CSV
  is unchanged and carries every metric.
- `clustering_report()` supersedes `clustering_agreement()`, which remains as a
  two-tuple wrapper.

### Fixed

- **One invalid pair no longer aborts the whole matrix.** NMF rejects signed
  input, a manifold method can fail on a disconnected neighbour graph, an
  optional backend may be missing — previously any of these discarded every
  pair that had already succeeded. Failures are now isolated per pair, named in
  the output, and recorded under `skipped_pairs` in the run manifest. A run
  still raises if *every* pair fails.
- **The protein-only descriptors did not receive the dataset alphabet**, so a
  nucleotide run with `physicochemical` or `ctd` would have scored DNA with
  amino acid scales instead of refusing.
- **Coverage silently hid whole functions.** The `exclude_lines` pattern for
  abstract-method ellipsis bodies was unanchored, so it also matched
  `tuple[str, ...]` in a type annotation and excluded that function's entire
  body — `evaluate()` among them. Now anchored to a line containing only `...`.
- `MDS` pins `init="random"` where the parameter exists, so scikit-learn's
  upcoming default change cannot silently move published numbers.
- Removed a duplicated `_as_dense` helper; the dense/sparse conversions now
  live once in `prot2vec._matrix` alongside the `EmbeddingMatrix` type alias,
  which replaced the `np.ndarray` annotations that sparse embedders were
  contradicting with `# type: ignore`.

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

[Unreleased]: https://github.com/nikhil-kunapareddy/Prot2Vec/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/nikhil-kunapareddy/Prot2Vec/releases/tag/v0.3.0
[0.2.0]: https://github.com/nikhil-kunapareddy/Prot2Vec/releases/tag/v0.2.0
[0.1.0]: https://github.com/nikhil-kunapareddy/Prot2Vec/releases/tag/v0.1.0
