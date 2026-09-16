# Contributing to Prot2Vec

Thanks for considering a contribution. Bug reports, new embedders, new metrics
and documentation fixes are all welcome.

## Getting set up

```bash
git clone https://github.com/nikhil-kunapareddy/Prot2Vec.git
cd Prot2Vec

python -m venv .venv && source .venv/bin/activate
pip install -e ".[esm,llm,dev]"
pre-commit install
```

Then check the suite is green before you change anything:

```bash
pytest
ruff check src tests
ruff format --check src tests
mypy
```

The test suite needs no network access and no model weights — everything that
would download something is either mocked or behind a marker.

## Before you open a pull request

```bash
pytest                    # all tests pass
ruff check src tests      # no lint findings
mypy                      # no type errors
```

`pre-commit install` runs the formatting and lint checks automatically on
commit, which is the least annoying way to keep CI green.

## What good looks like here

**Every embedder returns one row per input sequence, in input order.** This is
the contract the whole benchmark rests on: output rows are matched to family
labels positionally, so dropping a sequence silently shifts every later row
against the wrong label. If an input cannot be embedded, raise with a message
that says how to filter it — never return fewer rows. `_validate_rows()` on the
base class enforces this; call it before returning.

**Metrics say which space they measure.** `trustworthiness` compares the
embedding against its projection. Everything else is computed on the
full-dimensional embedding, because that is the representation under test.
If you add a metric, be explicit about this in its docstring, and make it
degrade gracefully — raise `ValueError` when its preconditions are not met, and
`evaluate()` will skip it with a warning rather than failing the whole run.

**Heavy dependencies stay optional.** PyTorch and API SDKs go behind a
pyproject extra (`[esm]`, `[llm]`), imported lazily inside the method that
needs them, with an `ImportError` that names the extra to install. A base
install must stay usable without a deep-learning stack.

**Experiments are YAML, not flags.** New comparisons belong in
`configs/experiments/`, not in new argparse arguments. The CLI takes only
operational flags (`--log-level`, `--no-cache`). This keeps an experiment
something you can commit, diff and cite.

**Randomness takes a seed.** Anything stochastic accepts a `random_state`
defaulting to `0`, and is recorded in the run manifest.

**Get the biology right.** Pfam accessions have specific identities; if you add
one to `src/prot2vec/data/families.py` or a config comment, check it against
[InterPro](https://www.ebi.ac.uk/interpro/). A wrong domain annotation is worse
than no annotation, because it silently misleads anyone interpreting the
results.

## Adding an embedder

1. Subclass `SequenceEmbedder` in `src/prot2vec/embedders/`.
2. Implement the `name` property and `fit_transform`.
3. Register it in `_build_embedder` in `src/prot2vec/cli.py`.
4. Document its config block in `configs/default.yaml`.
5. Add tests in `tests/test_embedders.py`.

Caching and manifest recording are automatic: `params` is derived from your
constructor signature, so anything your `__init__` accepts and stores becomes
part of the cache key. Override `params` if a setting should *not* affect
caching — `LLMEmbedder` does this for its retry options, so that changing them
does not invalidate cached vectors and re-bill every API request.

Adding a reducer works the same way via `DimReducer` and `_build_reducer`;
reducers additionally expose an explicit `params` property for the manifest.

## Style

Enforced by `ruff` and configured in `pyproject.toml`: 100-column lines,
numpy-style docstrings, type annotations on public functions.

On comments: explain *why*, not *what*. The line
`# UMAP requires n_neighbors < n_samples; seed alignments are small` is worth
keeping because it records a constraint that is not obvious from the code.
A comment restating the next line is not.

## Tests

- No network access, no downloaded weights. Use `monkeypatch` for HTTP and
  build fixtures in `tmp_path` — `tests/test_pfam.py` constructs a miniature
  Stockholm archive rather than downloading Pfam.
- Test the guard rails, not just the happy path. Most bugs found while building
  this were in degenerate cases: a family with one member, a sequence with no
  standard residues, a cached array of the wrong shape.
- Shared fixtures live in `tests/conftest.py`. `ConstantEmbedder` is a
  dependency-free embedder for pipeline tests.

Markers are available for tests that genuinely need more: `slow`, `network`,
`esm`, `llm`. Deselect them with `pytest -m "not slow"`.

## Reporting bugs

Open an issue with the config you ran, the command, the full error, and your
platform plus Python version. If the run got far enough to write
`results/run_manifest.json`, attaching it captures all the version information
in one go.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
