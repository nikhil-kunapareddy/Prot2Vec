## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- What problem this solves, or what question it lets someone answer. -->

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check src tests` and `ruff format --check src tests` are clean
- [ ] `mypy` is clean
- [ ] New behaviour has a test, including its failure cases
- [ ] Docstrings explain *why*, not just *what*
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

### If this adds an embedder or reducer

- [ ] Registered in `_build_embedder` / `_build_reducer` in `src/prot2vec/cli.py`
- [ ] Config block documented in `configs/default.yaml`
- [ ] Returns one row per input sequence, in order (`_validate_rows` called)
- [ ] Heavy dependencies are behind a pyproject extra and imported lazily

### If this adds a metric

- [ ] States which space it measures (full-dimensional or projected)
- [ ] Raises `ValueError` when its preconditions are unmet, so `evaluate()`
      can skip it rather than failing the run
- [ ] Documented in the README metrics table

### If this touches Pfam identities

- [ ] Accessions checked against [InterPro](https://www.ebi.ac.uk/interpro/)
