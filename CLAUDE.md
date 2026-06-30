# Prot2Vec

Benchmarking toolkit comparing protein sequence embedding methods (amino acid composition, k-mer TF-IDF, ESM-2, Google Gemini) on Pfam family separability.

## Layout

- `src/` — core package, `import src` (`data/`, `embedders/`, `reduction/`, `evaluation/`, `visualization/`, `pipeline.py`, `cli.py`)
- `configs/experiments/` — YAML configs (`quick.yaml`, `full.yaml`, `llm.yaml`)
- `src/cli.py` — console entry points: `main` (`prot2vec-benchmark`) and `download` (`prot2vec-download`)
- `tests/` — pytest suite (import via `from src...`)

## Where to find things

- **README.md** — installation, full CLI/library usage, config schema, hardware notes
- **configs/default.yaml** — canonical config with every option commented
- **configs/experiments/** — ready-made experiment presets (start from `quick.yaml`)
- **src/pipeline.py** — orchestration of embed → reduce → evaluate; `RunConfig` shape
- **src/embedders/base.py** — `SequenceEmbedder` ABC; all embedders implement `fit_transform` + `name`
- **data/raw/** — cached Pfam-A.seed.gz (~500 MB, downloaded once)

## Project-specific knowledge

- **Pfam IDs** (e.g. `PF00069`, `PF00072`) are EBI protein family accessions; the seed alignment is the curated subset, not the full family.
- **Sequences are stripped to the 20 standard AAs** before embedding — gaps (`-`), `X`, `B`, `Z`, `U`, `O` are removed by `ProteinDataset.from_pfam_records`.
- **Two metrics**: `trustworthiness` (local-neighborhood preservation post-reduction) and `knn_cv_accuracy` (5-NN family separability). Both are reported per `(embedder, reducer)` pair.
- **ESM-2** auto-detects device (MPS / CUDA / CPU); fp16 autocast on MPS. Use `esm2_t6_8M` on CPU.
- **LLM embedder** sends raw AA letters as plain text to a general-purpose embedding API — it's a baseline to show whether non-biological models incidentally pick up family structure.
- **Embedding cache**: `cache_embeddings: true` writes `.npy` files keyed by embedder name; delete `results/embeddings/` to force re-embed.

## Recurring mistakes to avoid

- **Don't commit `data/raw/`** — Pfam seed is ~500 MB and downloaded on demand. It's gitignored.
- **`GOOGLE_API_KEY` must be in `.env`** at repo root (loaded via `python-dotenv` in `src/cli.py`). Don't hardcode keys.
- **`test_perfect_preservation` in `tests/test_metrics.py` is a known-failing numerical assertion** unrelated to layout — don't chase it during refactors.

## Preferred patterns

- **New embedder** → subclass `SequenceEmbedder` in `src/embedders/`, expose via `_build_embedder` in `src/cli.py`, add a unit test in `tests/test_embedders.py`.
- **New reducer** → same pattern via `DimReducer` and `_build_reducer`.
- **Optional heavy deps** go behind pyproject extras (`[esm]`, `[llm]`) with a lazy `ImportError` message pointing at the extra.
- **CLI output** uses `rich` (panels, progress bars, tables) — match the existing style in `src/cli.py`.
- **Experiments are YAML, not CLI flags** — add a new file under `configs/experiments/` rather than extending argparse.

## Commands

```bash
pip install -e ".[esm,llm,dev]"
prot2vec-download --version 35.0 --cache-dir data/raw
prot2vec-benchmark --config configs/experiments/quick.yaml   # or: python -m src --config ...
pytest
```

Python ≥ 3.11.
