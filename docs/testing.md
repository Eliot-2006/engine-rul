# Testing and reproducibility

## Running tests

```bash
uv run pytest -q                 # everything (real-data tests skip if data/raw is missing)
uv run pytest -q -m "not data"   # synthetic-only (no download needed; what CI runs first)
uv run pytest -q -m data         # real FD001 checks (after `make data`)
uv run ruff check src tests
```

Unit tests use a synthetic, C-MAPSS-shaped dataset built in `tests/conftest.py` (10 training and 5 test engines, the same constant/two-valued columns as FD001, trailing spaces). No test asserts model accuracy on real data.

| File | Covers (PRD IDs) |
|---|---|
| `test_data.py` | parsing, schema violations, checksum tamper → delete, provenance, RUL labels, label↔unit mapping, real FD001 invariants (TEST-DATA-*, TEST-RUL-*) |
| `test_split.py` | disjoint/complete engine split, determinism, eval-point rules, committed split = regenerated (TEST-SPLIT-*) |
| `test_preprocess.py` | sensor selection, scaling, `expected_features` assertion, persistence round trip |
| `test_leakage.py` | scaler/selection fitted on train rows only, single-engine causal windows, train/tune never touch test files, `--final` lock, features independent of targets (TEST-LEAK-*) |
| `test_features.py` | window counts/content, padding, contract, slope math, real window count 17,731 (TEST-WIN-*) |
| `test_metrics.py` | hand-computed RMSE/MAE/NASA values, asymmetry, validation, bootstrap, stage bins (TEST-MET-*) |
| `test_models.py` | output shapes for all families, LSTM parameter-count formula, baseline closed forms, LSTM overfits a small batch (TEST-MOD-*) |
| `test_training.py` | checkpoint reload with `weights_only=True`, early stopping + best-weight restore, artifact round trips, `run.json` fields, MLflow failure tolerance, failed-run marking (TEST-TRAIN-*) |
| `test_fair_comparison.py` | identical evaluation points and clipping across families (TEST-FAIR-01) |
| `test_reproducibility.py` | same seed → same predictions (Ridge/HGB 1e-6, LSTM 1e-3 on CPU), config hashing, config validation (TEST-REPRO-*) |
| `test_e2e_smoke.py` | tune → final → locked test eval → benchmark → report on synthetic data; `summary.json` schema; `final` refuses unfrozen configs; test eval rejects non-final runs (TEST-E2E-01) |

## Reproducibility notes

- **Environment:** `uv.lock` pins every package. `uv sync --frozen` recreates the environment; Python 3.12 comes from `.python-version`.
- **Seeds:** `set_seed` seeds Python, NumPy and torch. The split and evaluation points use `numpy.random.default_rng(20261110)`; DataLoader shuffling uses a seeded `torch.Generator`.
- **Determinism:** CPU training uses `torch.use_deterministic_algorithms(True)`. Evidence: the frozen LSTM config retrained in the final stage produced exactly the validation RMSE recorded during tuning (13.583), and recomputing validation metrics from saved artifacts (`engine-rul evaluate --split val --run results/<run_id>`) matched `run.json` to six decimals for the LSTM and HGB runs checked.
- **Devices:** MPS/CUDA results are expected to differ slightly from CPU. The committed results are CPU-only.
- **Data versioning:** each run records the SHA-256 of the training file, the split file and the fitted preprocessor; each test evaluation records the test and label file hashes.
- **Test-set honesty:** `results/test_evaluations.jsonl` is append-only and committed; `docs/decisions.md` lists any post-test changes.
- **Not reproducible from a fresh clone without retraining:** per-run folders (`results/<run_id>/`, models) are git-ignored. `make reproduce` regenerates them in about 15 minutes on the development machine (Apple M3 Pro, CPU). Measured stage times from the logs: HGB tuning ≈ 3 min, LSTM tuning ≈ 3.5 min, sensitivity study ≈ 3 min, 5-seed final training ≈ 2.5 min; baselines, Ridge, test evaluation, benchmarks and report take under a minute each.
