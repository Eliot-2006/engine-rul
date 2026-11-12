# Testing and reproducibility

## Running tests

```bash
uv run pytest -q                 # everything (real-data tests skip if data/raw is missing)
uv run pytest -q -m "not data"   # synthetic-only (no download needed; what CI runs first)
uv run pytest -q -m data         # real FD001 checks (after `make data`)
uv run ruff check src tests
uv run ruff format --check src tests
make dashboard        # export data, npm ci, TypeScript check, build, bundle checks
make dashboard-test   # Playwright browser tests against the build (installs Chromium once)
```

`npm run build` runs `dashboard/scripts/check-dist.mjs`: every asset referenced by `index.html` must
exist under the `/engine-rul/` base path, all seven data files must be present with the expected
`protocol`, the canonical / Open Graph / Twitter tags must point at the bundled 1200×630 preview image,
and the bundle must stay under 1.5 MB. The browser tests (`dashboard/e2e/`) are described in
[`dashboard/README.md`](../dashboard/README.md#tests).

CI: the `ci` workflow runs lint, the synthetic tests, the dataset download, the real-data tests and a
`repro-check` smoke test. The `pages` workflow runs `tests/test_phase3.py` and
`tests/test_consistency.py`, the export, the type check and the build (`build` job), then Playwright
against that exact bundle (`e2e` job, Chromium headless shell only). Pull requests never deploy;
pushes to `main` deploy only after `e2e` passes.

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
| `test_phase3.py` | dashboard export writes every file with its protocol tag; headline derived from `summary.json`; official-test and validation results never share a file; consistent model identifiers and units; missing / malformed / NaN / unknown-model / test-set-flagged artifacts rejected with nothing written; frontend protocol map and schema version match the exporter; Pages base path matches the repository; README generated blocks are current; local links and images in README and `docs/` resolve; every published Phase-2 manifest's source snapshot is committed and hash-verified; `reproduce`, `reproduce-phase1`, `phase2` and `reports` never run `--split test`, and `official-test-eval` refuses without `CONFIRM` |
| `test_consistency.py` | README tables, `reports/report.md`, dashboard data and the dashboard's static link-preview text all show the `summary.json` numbers; the 100 official test predictions are complete, finite, non-negative and reproduce each model's published RMSE / MAE (within its bootstrap CI for the LSTM's median seed); 11 test engines above the 125 cap; every CI contains its estimate and "clear difference" means exactly "CI excludes 0" (official, CV, ablation), conformal coverage lies within its fold range; every dashboard source file is committed; the export is byte-for-byte deterministic; the preview image is 1200×630; `make reproduce` and `make repro-check` write only under `build/reproduce/` |
| `test_phase2.py` | CV folds partition engines and are deterministic; overlapping fold files rejected; inner roles disjoint; endpoints deterministic, valid and independent of fold membership; per-fold preprocessor fit on inner-train only; identical endpoints for every model; CV determinism (two runs → identical predictions); calibration engines disjoint from fit/selection/outer; conformal finite-sample coverage on exchangeable synthetic scores; Mondrian bins; intervals don't change point predictions; experiment-directory isolation; published test summary and Phase 1 benchmarks write-protected; benchmarks pick the median seed from `summary.json` or, in an isolated reproduction, the first seed of `final_runs.json`; frozen config hashes unchanged |

## Reproducibility notes

- **Environment:** `uv.lock` pins every package. `uv sync --frozen` recreates the environment; Python 3.12 comes from `.python-version`.
- **Seeds:** `set_seed` seeds Python, NumPy and torch. The split and evaluation points use `numpy.random.default_rng(20261110)`; DataLoader shuffling uses a seeded `torch.Generator`.
- **Determinism:** CPU training uses `torch.use_deterministic_algorithms(True)`. Evidence: the frozen LSTM config retrained in the final stage produced exactly the validation RMSE recorded during tuning (13.583), and recomputing validation metrics from saved artifacts (`engine-rul evaluate --split val --run results/<run_id>`) matched `run.json` to six decimals for the LSTM and HGB runs checked.
- **Phase 2:** `engine-rul repro-check` retrains frozen configs in isolation and compares with the published validation RMSE (exact on this machine; CI runs it with tolerance 1e-3). The ablation run recomputes the CV base predictions and records the max difference (see `reports/phase2_report.md`).
- **Devices:** MPS/CUDA results are expected to differ slightly from CPU. The committed results are CPU-only.
- **Data versioning:** each run records the SHA-256 of the training file, the split file and the fitted preprocessor; each test evaluation records the test and label file hashes.
- **Test-set honesty:** `results/test_evaluations.jsonl` is append-only and committed; `docs/decisions.md` lists any post-test changes.
- **Provenance:** Phase 1 was trained and evaluated at commit `c457958` with uncommitted changes (recorded as `git_dirty: true` in every `run.json` and test-log entry); Phase-1 runs predate source snapshots, so that exact working tree is not archived. What is verifiable: the frozen configs are hash-checked against `results/final/summary.json` and `repro-check` reproduces their validation RMSE exactly from the current code. Phase-2 manifests reference committed, hash-verified source snapshots (see `docs/phase2.md`). New experiments run from a dirty tree write a snapshot automatically; force-add it with the results (`git add -f results/snapshots/<id>.tar.gz`; `tests/test_phase3.py` fails until you do), or run from a clean commit.
- **Workflows and the test set:** `make reproduce` (= `reproduce-phase1` + `phase2`) never reads the official test labels, and neither does `make reports`, which only regenerates reports, README blocks and dashboard data from the committed results. The original evaluation is a separate, explicit step: `make official-test-eval CONFIRM=official-test` (add `FORCE=--force` to replace `results/final/summary.json`); it appends a round to the append-only log.
- **Isolated reproduction:** training and experiment targets write under `OUT` (default `build/reproduce/`, git-ignored), laid out like the repository: `results/` (runs, `final_runs.json`, tuning and sensitivity JSON, benchmarks), `configs/final/` (re-selected configs), `results/phase2/` and `reports/` (EDA figures, `phase2_report.md`). A fresh clone can therefore run `make reproduce` and compare with the published files without overwriting them; running again into the same `OUT` refuses to overwrite (pass `FORCE=--force` or another `OUT`). `OUT=.` targets the published locations, still guarded by `FORCE`. What stays fixed: Phase 2 uses the published frozen configs (`configs/final/`) exactly as the original Phase 2 did, and the ablation's determinism check compares its recomputed base models with the published CV predictions. Without a test evaluation there is no median seed, so an isolated benchmark times the first seed (latency does not depend on the seed). `make repro-check` is the quick automated comparison: it retrains every frozen config with seed 0 into `OUT` and compares with the published validation RMSE.
- **Runtime and hardware:** measured on the development machine (Apple M3 Pro, 11 cores, 18 GB RAM, CPU only; no GPU needed). `make reproduce-phase1` ≈ 15 min (HGB tuning ≈ 3 min, LSTM tuning ≈ 3.5 min, sensitivity ≈ 3 min, 5-seed final training ≈ 2.5 min; baselines, Ridge, EDA and benchmarks under a minute each); `make phase2` ≈ 1.5 h (nested CV alone 48 min); `make repro-check` < 1 min. Linux works too (CI); on non-Apple machines the benchmarks run on CPU only (`DEVICES`, default `cpu mps` on macOS). Expect small last-digit differences on other platforms and BLAS builds (CI tolerance 1e-3).
