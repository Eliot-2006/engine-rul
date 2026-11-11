# Testing and reproducibility

## Running tests

```bash
uv run pytest -q                 # everything (real-data tests skip if data/raw is missing)
uv run pytest -q -m "not data"   # synthetic-only (no download needed; what CI runs first)
uv run pytest -q -m data         # real FD001 checks (after `make data`)
uv run ruff check src tests
cd dashboard && npm ci && npm run check && npm run build   # dashboard type check, build, bundle checks
```

`npm run build` runs `dashboard/scripts/check-dist.mjs`: every asset referenced by `index.html` must
exist under the `/engine-rul/` base path, all seven data files must be present with the expected
`protocol`, and the bundle must stay under 1.5 MB. The `pages` workflow runs `tests/test_phase3.py`,
the export, the type check and the build on every pull request.

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
| `test_phase2.py` | CV folds partition engines and are deterministic; overlapping fold files rejected; inner roles disjoint; endpoints deterministic, valid and independent of fold membership; per-fold preprocessor fit on inner-train only; identical endpoints for every model; CV determinism (two runs → identical predictions); calibration engines disjoint from fit/selection/outer; conformal finite-sample coverage on exchangeable synthetic scores; Mondrian bins; intervals don't change point predictions; experiment-directory isolation; published test summary write-protected; frozen config hashes unchanged |

## Reproducibility notes

- **Environment:** `uv.lock` pins every package. `uv sync --frozen` recreates the environment; Python 3.12 comes from `.python-version`.
- **Seeds:** `set_seed` seeds Python, NumPy and torch. The split and evaluation points use `numpy.random.default_rng(20261110)`; DataLoader shuffling uses a seeded `torch.Generator`.
- **Determinism:** CPU training uses `torch.use_deterministic_algorithms(True)`. Evidence: the frozen LSTM config retrained in the final stage produced exactly the validation RMSE recorded during tuning (13.583), and recomputing validation metrics from saved artifacts (`engine-rul evaluate --split val --run results/<run_id>`) matched `run.json` to six decimals for the LSTM and HGB runs checked.
- **Phase 2:** `engine-rul repro-check` retrains frozen configs in isolation and compares with the published validation RMSE (exact on this machine; CI runs it with tolerance 1e-3). The ablation run recomputes the CV base predictions and records the max difference (see `reports/phase2_report.md`).
- **Devices:** MPS/CUDA results are expected to differ slightly from CPU. The committed results are CPU-only.
- **Data versioning:** each run records the SHA-256 of the training file, the split file and the fitted preprocessor; each test evaluation records the test and label file hashes.
- **Test-set honesty:** `results/test_evaluations.jsonl` is append-only and committed; `docs/decisions.md` lists any post-test changes.
- **Provenance:** Phase 1 was trained and evaluated at commit `c457958` with uncommitted changes (recorded as `git_dirty: true` in every `run.json` and test-log entry); Phase-1 runs predate source snapshots, so that exact working tree is not archived. What is verifiable: the frozen configs are hash-checked against `results/final/summary.json` and `repro-check` reproduces their validation RMSE exactly from the current code. Phase-2 manifests reference committed, hash-verified source snapshots (see `docs/phase2.md`). New experiments run from a dirty tree write a snapshot automatically; force-add it with the results (`git add -f results/snapshots/<id>.tar.gz`; `tests/test_phase3.py` fails until you do), or run from a clean commit.
- **Workflows and the test set:** `make reproduce` (= `reproduce-phase1` + `phase2` + `reports`) never reads the official test labels. The original evaluation is a separate, explicit step: `make official-test-eval CONFIRM=official-test` (add `FORCE=--force` to replace `results/final/summary.json`); it appends a round to the append-only log.
- **Not reproducible from a fresh clone without retraining:** per-run folders (`results/<run_id>/`, models) are git-ignored. `make reproduce-phase1` regenerates them in about 15 minutes on the development machine (Apple M3 Pro, CPU). Measured stage times from the logs: HGB tuning ≈ 3 min, LSTM tuning ≈ 3.5 min, sensitivity study ≈ 3 min, 5-seed final training ≈ 2.5 min; baselines, Ridge, test evaluation, benchmarks and report take under a minute each.
