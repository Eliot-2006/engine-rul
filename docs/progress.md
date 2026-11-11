# Progress log

Last updated: 2026-11-10 (local). Status: **MVP complete; Phase 2 (validation-only evaluation and reliability) complete**, with the follow-ups listed under *Remaining work*.

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| M0 Scaffold and tooling | ✅ | `pyproject.toml`, `uv.lock`, `Makefile`, CLI with 10 subcommands, `docs/decisions.md` (ADR-01..14) |
| M1 Data acquisition and verification | ✅ | `engine-rul data download --subsets all`: all 12 files match pinned SHA-256; `PROVENANCE.json` written |
| M2 Loading, schema, labels | ✅ | FD001 invariants (DATA-10 a–h) pass; `test_data.py` |
| M3 EDA | ✅ | `reports/eda_summary.json`, 6 figures + `constant_sensors.md` in `reports/figures/eda/` |
| M4 Split and preprocessing | ✅ | `splits/fd001_split.json` (80/20 engines, 200 eval points); 14 sensors selected on train-split rows; leakage tests |
| M5 Windows and features | ✅ | 17,731 windows over all training engines (14,217 train-split); 57 tabular features |
| M6 Metrics and evaluation | ✅ | hand-verified metric tests; `evaluate --split test` without `--final` exits 3 |
| M7 Tracking, harness, baselines | ✅ | MLflow (sqlite) + `run.json`; constant and life-prior baselines |
| M8 Classical models and tuning | ✅ | Ridge (17 α) and HGB (24 configs) tuned on validation; grids extended at the boundary (D-03) |
| M9 LSTM | ✅ | training loop, early stopping, checkpoints, CPU/MPS; CPU chosen (ADR-14) |
| M10 LSTM tuning, sensitivity, final seeds | ✅ | 8-config LSTM grid; sensitivity re-run with frozen configs (D-11); 5-seed final runs |
| M11 Test evaluation and benchmarks | ✅ | one logged evaluation (13 entries in `results/test_evaluations.jsonl`); `benchmarks.json` (CPU + MPS) |
| M12 Report and figures | ✅ | `reports/report.md`, 11 result/training figures, all generated from `results/final/` |
| M13 README and polish | ✅ | README with a generated results block; docs for dataset, experiments, testing, pipeline, decisions |

## Phase 2 milestones (validation only — the official test set was not used again)

| Milestone | Status | Evidence |
|---|---|---|
| P2-M1 Provenance and isolation | ✅ | `provenance.py`; manifests + source snapshots; `--force` needed to overwrite published artifacts |
| P2-M2 Repeated nested CV | ✅ | `splits/cv_fd001.json` (3 × 5 folds); `results/phase2/cv/fd001_nested_cv/` (48 min CPU) |
| P2-M3 Validation strategies | ✅ | uniform / late / stratified endpoints scored on the same models; rankings identical (τ = 1) |
| P2-M4 Ablations | ✅ | `results/phase2/ablation/fd001_ablation/`; recomputed base = CV predictions exactly |
| P2-M5 Prediction intervals | ✅ | `results/phase2/conformal/fd001_conformal/`; `engine-rul predict --levels` |
| P2-M6 Benchmarks | ✅ | model-only vs end-to-end, CPU + MPS: `results/phase2/benchmark/fd001_latency/` |
| P2-M7 Reproducibility check | ✅ | `engine-rul repro-check`: all 5 frozen configs reproduce validation RMSE exactly; CI smoke step |
| P2-M8 Report and docs | ✅ | `reports/phase2_report.md`, `docs/phase2.md`, README Phase-2 block (generated) |

## Tests

| Run | Result |
|---|---|
| `uv run pytest -q` (synthetic + real data) | **87 passed** (74 Phase 1 + 13 Phase 2) |
| `uv run ruff check src tests` | clean |
| `uv run ruff format --check src tests` | clean |
| `uvx mypy src --ignore-missing-imports` (not part of the PRD toolchain) | 27 findings, almost all "fitted attribute may be `None`" patterns in model/preprocessor classes; one real edge case (`train.final([])` with no families crashes). None affect recorded results |

Bugs found by tests during development: the LSTM checkpoint stored `torch.__version__` as a `TorchVersion` object, so loading with `weights_only=True` failed (fixed by storing a string); the hyperparameter plot crashed when a family had a single config (fixed); the E2E NaN check matched "mai**nan**ce" (fixed with a word-boundary regex).

## Experiments completed (all real runs, CPU, Apple M3 Pro)

| ID | What | Outcome |
|---|---|---|
| EXP-01 | Baselines | val RMSE (capped, 200 points): constant 40.44, life prior 30.93 |
| EXP-02 | Ridge, HGB tuning | Ridge α = 1000 (val 15.88); HGB lr 0.1 / 7 leaves / min_leaf 20 / 500 iters (val 14.00) |
| EXP-03 | LSTM tuning | H = 32, 1 layer, dropout 0.3 (val 13.58, seed 0) |
| EXP-04 | Sensitivity (R_max × W) | defaults (125, 30) best for both HGB and LSTM vs the capped-125 truth |
| EXP-05 | Final training, seeds 0–4 | LSTM seed 0 reproduced its tuning-run validation RMSE exactly |
| EXP-06 | Official test evaluation (once) | see `reports/report.md` §3; gradient boosting had the lowest test RMSE, and the LSTM (best on validation) was worse on test with a paired CI excluding 0 |
| EXP-07 | Benchmarks | latency, throughput, size; LSTM inference on CPU and MPS |
| P2-EXP-01 | Nested 3 × 5-fold CV | gradient boosting lowest RMSE (13.95, capped truth), best in 13/15 folds; vs LSTM Δ −1.23 [−2.14, −0.36] |
| P2-EXP-02 | Validation strategies | identical rankings; selected hyperparameters agree with uniform-endpoint selection in only 20–40 % of folds |
| P2-EXP-03 | Ablations | window slopes matter (+4–5 RMSE without them), std adds nothing; cycle channel helps the LSTM only; LSTM window 40 < 30 (validation-only observation, not adopted) |
| P2-EXP-04 | Split conformal | pooled/Mondrian coverage ≈ nominal; one-endpoint-per-engine (exact guarantee) correct on average but variable with 20 calibration engines |
| P2-EXP-05 | Benchmarks | end-to-end latency dominated by preprocessing/windowing (~18 ms for 100 engines) except for HGB (~9 ms fixed predict overhead) |

## Important decisions

See [`decisions.md`](decisions.md): ADR-14 (CPU training), D-02 (`data verify` range-checks labels), D-03 (grid extensions), D-04 (three validation truths), D-11 (sensitivity review), *Post-test changes* (presentation and formatting only).

## Issues encountered and resolutions

| Issue | Resolution |
|---|---|
| System Python 3.15 has no PyTorch wheels | `uv` with Python 3.12 (`.python-version`) |
| Official NASA ZIP unreachable | pinned HF mirror revision + SHA-256 + invariants; `verify --zip` hook |
| Ridge and HGB optima on grid edges | grids extended on validation data only (D-03) |
| First sensitivity run used a stale HGB config | re-ran before final training (D-11) |
| MPS slower than CPU for this small LSTM | CPU for committed runs (ADR-14) |
| Validation ranking (LSTM best) flipped on test (HGB best) | reported as a finding; nothing re-tuned after the test evaluation |

## Remaining work / known gaps

- Published benchmark numbers in `docs/external_results.md` are unfilled: primary sources weren't accessible to verify (EVAL-41).
- Byte-identity of mirror files with NASA's original ZIP remains unverified (needs the ZIP).
- Phase 1 results record `git_dirty = true` at commit `c457958` without a source snapshot (snapshots were added in Phase 2); Phase 2 results reference committed, hash-verified snapshots. Re-running from a clean commit would give `git_dirty = false`.
- Phase-1 figures use tab10 green/red for gradient boosting/LSTM, which fails a colour-vision-deficiency check (ΔE 3.9); Phase-2 figures use a validated palette. Recolouring Phase-1 figures was left alone to keep that report intact.
- Optional PRD extensions not started: OPT-2 (TCN), OPT-3 (FD002–FD004), OPT-4 (SHAP), OPT-6 (random forest), OPT-8 (Docker). (OPT-1 repeated CV and OPT-7 prediction intervals were done in Phase 2, OPT-5 the dashboard in Phase 3.)
- Optional cleanup: add `mypy` with explicit fitted-state asserts.

## Continuation instructions

1. `make setup data` on a fresh clone; `uv run pytest -q` runs the whole suite.
2. Reports, README blocks and dashboard data are regenerated without retraining via `make reports` (reads `results/final/` and `results/phase2/` only).
3. `make reproduce-phase1 FORCE=--force` (~15 min CPU) retrains Phase 1 and **overwrites** published Phase-1 run artifacts; it does not touch the test set. To check reproducibility without touching anything, use `make repro-check`.
4. Phase 2: `make phase2` (~1.5 h CPU) re-runs CV, ablations, conformal and benchmarks into `results/phase2/` (refuses to overwrite; add `FORCE=--force`), then `make report-phase2`.
5. The official test evaluation is a separate opt-in: `make official-test-eval CONFIRM=official-test`. It appends a round to `results/test_evaluations.jsonl`, which must be explained in `docs/decisions.md`.
6. Dashboard: `make dashboard`; deployment is `.github/workflows/pages.yml` (needs *Settings → Pages → Source: GitHub Actions* once).
