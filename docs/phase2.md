# Phase 2 — validation-only evaluation and reliability

Results: [`reports/phase2_report.md`](../reports/phase2_report.md). Decisions: [`decisions.md`](decisions.md) (P2-01…P2-09).

**Ground rule.** The official FD001 test set was evaluated once in Phase 1. Phase 2 uses the 100
training engines only; no Phase-2 code path reads `test_FD001.txt` or `RUL_FD001.txt` (the
benchmarks also use training-file sensor rows). Phase-1 frozen configs, `results/final/`,
`results/final_runs.json` and `results/test_evaluations.jsonl` are unchanged and write-protected
(`--force` needed). Nothing learned here was used to change the Phase-1 models.

## What was found in the Phase-1 protocol

| Aspect | Phase 1 | Issue | Phase 2 |
|---|---|---|---|
| Validation | one 80/20 engine split, 20 engines | high variance; the LSTM/HGB order flipped on test | 3 × 5-fold grouped CV, every engine held out once per repeat |
| Selection vs evaluation | the same 20 engines | selection bias in the reported validation score | nested: inner split selects, outer fold only scores |
| LSTM early stopping | dense validation windows | different objective from selection (10 endpoints/engine) | stops on the inner selection endpoints |
| Endpoints | uniform, 10 per engine | one strategy only | uniform / late-life / stage-stratified, identical across models |
| Inference | point estimates and bootstrap CIs of aggregate metrics | no per-engine uncertainty | split-conformal intervals |

## Cross-validation (`cv.py`, `configs/phase2/cv.yaml`)

* **Folds** (`splits/cv_fd001.json`, committed): engines sorted by lifetime are cut into blocks of
  5, and each block's engines go to distinct folds at random (seed per repeat: 101, 202, 303).
  Every engine is in exactly one fold per repeat. The file records the training-file SHA-256.
* **Nested roles.** For outer fold *k*, the development engines (the other 80) are split by
  engine, also lifetime-stratified, into inner-train (64) and inner-val (16). The preprocessor
  (sensor selection, z-score statistics) is fit on inner-train rows only. Every grid config is
  trained on inner-train and scored on inner-val endpoints. The best config (ties within 0.01 go
  to the earlier, simpler entry, as in Phase 1) then predicts the outer endpoints. The outer
  engines are never used for fitting, preprocessing, selection or early stopping.
* **Seeds.** Model seed = repeat index. Endpoints are seeded per (endpoint seed, salt, strategy,
  engine), so an engine's endpoints don't depend on which other engines share its fold. Outer
  endpoints are salted by the repeat; inner endpoints are also salted by the outer fold.

### Endpoint (truncation) strategies

For each engine with life *T*, candidate cut cycles are `min_cycle … T` (min 40). Ten per engine:

| Strategy | Draw | Measures | Assumption / trade-off |
|---|---|---|---|
| `uniform` | uniform over candidates | average accuracy over the observable life | many early-life points whose capped truth is 125, so errors there are small and flatter the score |
| `late` | uniform over cuts with true RUL ≤ 125 | accuracy where maintenance decisions are made | ignores the plateau. Closer to "the engine is degrading" use cases |
| `stratified` | round-robin across 25-cycle true-RUL stages | equal weight per degradation stage | the stage mix is a design choice and up-weights rare very-late stages |

`uniform` is the selection objective, the same as Phase 1. The other two are scored on the same
outer models, and the selection table records which config each strategy *would* have picked.
None of them was chosen by looking at the test set. The test set's truncation process (a
truncated trajectory, label up to 145) is not used to argue for any strategy.

### Statistics

* **Unit of inference = engine.** Errors are averaged within each engine first, then engines
  get equal weight. Endpoints or windows are never treated as independent.
* **Fold metrics:** RMSE/MAE/NASA per outer fold (20 engines), shown as mean ± sd and range.
  This spread is the most honest description of variability.
* **Pooled CIs:** a percentile cluster bootstrap over engines (2000 resamples) on the per-engine
  errors pooled over all repeats. **Approximate:** it ignores the dependence that overlapping CV
  training sets and selection noise create, so it is likely somewhat too narrow.
* **Paired comparisons:** Δ RMSE between two models on the same engines and endpoints, with a
  paired engine bootstrap. "Better" is only claimed when the CI excludes 0.
* **Ranking stability:** mean rank and rank-1 share across the 15 outer folds; rank-1 share
  under the engine bootstrap; Kendall τ between strategy rankings.

## Ablations (`configs/phase2/ablation.yaml`)

Each variant takes the CV run's per-fold selected hyperparameters and changes exactly one thing:
input channels (`sensors_only`, `cycle_only`), tabular window statistics (`stats_last`,
`stats_last_mean`, `stats_last_mean_slope`, versus all four in `base`), LSTM window length
(20/40 vs 30), LSTM depth (1/2), and the early-stopping set (`early_stop_dense`). Same inner
roles, same outer endpoints (uniform), Δ paired on engines. The recomputed `base` must reproduce
the CV predictions exactly, which doubles as a determinism check. Tabular models use 3 repeats;
LSTM variants use repeat 0 only (cost).

## Prediction intervals (`conformal.py`, `configs/phase2/conformal.yaml`)

* Same outer folds (repeat 0). Development engines → inner-train 48 / inner-val 12 / calibration
  20, by engine. Calibration engines are never used for fitting, preprocessing, selection or early
  stopping (tested in `tests/test_phase2.py`).
* Score `s = |min(RUL, 125) − ŷ|`. Quantile `q = s₍ₖ₎` with `k = ⌈(n+1)·level⌉` (∞ if k > n).
  Interval `[max(0, ŷ − q), ŷ + q]`.
* Methods:
  * `global_one_per_engine` (default): one seeded random endpoint per calibration engine
    (n = 20). **Guarantee:** if engines are exchangeable and a new engine is cut at a point drawn
    the same way (uniform), then P(capped RUL ∈ interval) ≥ level. The guarantee is marginal,
    averaged over engines and cut points. It is not conditional on stage, and it does not hold
    for cuts drawn differently.
  * `global_pooled`: all 10 endpoints per calibration engine. These scores are dependent within
    an engine, so they are not exchangeable with a new engine's score. **No finite-sample
    guarantee**; coverage is reported empirically.
  * `mondrian_pooled`: like pooled, but with a separate quantile per predicted-RUL class [0, 50),
    [50, 100), [100, ∞). This aims at more uniform coverage across stages. Same caveat as pooled.
* **Evaluation:** engine-weighted coverage and mean width on outer `uniform` endpoints (matching
  the calibration draw) and on `late` endpoints (a distribution-shift check), plus coverage by
  true-RUL stage. Coverage is also computed against uncapped truth, which early in life is lower
  by construction.
* **Serving:** each conformal fold model is saved as a standard run directory with
  `conformal.json`. `engine-rul predict --run <dir> --input <sensor file> --levels 0.8 0.9` adds
  `lower_*`/`upper_*` columns. Without `--levels`, the output is exactly the Phase-1 output
  (verified bit-identical on all five frozen models). Phase-1 runs have no calibration because
  their validation engines were used for selection.
* **Performance vs prediction uncertainty.** A CI for RMSE says how precisely we know a model's
  average error, and it shrinks with more engines. A prediction interval says where one engine's
  RUL probably lies; it is tens of cycles wide and only shrinks with a better model.

## Provenance and isolation (`provenance.py`)

* Every Phase-2 experiment writes `results/phase2/<category>/<name>/manifest.json` with: git
  commit and dirty flag; when dirty, a content-addressed source snapshot (`git diff HEAD` plus
  untracked source files) under `results/snapshots/<sha>.tar.gz`; the spec and its SHA-256; the
  fold-file and training-data SHA-256; library versions and device; and
  `official_test_set_used: false`. Phase-1 style runs (`train_one`) also record `provenance`.
* Categories: `cv`, `ablation`, `validation_strategy`, `conformal`, `benchmark`, `repro`,
  `exploratory`. An existing non-empty experiment directory is never reused without `--force`.
  The same applies to `tune`/`sensitivity`/`final`/`evaluate --split test` and the published
  files they would overwrite.
* MLflow: one run per experiment in `fd001-phase2-<category>`; repro runs go to
  `fd001-phase2-repro`.
* `engine-rul repro-check` retrains the frozen configs into `results/phase2/repro/<name>/` and
  compares validation RMSE and config hash with `results/final/summary.json`. On this machine
  the match is exact (|Δ| = 0). CI runs it for the four fast models with tolerance 1e-3 to allow
  for cross-platform BLAS/OpenMP differences. `tests/test_phase2.py` checks that CV is
  deterministic (two identical runs produce identical predictions).

## Benchmarks (`benchmark --phase2`)

Two scopes, both on Phase-1 median-seed models with training-file sensor rows:

* **Model inference:** `model.predict` on prepared windows, batch sizes 1/10/100/1000. Includes
  tabular feature extraction for Ridge/HGB, because that happens inside `predict`. Excludes I/O,
  preprocessing and windowing.
* **End-to-end:** `Predictor.predict` on raw rows of 1/10/100 engines, covering preprocessing,
  windowing, features, prediction, clipping and the output frame. Excludes file parsing and model
  loading (done once).

Median and p95 latency, throughput, tracemalloc peak (Python/NumPy allocations only, not
PyTorch's native allocator), and process peak RSS. 5 warm-up calls, 20–50 repetitions,
`torch.mps.synchronize()` after every MPS call.

## Commands

```bash
make cv            # nested CV          -> results/phase2/cv/fd001_nested_cv/
make ablation      # needs the CV run   -> results/phase2/ablation/fd001_ablation/
make conformal     #                    -> results/phase2/conformal/fd001_conformal/
make bench-phase2  #                    -> results/phase2/benchmark/fd001_latency/
make repro-check   # all frozen configs -> results/phase2/repro/frozen_configs/
make report-phase2 # reports/phase2_report.md + reports/figures/phase2/
make phase2        # cv ablation conformal bench-phase2 report-phase2
```

Add `FORCE=--force` to replace an existing experiment directory.
