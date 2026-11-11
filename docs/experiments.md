# Experiments, configurations and benchmark methodology

## Configuration

`configs/base.yaml` holds the shared defaults; each model config `extends` it. YAML is parsed into frozen dataclasses (`engine_rul.config`), and unknown keys or invalid values raise `ConfigError`. A config's **hash** covers everything that defines the experiment (model, data, training), but not the seed, device, or output paths, so the five seeds of one config share a hash.

| File | Purpose |
|---|---|
| `configs/baseline_constant.yaml`, `baseline_life_prior.yaml`, `ridge.yaml`, `hgb.yaml`, `lstm.yaml` | starting configs |
| `configs/grids/*.yaml` | hyperparameter grids, listed simplest first (ties within 0.01 RMSE go to the earlier entry) |
| `configs/final/*.yaml` | **frozen** configs written by `engine-rul tune` (header records the selection score and run) |
| `configs/sensitivity.yaml` | EXP-04 definition |

## Experiment sequence (all commands are in the `Makefile`)

| ID | Command | Data | Output |
|---|---|---|---|
| EXP-01–03 | `engine-rul tune --config … --grid …` | train → validation | `results/final/tuning_<family>.json`, `configs/final/<family>.yaml` |
| EXP-04 | `engine-rul sensitivity` | train → validation | `results/final/sensitivity.json` |
| EXP-05 | `engine-rul final --family all --seeds 0 1 2 3 4` | train → validation (early stopping) | `results/final_runs.json`, `results/<run_id>/` |
| EXP-06 | `engine-rul evaluate --split test --final --family all` | official test | `results/final/summary.json`, `results/test_evaluations.jsonl` |
| EXP-07 | `engine-rul benchmark --family all --devices cpu mps` | test **sensor windows** only | `results/final/benchmarks.json` |
| — | `engine-rul report` | committed results | `reports/report.md`, figures, README block |

Grid budgets: Ridge 17 α values; HGB 24 configs; LSTM 8 configs (`hidden_size` {32, 64} × `num_layers` {1, 2} × `dropout` {0.1, 0.3}); LSTM training: Adam lr 1e-3, batch 256, ≤ 100 epochs, patience 10, min_delta 0.01, gradient clip 1.0. The Ridge and HGB grids were extended beyond the PRD because the first optimum sat on a grid edge (`docs/decisions.md` D-03).

Sensitivity outcome (validation only): R_max = 125 and W = 30 gave the lowest validation RMSE against a fixed truth capped at 125 for both HGB and the LSTM, so the defaults were kept. Uncapped validation RMSE favours uncapped training, but the uniform validation truncation points include many early-life points with RUL up to ~300, unlike the test labels (max 145), so that metric was not used to change defaults.

## Tracking and artifacts

Each run writes `results/<run_id>/` (git-ignored) containing:

- `run.json`: resolved config, config hash, seed, git commit and dirty flag, data/split/preprocessor hashes, environment, validation metrics, timings, fit info
- the model artifact: `model.json` (baselines), `model.joblib` (sklearn), or `checkpoint.pt` plus `best.pt` (LSTM, loaded with `weights_only=True`)
- `preprocessor.json`, plus `predictions_val.csv` and `predictions_val_dense.csv`

The same run is logged to MLflow (`sqlite:///mlflow.db`, experiments `fd001-dev` and `fd001-final`). If MLflow fails, the run still completes and writes JSON. Small summaries needed by the report are copied into the committed `results/final/`.

## Benchmark methodology

- **Training time:** wall-clock `perf_counter` around `fit` for the median-seed final run (CPU).
- **Latency:** one test window per call, 20 warm-up + 200 timed calls; median and p95. Tabular models include feature extraction. On MPS, the timer reads after `torch.mps.synchronize()`.
- **Throughput:** all 100 test windows in one call, median of 20.
- **Size:** artifact bytes on disk, plus parameter count (LSTM), coefficients (Ridge) or total tree nodes (HGB).
- **Environment:** recorded in `benchmarks.json`. Numbers are comparable only within one machine.
- The HGB latency (~10 ms) is dominated by scikit-learn's per-call overhead for a single row, not by model size; its batch throughput is reported separately.
