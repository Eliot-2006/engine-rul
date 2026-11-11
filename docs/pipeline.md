# Pipeline and module architecture

## Phase 1: end-to-end pipeline and the locked test evaluation

```mermaid
flowchart TD
  A[Pinned HF mirror<br/>raw train/test/RUL .txt] -->|SHA-256 verified| B[data.py<br/>parse + schema + DATA-10 invariants]
  B --> C[split.py<br/>80/20 engine-level split<br/>+ 200 val truncation points]
  C --> D[preprocess.py<br/>fit on 80 train engines only:<br/>sensor selection, z-score, cycle scale]
  D --> E[features.py<br/>30-cycle windows per engine<br/>+ 57 window statistics]
  E --> F1[Baselines<br/>constant, life prior]
  E --> F2[Ridge / HistGradientBoosting]
  E --> F3[LSTM - PyTorch]
  F1 & F2 & F3 --> G[Validation-only tuning<br/>configs/final/*.yaml frozen]
  G --> H[Final training, stage=final<br/>LSTM + HGB: seeds 0-4 HGB bit-identical<br/>baselines + Ridge: seed 0, deterministic]
  H --> I{{evaluate --split test --final<br/>run once, logged to test_evaluations.jsonl}}
  I --> J[results/final/summary.json]
  J --> K[report.py → reports/report.md,<br/>figures, README tables]
  J --> W[dashboard.py → README glance + hero,<br/>dashboard JSON]
  H -.-> M[(MLflow sqlite<br/>params, metrics, artifacts)]
```

`make official-test-eval CONFIRM=official-test` is the only workflow that runs step I; `make reproduce` stops at H.

## Phase 2: validation-only experiments (training engines only)

```mermaid
flowchart TD
  T[100 FD001 training engines] --> F[splits/cv_fd001.json<br/>3 repeats × 5 lifetime-stratified, engine-grouped folds]
  F --> O[Outer fold k: held-out engines]
  F --> D[Development engines]
  D --> R{inner roles}
  R --> IT[inner-train: preprocessing + fit]
  R --> IV[inner-validation: hyperparameters,<br/>LSTM early stopping]
  R -.->|conformal run only| CA[calibration engines<br/>never fit or selected on]
  IT & IV --> S[selected model per fold]
  S --> E[score held-out engines<br/>identical cut points: uniform / late / stratified]
  O --> E
  E --> SUM[cv summary: pooled + per-fold RMSE,<br/>paired engine bootstrap, rankings]
  S --> AB[ablations: same folds, same selected<br/>hyperparameters, one change per variant]
  AB --> ABS[ablation summary: Δ RMSE vs full model]
  CA --> Q[conformal quantiles]
  Q --> COV[intervals + coverage on held-out engines]
  SUM & ABS & COV --> REP[report_phase2.py → phase2_report.md<br/>dashboard.py → README + dashboard JSON]
```

Calibration engines are disjoint from the inner-train, inner-validation and held-out engines of
the same fold (`tests/test_phase2.py`). No Phase-2 step reads the official test files.

## Module responsibilities

| Module | Owns | Must not |
|---|---|---|
| `data.py` | download, checksum, parse, schema, RUL labels, `load_test_rul` | normalize or select features |
| `split.py` | engine-level split + validation truncation points | read test files |
| `preprocess.py` | `Preprocessor` (fit on train-split rows only) | see validation/test rows at fit time |
| `features.py` | `WindowSet`, windows, tabular stats from `X_seq` only | use targets to build features |
| `models/` | five families behind one fit/predict/save/load protocol | clip predictions or load data |
| `train.py` | single runs, tuning, sensitivity, final seeds | import test loaders |
| `evaluate.py` | validation re-evaluation, locked test evaluation, summary | change models |
| `metrics.py` | RMSE, MAE, NASA score, bootstrap, stage bins | depend on project modules |
| `tracking.py` | MLflow + `run.json` (JSON is the source of truth) | fail a run because MLflow failed |
| `benchmark.py` | latency, throughput, size | read test labels |
| `plots.py`, `eda.py`, `report.py` | figures and reports from saved results | recompute metrics |
| `cv.py` (Phase 2) | CV folds, nested roles, endpoint strategies, per-fold preparation, CV/ablation runners and engine-level analysis | read test files |
| `conformal.py` (Phase 2) | split-conformal calibration, intervals, coverage analysis | touch fitting/selection data |
| `inference.py` | `Predictor` / `predict_engines`: raw sensor rows → predictions (+ intervals if calibrated) | change point predictions |
| `provenance.py` | git/dirty/source snapshot, experiment directories, overwrite protection | — |
| `report_phase2.py` | `reports/phase2_report.md` from `results/phase2/` | recompute model predictions (except the interval figure) |
| `dashboard.py` (Phase 3) | validated dashboard JSON (one file per evaluation protocol), README result blocks, hero figure | train, read raw data or mix protocols in one file |
| `dashboard/` (Phase 3) | static Vite + TypeScript + Observable Plot site; loads the exported JSON and checks each file's `protocol` | compute statistics |

## LSTM architecture

```mermaid
flowchart LR
  X["window [B, 30, 15]<br/>14 z-scored sensors + cycle_norm"] --> L1[LSTM layer(s)<br/>hidden H]
  L1 --> Hn["last time step [B, H]"]
  Hn --> D1["Linear H→H/2 + ReLU + Dropout"]
  D1 --> O["Linear H/2→1"]
  O --> Y["× R_max → RUL (cycles)"]
```
