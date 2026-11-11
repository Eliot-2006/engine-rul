# Architecture and design decisions

ADR-01..13 are copied from `PRD.md` §3.2. ADR-14 onward and the deviations (D-xx) were recorded
during implementation. Anything that changes a frozen config after a test evaluation goes under
*Post-test changes*.

## Architecture Decision Records

| ADR | Decision | Rationale | Tradeoff accepted |
|---|---|---|---|
| ADR-01 | Data source: HF mirror at a pinned revision, raw `.txt` only | The official download is unreliable. Pinning plus SHA-256 makes the data immutable for the project. Raw files avoid third-party preprocessing. | Byte-identity with NASA's original can't be proven until the official ZIP is reachable (`DATA-04` is the hook). |
| ADR-02 | Canonical feature set for FD001: 14 sensors (s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21) plus normalized current cycle; drop op1–op3 | Constant and binary sensors carry no information in FD001. Operating settings are noise around a single condition. This matches widespread practice. | Different feature sets in some published papers reduce comparability. Documented in the external reference table. |
| ADR-03 | Piecewise-linear target with R_max = 125; evaluate primarily against uncapped official labels | Early-life RUL isn't identifiable from sensors in a healthy engine. A capped target reduces irreducible noise in training. 125 is a widely used value. Evaluating against official labels keeps the headline metric faithful to the benchmark definition. | Errors on the 11 test engines with RUL > 125 are partly irreducible. A sensitivity study runs on validation with R_max ∈ {115, 125, 135, none} (`EXP-04`). |
| ADR-04 | Window length W = 30 | The shortest FD001 test trajectory is 31 cycles, so no padding is needed. 30 is common in the literature. | May not be optimal. W ∈ {20, 30} is checked as a validation ablation (`EXP-04`). |
| ADR-05 | One fixed, committed 80/20 engine-level split; final models trained on 80 engines with validation early stopping (no retrain on all 100) | Simple, identical for all model families, and the same early-stopping procedure is used in development and final training. | Final models use 20% less data than a full retrain. Split variance isn't captured unless OPT-1 (repeated CV) is added. Both are stated in Limitations. |
| ADR-06 | Tree model: `HistGradientBoostingRegressor`, not XGBoost | Already in scikit-learn (no new dependency), fast, strong on tabular data. | Slightly fewer knobs than XGBoost, which doesn't matter at this scale. |
| ADR-07 | MLflow for tracking; JSON in `results/` as source of truth | MLflow gives a browsable UI and artifact logging. Plain JSON keeps reports reproducible and diffable without an MLflow dependency at report time. | Small duplication: each run writes JSON and logs to MLflow through one helper (`tracking.py`). |
| ADR-08 | Primary sequence model: 2-layer LSTM | Well understood, widely used on C-MAPSS, easy to implement correctly, fast on CPU at this data size. | A TCN may train faster or perform differently. That is OPT-2. |
| ADR-09 | Configs: YAML parsed into frozen `dataclasses` with explicit validation | No extra framework (Hydra or pydantic not needed). Typed and immutable. | Manual validation code, kept small. |
| ADR-10 | CLI: stdlib `argparse` with subcommands under one console script `engine-rul` | Zero dependencies, sufficient for about 8 commands. | Less ergonomic than Typer or Click. Acceptable. |
| ADR-11 | Dependency management: `uv` with `pyproject.toml` and committed `uv.lock`; Python 3.11 or 3.12 | Fast, reproducible lockfile, a single tool for venv and run. | Requires installing `uv`. The README gives a `pip install -e .` fallback without lock guarantees. |
| ADR-12 | Test-set lock: test labels are only loadable through `evaluate --final`, and each use is logged | Makes honest test usage auditable. | Slightly more ceremony. The lock is a **procedural safeguard, not a security boundary**: the label file sits in `data/raw/` and anyone can read it directly or edit the code. What it guarantees is that the *pipeline* cannot touch test labels by accident, that each sanctioned use leaves an append-only log entry, and that overwriting the published `summary.json` needs an explicit `--force`. Honest use still depends on the author. |
| ADR-13 | Predictions clipped below at 0 for all models; no upper clip | RUL can't be negative. The same post-processing for every model keeps the comparison fair. | None material. |
| ADR-14 | Committed experiments train on **CPU** (`device: cpu` in `configs/base.yaml`); `--device auto`, `mps` and `cuda` remain supported | Measured on the development machine (Apple M3 Pro): LSTM epoch 1.28 s on CPU vs 1.70 s on MPS for the default config. CPU is also the deterministic reference device (REPRO-08). | GPU users can still opt in; their numbers may differ slightly (documented in README). |

## Deviations from the PRD and resolved ambiguities

| ID | What | Why |
|---|---|---|
| D-01 | Device default `cpu` instead of `auto` for committed runs | ADR-14 (measured; reproducibility). |
| D-02 | `engine-rul data verify` reads `RUL_FD001.txt` | DATA-10(g) requires checking the label range. The values are only range-checked and never reach any model, tuning or selection code; the import rule in EVAL-20 is otherwise respected (`train.py` never imports test loaders; enforced by `tests/test_leakage.py`). |
| D-03 | Hyperparameter grids extended at the boundary | Ridge: the PRD grid (α ∈ 1e-3..1e3) selected the boundary α = 1000 with monotone improvement, so the grid was extended to 1e5; the re-run selected the interior α = 1000. HGB: the PRD grid selected the boundary `max_leaf_nodes = 15`, so 7 was added (24 configs); 7 was selected but is only 0.057 RMSE better than 15 (flat landscape, below validation noise), so no further extension. Both decisions used validation data only, before any test evaluation. |
| D-04 | Validation metrics computed against three truths: uncapped, capped at the config's R_max, and capped at a fixed 125 | The fixed-125 variant makes the R_max sensitivity study (EXP-04) comparable across caps; model selection still uses the config-capped truth (EVAL-12). |
| D-05 | Frozen-run registry `results/final_runs.json`; the test evaluation re-checks each run's `config_hash` against `configs/final/<family>.yaml` | Concrete implementation of EVAL-22. |
| D-06 | HGB trained with 5 seeds although it is deterministic here | `early_stopping=False` and no feature subsampling mean `random_state` has no effect; the 5 seeds are kept for protocol uniformity and give std = 0 (expected, not a bug). |
| D-07 | `torch.use_deterministic_algorithms(True, warn_only=True)` | CPU runs are bit-reproducible (verified by `tests/test_reproducibility.py`); `warn_only` keeps MPS/CUDA runs from crashing on ops without deterministic kernels. |
| D-08 | Python 3.11–3.12 | PyTorch wheels; the machine's system Python 3.15 is not supported, so `uv` provides 3.12 (`.python-version`). |
| D-09 | The report reads small committed copies in `results/final/` (median-seed predictions, LSTM histories, importances) | So `engine-rul report` regenerates every table and figure from a fresh clone without retraining. Run folders under `results/<run_id>/` stay git-ignored. |
| D-10 | Exceptions live in `engine_rul/errors.py` | ERR-01 allowed `errors.py` or `utils.py`. |
| D-11 | Sensitivity study reviewed; defaults kept (R_max = 125, W = 30) | Lowest validation RMSE against the fixed capped-125 truth for both HGB and LSTM. Uncapped-truth validation RMSE favours uncapped training, but the uniform validation truncation points contain many early-life points (true RUL up to ~300) unlike the test labels (max 145), so it is not a sound basis for changing the cap. The first sensitivity run used a stale HGB config (it started before the HGB grid extension finished); it was re-run with the frozen configs before any final training. |
| D-12 | `make bench` benchmarks LSTM inference on CPU and MPS | BENCH-06. Training stays on CPU (ADR-14). |

## Phase 2 decisions (validation only; see [`phase2.md`](phase2.md))

| ID | Decision | Why |
|---|---|---|
| P2-01 | Phase 2 never reads the official test files; all new experiments use the 100 training engines | Test set already spent (one evaluation). Phase-2 outputs live in `results/phase2/` and are reported in `reports/phase2_report.md`, separate from `reports/report.md`. |
| P2-02 | Repeated (3 × 5-fold) lifetime-stratified, engine-grouped CV; nested inner split (80/20 of development engines) for preprocessing, hyperparameters and LSTM early stopping | Outer engines influence nothing but the score; lifetime stratification keeps folds comparable (lifetime is a strong driver of error). |
| P2-03 | Endpoints sampled per (seed, strategy, engine), `min_cycle = 40` | Identical endpoints for every model/variant; 40 ≥ the largest ablation window so window-length variants are scored on the same points. |
| P2-04 | Selection on uniform endpoints; late/stratified are reported, not selected on | Uniform matches the Phase-1 selection objective; the comparison shows how the choice would change rankings and selections, without picking a "winner" strategy using test data. |
| P2-05 | LSTM early stopping on inner-validation *selection endpoints* (`train.early_stopping: endpoints`) in Phase 2 | Aligns the stopping criterion with the selection objective. Phase-1 runs keep `dense`; the new field is omitted from config hashes at its default, so all frozen hashes are unchanged. |
| P2-06 | Statistical unit = engine (per-engine mean errors, engine cluster bootstrap) | Endpoints/windows from one engine are strongly dependent. CIs on pooled CV predictions are labelled approximate (training-set overlap ignored). |
| P2-07 | Ablations reuse the CV run's per-fold selected hyperparameters and change one thing | One dimension at a time, no re-tuning per variant (which would mix selection noise into Δ). |
| P2-08 | Split conformal with a disjoint calibration role (60/15/25 of development engines); one-endpoint-per-engine calibration as the default | The only variant with a finite-sample guarantee under engine exchangeability; pooled variants are reported as empirical. |
| P2-09 | Published artifacts are write-protected (`--force` needed) and experiments go to `results/phase2/<category>/<name>/` with a manifest and source snapshot | Prevents silent overwrites of frozen configs, published results and the test log. Consequence: `make reproduce` now needs `FORCE=--force`. |

## Post-test changes

The official test set was evaluated **once** (2026-11-11 03:38 UTC, 13 log entries = 1 constant +
1 life-prior + 1 Ridge + 5 HGB + 5 LSTM). No model, config, preprocessing, training or metric code
changed afterwards, and no model was retrained.

Changes made after that evaluation affect presentation only and do not touch any metric:

- `plots.py`: shorter hyperparameter labels in `hparam_sensitivity.png`.
- `report.py`: auto-generated findings now say that paired comparisons use the median seed, state
  which model led on validation when the validation and test rankings differ, and flag uncommitted
  working-tree changes at evaluation time.
- `ruff format` applied to `src/` and `tests/` (whitespace/line-wrapping only; the full test suite
  passed before and after).
- Phase 2 (`report.py`): the five gradient-boosting runs are labelled "5 (identical runs)" and the
  table footnote says they are bit-identical and not independent evidence of stability (D-06).
  Wording only; every number in `reports/report.md` is unchanged.
- Phase 2: docstrings now state that a single-layer LSTM has no inter-layer (recurrent) dropout and
  only the head dropout is active, which applies to the frozen LSTM (1 layer, dropout 0.3).

The validation→test ranking flip between LSTM and gradient boosting was **not** acted on: the LSTM
was not re-tuned and gradient boosting was not re-selected after seeing test results.
