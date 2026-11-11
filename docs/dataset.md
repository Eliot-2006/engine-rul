# Dataset provenance and preparation

## Source

| | |
|---|---|
| Dataset | NASA C-MAPSS Jet Engine Simulated Data (PCoE). Saxena, Goebel, Simon, Eklund, PHM08 (2008) |
| Official listing | https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data (ZIP at `data.nasa.gov/docs/legacy/CMAPSSData.zip`) |
| Official download status | Not reachable from the development environment (S3 redirect returned 403; a ranged GET timed out; checked 2026-11-10) |
| Mirror used | https://huggingface.co/datasets/SoyVitou/NASA-C-MAPSS-Turbofan-Engine, **pinned to revision `4d14a0bf6248235bce31cbdacb297e03e0f75d52`** |
| Files used | `data/{train,test,RUL}_FD00x.txt` (raw). The mirror's `processed/*.parquet` (its own RUL column and naming) is **not** used |
| License | data.gov: public access, no license field. Mirror: `license: other`. No license is claimed here; the data is downloaded at setup and never committed |

## Integrity checks (`engine-rul data download` / `verify`)

1. Every file's SHA-256 is checked against [`configs/dataset_manifest.yaml`](../configs/dataset_manifest.yaml). A mismatched file is deleted and the command exits with code 2.
2. The downloaded files also match the git blob hashes published by the mirror's API (checked while writing the PRD).
3. Structural invariants for FD001 (DATA-10): 20,631 train / 13,096 test rows; units exactly 1–100 in both; cycles start at 1 and are contiguous; no missing or non-finite values; training lifetimes 128–362; test lengths 31–303; 100 test labels in 7–145.
4. `data/raw/PROVENANCE.json` records the source, revision, download time and per-file hashes.
5. `engine-rul data verify --zip CMAPSSData.zip` compares NASA's original archive with the manifest whenever it becomes available. This is the only missing provenance step.

**Discrepancy noted:** data.gov and the mirror README both list FD004 as 248 training and 249 test trajectories; the files contain 249 training and 248 test units (with 248 labels). The files are internally consistent and are treated as authoritative. FD004 is outside the MVP.

## Parsing

Whitespace-separated, no header, 26 numeric columns, **trailing spaces on every line**, so parsing uses `sep=r"\s+"`. A single-space delimiter would produce empty extra columns. Columns: `unit, cycle, op1–op3, s1–s21` (sensor meanings in `engine_rul.data.SENSOR_DESCRIPTIONS`).

## Preparation steps

| Step | Rule | Fitted on |
|---|---|---|
| RUL label | `RUL = T_u − t` (0 at the last cycle). Test: `RUL_FD001.txt[i]` ↔ unit `i+1` | — |
| Target | `min(RUL, 125)` for training; uncapped RUL kept alongside | — |
| Split | 80 / 20 engines, lifetime-stratified (20 bins of 5 engines, 1 per bin to validation), seed 20261110, committed in `splits/fd001_split.json` | all training engines (labels unused) |
| Validation eval points | 10 truncation cycles per validation engine, uniform in `[30, T_u]` | — |
| Column selection | drop op1–op3 (FD001 has one operating condition); drop sensors with ≤ 2 distinct values (s1, s5, s6, s10, s16, s18, s19). 14 sensors remain | train-split rows |
| Normalization | z-score per sensor (`ddof=0`) | train-split rows |
| Cycle feature | `cycle / max training lifetime` | train-split rows |
| Windows | 30 cycles ending at each cycle `t ≥ 30`, per engine (14,217 training windows); test windows end at each engine's last observed cycle | — |
| Tabular features | per sensor over the window: last, mean, std, OLS slope; plus `cycle_norm` = 57 | computed from the window only |

EDA outputs (all from training data): [`reports/eda_summary.json`](../reports/eda_summary.json) and [`reports/figures/eda/`](../reports/figures/eda/).
