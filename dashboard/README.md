# EngineRUL results dashboard

A static, single-page dashboard (Vite + TypeScript + [Observable Plot](https://observablehq.com/plot/)) that
shows the committed EngineRUL results. It has no backend, needs no model or MLflow at runtime, and
reads only the JSON written by `engine-rul export-dashboard`.

```bash
make dashboard-data                     # or: uv run engine-rul export-dashboard
cd dashboard && npm ci
npm run dev                             # http://localhost:5173/engine-rul/
npm run check                           # TypeScript type check
npm run build                           # dashboard/dist + scripts/check-dist.mjs
```

## Data

`engine-rul export-dashboard` (`src/engine_rul/dashboard.py`) reads `results/final/` and
`results/phase2/`, validates them (required fields, finite numbers, known model identifiers,
complete engine coverage, Phase 2 manifests certifying `official_test_set_used: false`), and writes
small JSON files to `dashboard/public/data/` (git-ignored; CI regenerates them):

| File | Protocol | Source |
|---|---|---|
| `official_test.json` | `official_test` | `results/final/summary.json`, `test_predictions_median_seed.csv` |
| `phase1_validation.json` | `phase1_validation` | `results/final/val_dense_predictions_median_seed.csv` |
| `cv.json` | `nested_cv` | `results/phase2/cv/fd001_nested_cv/` |
| `ablation.json` | `nested_cv_ablation` | `results/phase2/ablation/fd001_ablation/` |
| `conformal.json` | `nested_cv_conformal` | `results/phase2/conformal/fd001_conformal/` |
| `benchmarks.json` | `benchmark` | `results/phase2/benchmark/fd001_latency/`, `results/final/benchmarks.json` |
| `meta.json` | `meta` | model labels, disclaimer |

Each file holds exactly one evaluation protocol. The frontend's loader (`src/data.ts`) checks the
`protocol` field against the chart that asked for it, so official-test and validation-only numbers
cannot be mixed by accident; a missing or mismatched file shows an error in that section only.

## Deployment

`.github/workflows/pages.yml` regenerates the data, type-checks, builds and checks the bundle on every
pull request, and deploys `dashboard/dist` to GitHub Pages on pushes to `main`. The site is served
from `/engine-rul/` (`vite.config.ts`; override with `DASHBOARD_BASE=/` for a root deployment).
One-time repository setting: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Design notes

Colours follow a colour-vision-deficiency-checked palette (learned models in blue / orange / aqua,
baselines in neutral greys), with separate steps for dark mode. Every chart has hover tooltips; tables
accompany the main charts. Dependencies are pinned exactly in `package.json` / `package-lock.json`.
