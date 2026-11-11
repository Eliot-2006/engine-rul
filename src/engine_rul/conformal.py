"""Phase 2: split-conformal prediction intervals for individual RUL predictions (validation only).

The conformal experiment is a nested CV run (``configs/phase2/conformal.yaml``) whose development
engines are split three ways: inner-train (fit + preprocessing), inner-val (hyperparameters, LSTM
early stopping) and calibration. Calibration engines never influence fitting or selection. This
module turns the saved calibration / held-out predictions into intervals and coverage tables.

Score: |y - y_hat| against the capped truth (the quantity the models are trained to predict).
Interval: [max(0, y_hat - q), y_hat + q], q = k-th smallest score, k = ceil((n + 1) * level).

Exchangeability (see docs/phase2.md):
* ``global_one_per_engine`` uses one randomly chosen endpoint per calibration engine. If engines
  are exchangeable and a new engine is observed at a cut point drawn the same way, the usual
  finite-sample guarantee P(y in interval) >= level holds for that new engine's endpoint.
* ``global_pooled`` / ``mondrian_pooled`` use all K endpoints per engine. Endpoints of one engine
  are dependent, so the scores are not exchangeable and there is **no** finite-sample guarantee;
  coverage is checked empirically only.
* No method's guarantee carries over to endpoints drawn differently (e.g. late-life cuts, or the
  official test set's truncation), which is why ``late`` endpoints are reported as a shift check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul.metrics import STAGE_BINS, stage_labels

LEVELS = (0.8, 0.9, 0.95)
METHODS = ("global_pooled", "global_one_per_engine", "mondrian_pooled")
PRED_BINS = (0.0, 50.0, 100.0, np.inf)  # Mondrian classes on *predicted* RUL (known at inference)
DEFAULT_METHOD = "global_one_per_engine"


def conformal_quantile(scores, level: float) -> float:
    s = np.sort(np.asarray(scores, dtype=np.float64))
    k = int(np.ceil((len(s) + 1) * level))
    return float(s[k - 1]) if k <= len(s) else float("inf")


def one_per_engine(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """One endpoint per engine, chosen by a per-engine seeded draw."""
    idx = [
        g.index[np.random.default_rng([seed, int(u)]).integers(len(g))]
        for u, g in df.groupby("unit")
    ]
    return df.loc[idx]


def calibrate(cal: pd.DataFrame, levels=LEVELS, seed: int = 0) -> dict:
    """``cal`` has columns unit, y_pred, y_target. Returns JSON-serialisable quantiles."""
    score = (cal["y_target"] - cal["y_pred"]).abs()
    one = one_per_engine(cal.assign(score=score), seed)["score"]
    cls = np.digitize(cal["y_pred"], PRED_BINS[1:-1])
    return {
        "score": "abs(capped_truth - prediction)",
        "levels": list(levels),
        "pred_bins": [b if np.isfinite(b) else None for b in PRED_BINS],
        "n_cal_engines": int(cal["unit"].nunique()),
        "n_cal_points": int(len(cal)),
        "n_cal_points_per_bin": [int((cls == c).sum()) for c in range(len(PRED_BINS) - 1)],
        "global_pooled": {str(lv): conformal_quantile(score, lv) for lv in levels},
        "global_one_per_engine": {str(lv): conformal_quantile(one, lv) for lv in levels},
        "mondrian_pooled": {
            str(lv): [conformal_quantile(score[cls == c], lv) for c in range(len(PRED_BINS) - 1)]
            for lv in levels
        },
    }


def interval(
    pred, cal: dict, level: float, method: str = DEFAULT_METHOD
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    q = cal[method][str(level)]
    if method == "mondrian_pooled":
        q = np.asarray(q, dtype=np.float64)[np.digitize(pred, PRED_BINS[1:-1])]
    return np.clip(pred - q, 0, None), pred + q


def analyze_conformal(exp: str | Path, levels=LEVELS) -> dict:
    exp = Path(exp)
    man = json.loads((exp / "manifest.json").read_text())
    cap = man["r_max_truth_cap"]
    primary = man["spec"]["endpoints"].get("primary", "uniform")
    p = pd.read_csv(exp / "predictions.csv.gz")
    p["y_target"] = np.minimum(p["y_true"], cap) if cap else p["y_true"]
    p["stage"] = pd.cut(p["y_true"], STAGE_BINS, right=False, labels=stage_labels())
    rows = []
    for (r, k, fam), g in p.groupby(["repeat", "fold", "family"], sort=True):
        cal = calibrate(g[(g.role == "cal") & (g.strategy == primary)], levels, seed=int(r))
        model_dir = exp / "models" / f"r{r}_f{k}_{fam}"
        if model_dir.exists():
            (model_dir / "conformal.json").write_text(json.dumps(cal, indent=1) + "\n")
        for strat, o in g[g.role == "outer"].groupby("strategy"):
            for m in METHODS:
                for lv in levels:
                    lo, hi = interval(o["y_pred"], cal, lv, m)
                    rows.append(
                        o[["unit", "stage", "y_true", "y_target"]].assign(
                            repeat=r,
                            fold=k,
                            family=fam,
                            strategy=strat,
                            method=m,
                            level=lv,
                            width=hi - lo,
                            covered=(o["y_target"] >= lo) & (o["y_target"] <= hi),
                            covered_uncapped=(o["y_true"] >= lo) & (o["y_true"] <= hi),
                        )
                    )
    cov = pd.concat(rows, ignore_index=True)
    keys = ["family", "strategy", "method", "level"]
    out: dict = {
        "experiment": man["name"],
        "truth": "capped",
        "truth_cap": cap,
        "calibration_strategy": primary,
        "levels": list(levels),
        "methods": list(METHODS),
        "summary": [],
        "per_stage": [],
    }
    for key, g in cov.groupby(keys, sort=True):
        per_eng = g.groupby(["fold", "unit"])[["covered", "covered_uncapped", "width"]].mean()
        per_fold = g.groupby("fold")["covered"].mean()
        out["summary"].append(
            {
                **dict(zip(keys, key, strict=True)),
                "coverage": float(per_eng["covered"].mean()),  # engine-weighted
                "coverage_points": float(g["covered"].mean()),
                "coverage_uncapped_truth": float(per_eng["covered_uncapped"].mean()),
                "coverage_fold_min": float(per_fold.min()),
                "coverage_fold_max": float(per_fold.max()),
                "mean_width": float(per_eng["width"].mean()),
                "median_width": float(g["width"].median()),
                "n_engines": int(len(per_eng)),
                "n_points": int(len(g)),
            }
        )
        for st, gs in g.groupby("stage", observed=True):
            out["per_stage"].append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "stage": str(st),
                    "coverage": float(gs["covered"].mean()),
                    "mean_width": float(gs["width"].mean()),
                    "n_points": int(len(gs)),
                }
            )
    (exp / "conformal_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
