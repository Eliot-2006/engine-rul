"""Exploratory data analysis (PRD 2.8, EDA-01..08). Uses training data; test *sensor* data only
for trajectory lengths. Test labels are never read here."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul import data as D
from engine_rul import plots as P
from engine_rul.split import load_split

log = logging.getLogger(__name__)


def column_quality(df: pd.DataFrame) -> dict:
    out = {}
    for c in D.COLUMNS:
        v = df[c].to_numpy(np.float64)
        out[c] = {
            "count": int(len(v)),
            "missing": int(np.isnan(v).sum()),
            "non_finite": int((~np.isfinite(v)).sum()),
            "min": float(np.nanmin(v)),
            "max": float(np.nanmax(v)),
            "mean": float(np.nanmean(v)),
            "std": float(np.nanstd(v)),
            "nunique": int(pd.Series(v).nunique()),
        }
    return out


def run_eda(
    data_dir="data/raw", subset="FD001", split_file="splits/fd001_split.json", out="reports"
) -> Path:
    P.style()
    out = Path(out)
    fig_dir = out / "figures" / "eda"
    train = D.load_train(data_dir, subset)
    test = D.load_test_features(data_dir, subset)
    split = load_split(split_file)
    tr = train[train["unit"].isin(split.train_units)]
    quality = column_quality(train)
    constant = [c for c in D.COLUMNS[2:] if quality[c]["nunique"] == 1]
    binary = [c for c in D.COLUMNS[2:] if quality[c]["nunique"] == 2]
    informative = [c for c in D.SENSORS if c not in constant + binary]
    life = train.groupby("unit")["cycle"].max()
    tlen = test.groupby("unit")["cycle"].max()
    rho = tr[informative + ["rul"]].corr(method="spearman")["rul"].drop("rul")
    summary = {
        "subset": subset,
        "train": {
            "rows": len(train),
            "units": int(train["unit"].nunique()),
            "lifetime": {
                "min": int(life.min()),
                "median": float(life.median()),
                "mean": float(life.mean()),
                "max": int(life.max()),
            },
        },
        "test_sensor_data": {
            "rows": len(test),
            "units": int(test["unit"].nunique()),
            "observed_length": {
                "min": int(tlen.min()),
                "median": float(tlen.median()),
                "max": int(tlen.max()),
            },
        },
        "constant_columns": constant,
        "two_valued_columns": binary,
        "near_constant_settings": [c for c in ("op1", "op2") if quality[c]["std"] < 0.01],
        "informative_sensors": informative,
        "spearman_with_rul_train_split": {k: float(v) for k, v in rho.items()},
        "column_quality": quality,
        "note": "Column quality and lifetimes use all 100 training engines; correlations use train-split engines only.",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    P.lifetime_hist(life.to_numpy(), tlen.to_numpy(), fig_dir / "lifetime_hist.png")
    P.sensor_trajectories(tr, informative, fig_dir / "sensor_trajectories.png")
    P.sensor_distributions(tr, informative, fig_dir / "sensor_distributions.png")
    P.correlation_heatmap(tr, informative + ["rul"], fig_dir / "correlation_heatmap.png")
    P.sensor_rul_correlation(rho, fig_dir / "sensor_rul_correlation.png")
    example = [
        int(u)
        for u in tr.groupby("unit")["cycle"]
        .max()
        .sort_values()
        .index[[0, len(split.train_units) // 2, -1]]
    ]
    P.rul_target_shapes(tr, example, 125, fig_dir / "rul_target_shapes.png")
    lines = [
        "# Dropped columns (EDA-07)",
        "",
        "Measured on all 100 FD001 training engines. Re-derived on train-split rows only at fit time (PREP-01).",
        "",
        "| Column | Distinct values | Std | Reason |",
        "|---|---|---|---|",
    ]
    for c in constant + binary:
        reason = "constant" if c in constant else "only two distinct values (quantization noise)"
        lines.append(
            f"| `{c}` ({D.SENSOR_DESCRIPTIONS.get(c, 'operational setting')}) | {quality[c]['nunique']} | "
            f"{quality[c]['std']:.2e} | {reason} |"
        )
    for c in summary["near_constant_settings"]:
        lines.append(
            f"| `{c}` (operational setting) | {quality[c]['nunique']} | {quality[c]['std']:.2e} | "
            "noise around FD001's single operating condition; dropped by config (ADR-02) |"
        )
    lines += ["", f"Kept: {len(informative)} sensors: {', '.join(informative)}."]
    (fig_dir / "constant_sensors.md").write_text("\n".join(lines) + "\n")
    log.info("EDA written to %s", out)
    return out
