"""Phase 3: static dashboard data and README presentation, generated from committed artifacts only.

``export_dashboard`` reads ``results/final`` (official test, Phase 1) and ``results/phase2``
(validation only) and writes compact JSON for the static dashboard (``dashboard/``). Each file
carries a ``protocol`` field and is built from exactly one protocol's artifacts, so official-test
and validation-only numbers can never end up in the same file; the dashboard refuses a file whose
protocol does not match the chart it feeds. No model, MLflow store or raw data file is read.

``update_readme`` refreshes the generated README blocks (results at a glance, Phase 1 table,
Phase 2 table) and the README hero figure from the same artifacts.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul.errors import DataValidationError
from engine_rul.plots import FAMILY_LABELS, FAMILY_ORDER
from engine_rul.utils import REPO_ROOT

SCHEMA_VERSION = 1
BASELINE = "life_prior"  # the cycle-only baseline the headline improvement is measured against
LEARNED = ("ridge", "hgb", "lstm")
PROTOCOLS = {
    "official_test.json": "official_test",
    "phase1_validation.json": "phase1_validation",
    "cv.json": "nested_cv",
    "ablation.json": "nested_cv_ablation",
    "conformal.json": "nested_cv_conformal",
    "benchmarks.json": "benchmark",
}
STRATEGY_LABELS = {
    "uniform": "Uniform cut points",
    "late": "Late-life cut points (RUL ≤ 125)",
    "stratified": "Stage-stratified cut points",
}
ABLATION_GROUPS = {
    "base": (
        "Full model",
        "Frozen-config inputs: sensors + normalized cycle, all window statistics",
    ),
    "sensors_only": ("Inputs", "Drops the normalized-cycle channel"),
    "cycle_only": ("Inputs", "Keeps only the normalized-cycle channel"),
    "stats_last": ("Window features", "Per-sensor last value only"),
    "stats_last_mean": ("Window features", "Last value + window mean"),
    "stats_last_mean_slope": ("Window features", "Last value + mean + slope (no std)"),
    "window_20": ("Sequence window", "20-cycle LSTM input window (default 30)"),
    "window_40": ("Sequence window", "40-cycle LSTM input window (default 30)"),
    "layers_1": ("Architecture", "LSTM depth fixed to 1 layer"),
    "layers_2": ("Architecture", "LSTM depth fixed to 2 layers"),
    "early_stop_dense": ("Training", "Early stopping on dense inner-validation windows"),
}
CONFORMAL_METHODS = {
    "global_one_per_engine": (
        "Global, one cut point per engine",
        "finite_sample",
        "Finite-sample marginal guarantee if engines are exchangeable and the new cut point is drawn "
        "the same way as in calibration.",
    ),
    "global_pooled": (
        "Global, all cut points pooled",
        "empirical",
        "Cut points of one engine are dependent: no finite-sample guarantee; coverage checked empirically.",
    ),
    "mondrian_pooled": (
        "Per predicted-RUL band (Mondrian)",
        "empirical",
        "Separate quantile per predicted-RUL band, pooled cut points: empirical coverage only.",
    ),
}


# ----------------------------------------------------------------------------- helpers


def _r(x, nd: int = 2):
    """Round once, to display precision (cycles: 2 decimals; fractions: 3), so the dashboard and
    README never re-round an already rounded number. Rejects NaN / inf (malformed artifacts)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        raise DataValidationError(f"non-finite number in artifact: {x}")
    return round(x, nd)


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise DataValidationError(f"missing artifact: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise DataValidationError(f"malformed JSON in {path}: {e}") from e


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"missing artifact: {path}")
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise DataValidationError(f"{path}: missing columns {sorted(missing)}")
    return df


def _families(ids) -> list[str]:
    ids = list(ids)
    unknown = set(ids) - set(FAMILY_ORDER)
    if unknown:
        raise DataValidationError(f"unknown model identifiers {sorted(unknown)}")
    return [f for f in FAMILY_ORDER if f in ids]


def _ms(d: dict) -> dict:
    return {"mean": _r(d["mean"]), "std": _r(d["std"])}


def _phase2_manifest(d: Path) -> dict:
    man = _read_json(d / "manifest.json")
    if man.get("official_test_set_used") is not False:
        raise DataValidationError(f"{d}: manifest does not certify official_test_set_used = false")
    prov = man.get("provenance", {})
    snap = prov.get("source_snapshot")
    return {
        "experiment": man["name"],
        "created_at_utc": man["created_at_utc"],
        "git_commit": prov.get("git_commit"),
        "git_dirty": prov.get("git_dirty"),
        "source_snapshot": snap["path"] if snap else None,
        "official_test_set_used": False,
    }


def family_meta() -> list[dict]:
    kinds = {"constant": "baseline", "life_prior": "baseline"}
    return [
        {"id": f, "label": FAMILY_LABELS[f], "kind": kinds.get(f, "model")} for f in FAMILY_ORDER
    ]


# ----------------------------------------------------------------------------- official test


def official_test(results: Path) -> dict:
    s = _read_json(results / "final" / "summary.json")
    fams = _families(s["families"])
    preds = _read_csv(
        results / "final" / "test_predictions_median_seed.csv",
        ["family", "unit", "cycle", "y_true", "y_pred"],
    )
    rows = []
    for f in fams:
        v = s["families"][f]
        tp, ci = v["test_primary"], v["test_ci95_median_seed"]
        identical = len(v["seeds"]) > 1 and tp["rmse"]["std"] == 0
        rows.append(
            {
                "id": f,
                "seeds": len(v["seeds"]),
                "seeds_identical": identical,
                "rmse": _ms(tp["rmse"]),
                "rmse_ci95": [_r(x) for x in ci["rmse"]],
                "mae": _ms(tp["mae"]),
                "mae_ci95": [_r(x) for x in ci["mae"]],
                "nasa_score": _ms(tp["nasa_sum"]),
                "bias": _ms(tp["bias"]),
                "late_fraction": _r(tp["late_fraction"]["mean"], 3),
                "rmse_capped_truth": _r(v["test_capped"]["rmse"]["mean"]),
            }
        )
    by = {r["id"]: r for r in rows}
    best = min(fams, key=lambda f: by[f]["rmse"]["mean"])
    engines = []
    for unit, g in preds.groupby("unit", sort=True):
        if set(g["family"]) != set(fams) or g["y_true"].nunique() != 1 or g["cycle"].nunique() != 1:
            raise DataValidationError(
                f"test predictions for engine {unit} are incomplete or inconsistent"
            )
        engines.append(
            {
                "unit": int(unit),
                "last_cycle": int(g["cycle"].iloc[0]),
                "y_true": _r(g["y_true"].iloc[0], 1),
                "pred": {r.family: _r(r.y_pred, 2) for r in g.itertuples()},
            }
        )
    if len(engines) != s["n_test_engines"]:
        raise DataValidationError(
            f"{len(engines)} engines with predictions, summary says {s['n_test_engines']}"
        )
    base = by[BASELINE]["rmse"]["mean"]
    raw = lambda f: s["families"][f]["test_primary"]["rmse"]["mean"]  # noqa: E731
    return {
        "protocol": "official_test",
        "title": "Official FD001 test set (Phase 1, evaluated once with frozen configs)",
        "dataset": "NASA C-MAPSS FD001",
        "split": f"Official test set: {s['n_test_engines']} engines truncated before failure",
        "target": "Official RUL label (uncapped) at each engine's last observed cycle",
        "unit": "cycles",
        "n_engines": s["n_test_engines"],
        "n_true_rul_above_cap": s["n_test_rul_above_r_max"],
        "training_target_cap": s["r_max"],
        "ci_method": "2000-resample engine bootstrap of the median-seed model",
        "evaluated_at": s["generated_at"],
        "git_commit": s["git_commit"],
        "sources": [
            "results/final/summary.json",
            "results/final/test_predictions_median_seed.csv",
        ],
        "headline": {
            "best_family": best,
            "best_rmse": by[best]["rmse"]["mean"],
            "baseline_family": BASELINE,
            "baseline_rmse": base,
            "rmse_reduction_pct": _r(100 * (1 - raw(best) / raw(BASELINE)), 1),  # unrounded inputs
        },
        "families": rows,
        "paired": [
            {
                "a": p["a"],
                "b": p["b"],
                "delta_rmse": _r(p["delta_rmse"]),
                "ci95": [_r(x) for x in p["ci95"]],
                "ci_excludes_0": bool(p["ci_excludes_0"]),
            }
            for p in s["paired_comparisons"]
        ],
        "stages": {
            f: [
                {"bin": b["bin"], "n": b["n"], "rmse": _r(b["rmse"]), "bias": _r(b["bias"])}
                for b in s["stage_metrics_test"][f]
            ]
            for f in fams
        },
        "engines": engines,
    }


def phase1_validation(results: Path) -> dict:
    s = _read_json(results / "final" / "summary.json")
    fams = _families(s["families"])
    dense = _read_csv(
        results / "final" / "val_dense_predictions_median_seed.csv",
        ["unit", "cycle", "y_true", "y_capped", "y_pred", "family"],
    )
    traj = []
    for unit, g in dense.groupby("unit", sort=True):
        w = g.pivot(index="cycle", columns="family", values="y_pred").sort_index()
        if set(w.columns) != set(fams) or w.isna().any().any():
            raise DataValidationError(f"validation trajectories for engine {unit} are incomplete")
        truth = g.drop_duplicates("cycle").set_index("cycle").loc[w.index, "y_true"]
        traj.append(
            {
                "unit": int(unit),
                "failure_cycle": int(w.index.max() + truth.loc[w.index.max()]),
                "cycle": [int(c) for c in w.index],
                "y_true": [_r(x, 0) for x in truth],
                "pred": {f: [_r(x, 1) for x in w[f]] for f in fams},
            }
        )
    return {
        "protocol": "phase1_validation",
        "title": "Phase 1 validation engines (engine-level hold-out used for model selection)",
        "dataset": "NASA C-MAPSS FD001 training file",
        "split": f"{len(traj)} validation engines from splits/fd001_split.json, never used for fitting",
        "target": "RUL = failure cycle − current cycle; models are trained on RUL capped at "
        f"{s['r_max']}",
        "unit": "cycles",
        "sources": [
            "results/final/summary.json",
            "results/final/val_dense_predictions_median_seed.csv",
        ],
        "selection_rmse": {
            f: _r(s["families"][f]["val_points_capped"]["rmse"]["mean"]) for f in fams
        },
        "selection_note": "Selection RMSE: 200 test-like truncation points (10 per engine), capped truth.",
        "trajectories": traj,
    }


# ----------------------------------------------------------------------------- phase 2


def cross_validation(p2: Path, name: str = "fd001_nested_cv") -> dict:
    d = p2 / "cv" / name
    cv, prov = _read_json(d / "summary.json"), _phase2_manifest(d)
    man = _read_json(d / "manifest.json")
    fm = _read_csv(
        d / "fold_metrics.csv",
        ["strategy", "family", "variant", "repeat", "fold", "truth", "rmse", "mae", "nasa_mean"],
    )
    fm = fm[(fm.truth == "capped") & (fm.variant == "base")]
    strategies = []
    for sid, st in cv["strategies"].items():
        fams = _families(st["families"])
        g = fm[fm.strategy == sid]
        strategies.append(
            {
                "id": sid,
                "label": STRATEGY_LABELS.get(sid, sid),
                "n_points": st["n_points"],
                "families": [
                    {
                        "id": f,
                        "pooled_rmse": _r(st["families"][f]["pooled"]["capped"]["rmse"]),
                        "pooled_rmse_ci95": [
                            _r(x) for x in st["families"][f]["rmse_ci95_engine_bootstrap"]["capped"]
                        ],
                        "pooled_mae": _r(st["families"][f]["pooled"]["capped"]["mae"]),
                        "fold_rmse_mean": _r(st["families"][f]["fold_rmse_capped"]["mean"]),
                        "fold_rmse_std": _r(st["families"][f]["fold_rmse_capped"]["std"]),
                        "mean_rank": _r(st["ranking"][f]["mean_rank"], 3),
                        "rank1_frac": _r(st["ranking"][f]["rank1_frac"], 3),
                        "bootstrap_rank1_frac": _r(st["ranking"][f]["bootstrap_rank1_frac"], 3),
                    }
                    for f in fams
                ],
                "paired": [
                    {
                        "a": p["a"],
                        "b": p["b"],
                        "delta_rmse": _r(p["delta_rmse"]),
                        "ci95": [_r(x) for x in p["ci95"]],
                        "ci_excludes_0": bool(p["ci_excludes_0"]),
                    }
                    for p in st["paired"]
                    if p["truth"] == "capped"
                ],
                "folds": [
                    {
                        "family": r.family,
                        "repeat": int(r.repeat),
                        "fold": int(r.fold),
                        "rmse": _r(r.rmse),
                        "mae": _r(r.mae),
                    }
                    for r in g.sort_values(["family", "repeat", "fold"]).itertuples()
                ],
            }
        )
    n_folds = {len(s["folds"]) // len(s["families"]) for s in strategies}
    if n_folds != {cv["n_outer_folds"]}:
        raise DataValidationError(f"{d}: fold metrics do not cover every outer fold")
    ep = man["spec"]["endpoints"]
    return {
        "protocol": "nested_cv",
        "title": "Repeated nested cross-validation (Phase 2, training engines only)",
        "dataset": "NASA C-MAPSS FD001 training file",
        "split": f"{cv['n_repeats']} repeats × {cv['n_outer_folds'] // cv['n_repeats']}-fold engine-grouped outer CV over "
        f"{cv['n_engines']} training engines; hyperparameters selected in inner folds",
        "target": f"RUL capped at {cv['truth_cap']} (the quantity the models are trained to predict)",
        "unit": "cycles",
        "endpoints": f"{ep['n_per_engine']} truncation cut points per held-out engine (cycle ≥ "
        f"{ep['min_cycle']}), identical for every model",
        "primary_strategy": cv["primary_strategy"],
        "n_outer_folds": cv["n_outer_folds"],
        "ci_method": "engine cluster bootstrap over pooled predictions (approximate: ignores "
        "overlapping training sets)",
        "provenance": prov,
        "sources": [
            f"results/phase2/cv/{name}/summary.json",
            f"results/phase2/cv/{name}/fold_metrics.csv",
        ],
        "strategies": strategies,
        "rank_agreement": [
            {"a": x["a"], "b": x["b"], "kendall_tau": _r(x["kendall_tau"], 3)}
            for x in cv["strategy_rank_agreement"]
        ],
        "selection_agreement_with_primary": {
            s_: {f: _r(v, 3) for f, v in d_.items()}
            for s_, d_ in cv["selection_agreement_with_primary"].items()
        },
    }


def ablation(p2: Path, name: str = "fd001_ablation") -> dict:
    d = p2 / "ablation" / name
    ab, prov = _read_json(d / "summary.json"), _phase2_manifest(d)
    fams = _families(ab["families"])
    out = []
    for f in fams:
        rows = ab["families"][f]
        if not rows or rows[0]["variant"] != "base":
            raise DataValidationError(f"{d}: {f} ablation has no base row")
        for r in rows:
            if r["variant"] not in ABLATION_GROUPS:
                raise DataValidationError(f"{d}: unknown ablation variant {r['variant']!r}")
            group, desc = ABLATION_GROUPS[r["variant"]]
            out.append(
                {
                    "family": f,
                    "variant": r["variant"],
                    "group": group,
                    "description": desc,
                    "pooled_rmse": _r(r["rmse"]),
                    "delta_rmse": _r(r["delta_rmse"]),
                    "ci95": [_r(x) for x in r["ci95"]],
                    "ci_excludes_0": bool(r["ci_excludes_0"]),
                    "repeats": len(r["repeats"]),
                    "n_folds": r["n_folds"],
                }
            )
    return {
        "protocol": "nested_cv_ablation",
        "title": "One-factor ablations (Phase 2, training engines only)",
        "dataset": "NASA C-MAPSS FD001 training file",
        "split": "Same engine-grouped outer folds as the nested CV; each fold's selected "
        "hyperparameters are reused for every variant",
        "target": f"RUL capped at {ab['truth_cap']}",
        "unit": "cycles",
        "delta": "Δ pooled RMSE vs the same family's full model on identical cut points; "
        "95% paired engine-bootstrap CI",
        "provenance": prov,
        "sources": [f"results/phase2/ablation/{name}/summary.json"],
        "determinism_check_max_abs_diff": {
            k: _r(v, 6) for k, v in ab["base_vs_source_cv_max_abs_diff"].items()
        },
        "rows": out,
    }


def conformal(p2: Path, name: str = "fd001_conformal") -> dict:
    d = p2 / "conformal" / name
    c, prov = _read_json(d / "conformal_summary.json"), _phase2_manifest(d)
    roles = _read_json(d / "manifest.json")["spec"].get("inner_fractions", {})
    keep = lambda r: {  # noqa: E731
        k: (_r(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "median_width"
    }
    for r in c["summary"]:
        if r["method"] not in CONFORMAL_METHODS:
            raise DataValidationError(f"{d}: unknown conformal method {r['method']!r}")
    _families({r["family"] for r in c["summary"]})
    return {
        "protocol": "nested_cv_conformal",
        "title": "Split-conformal prediction intervals (Phase 2, training engines only)",
        "dataset": "NASA C-MAPSS FD001 training file",
        "split": "Inside each outer fold the development engines are split into inner-train / "
        "inner-validation / calibration engines; calibration engines are never used for fitting or "
        "selection, and coverage is measured on the held-out outer-fold engines",
        "inner_fractions": roles,
        "target": f"RUL capped at {c['truth_cap']}",
        "unit": "cycles",
        "calibration_strategy": c["calibration_strategy"],
        "levels": c["levels"],
        "methods": [
            {"id": m, "label": lab, "guarantee": g, "note": note}
            for m, (lab, g, note) in CONFORMAL_METHODS.items()
        ],
        "caveat": "Coverage is marginal (averaged over engines and cut points), rests on engine "
        "exchangeability and on how cut points are drawn, and says nothing about real aircraft.",
        "provenance": prov,
        "sources": [f"results/phase2/conformal/{name}/conformal_summary.json"],
        "summary": [keep(r) for r in c["summary"]],
        "per_stage": [
            keep(r) for r in c["per_stage"] if r["strategy"] == c["calibration_strategy"]
        ],
    }


def benchmarks(results: Path, p2: Path, name: str = "fd001_latency") -> dict:
    d = p2 / "benchmark" / name
    b, prov = _read_json(d / "benchmarks.json"), _phase2_manifest(d)
    b1 = _read_json(results / "final" / "benchmarks.json")
    _families({r["family"] for r in b["rows"]})
    return {
        "protocol": "benchmark",
        "title": "Inference latency and memory (Phase 2 benchmark; Phase 1 training cost)",
        "environment": b["environment"],
        "scope_definitions": b["scope_definitions"],
        "method": "5 warm-up calls, 20–50 timed repetitions, device synchronisation after every "
        "call; inputs are training-file sensor rows",
        "memory_note": b["memory_note"],
        "process_max_rss_mb": _r(b["process_max_rss_mb"], 1),
        "provenance": prov,
        "sources": [
            f"results/phase2/benchmark/{name}/benchmarks.json",
            "results/final/benchmarks.json",
        ],
        "rows": [
            {
                "family": r["family"],
                "device": r["device"],
                "scope": r["scope"],
                "unit": r["unit"],
                "n": r["n"],
                "latency_ms_median": _r(r["latency_ms_median"], 4),
                "latency_ms_p95": _r(r["latency_ms_p95"], 4),
                "throughput_per_s": _r(r["throughput_per_s"], 1),
                "python_peak_alloc_mb": _r(r["python_peak_alloc_mb"], 3),
            }
            for r in b["rows"]
        ],
        "training": [
            {
                "family": r["family"],
                "device": r["train_device"],
                "train_time_s": _r(r["train_time_s"], 2),
                "epochs": r["epochs_run"],
                "n_params": r["n_params"],
                "artifact_bytes": r["artifact_bytes"],
            }
            for r in b1["rows"]
            if r["device"] == "cpu"
        ],
    }


# ----------------------------------------------------------------------------- export


def build_payloads(results: str | Path = "results") -> dict[str, dict]:
    """All dashboard files as dicts. Raises DataValidationError on missing or malformed input."""
    results = Path(results)
    p2 = results / "phase2"
    try:
        files = {
            "official_test.json": official_test(results),
            "phase1_validation.json": phase1_validation(results),
            "cv.json": cross_validation(p2),
            "ablation.json": ablation(p2),
            "conformal.json": conformal(p2),
            "benchmarks.json": benchmarks(results, p2),
        }
    except (KeyError, TypeError, ValueError) as e:
        raise DataValidationError(
            f"artifact is missing a required field or is malformed: {e!r}"
        ) from e
    for name, payload in files.items():
        if payload["protocol"] != PROTOCOLS[name]:
            raise DataValidationError(
                f"{name}: protocol {payload['protocol']!r} != {PROTOCOLS[name]!r}"
            )
        payload["schema_version"] = SCHEMA_VERSION
    return files


def export_dashboard(
    results: str | Path = "results", out: str | Path = "dashboard/public/data"
) -> list[Path]:
    files = build_payloads(results)
    s = files["official_test.json"]
    files["meta.json"] = {
        "protocol": "meta",
        "schema_version": SCHEMA_VERSION,
        "project": "EngineRUL",
        "repository": "https://github.com/Eliot-2006/engine-rul",
        "families": family_meta(),
        "protocols": {name[:-5]: p["title"] for name, p in files.items()},
        "official_test_evaluated_at": s["evaluated_at"],
        "disclaimer": "Simulated NASA C-MAPSS data. Not validated for real aircraft maintenance or "
        "flight-safety decisions.",
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in files.items():
        p = out / name
        p.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
        written.append(p)
    return written


# ----------------------------------------------------------------------------- README


def glance_block(official: dict) -> str:
    h = official["headline"]
    best = FAMILY_LABELS[h["best_family"]].lower()
    return "\n".join(
        [
            "| Best official test RMSE | vs. cycle-only baseline | Evaluation |",
            "|:---:|:---:|:---:|",
            f"| **{h['best_rmse']:.2f} cycles**<br><sub>{best}</sub> "
            f"| **−{h['rmse_reduction_pct']:.1f}% RMSE**<br><sub>{h['baseline_rmse']:.2f} → "
            f"{h['best_rmse']:.2f} cycles</sub> "
            f"| **{official['n_engines']} test engines**<br><sub>NASA C-MAPSS FD001</sub> |",
            "",
            "<sub>Official FD001 test split, one prediction per engine at its last observed cycle, scored "
            "against the official (uncapped) RUL labels; evaluated once with frozen configurations. RMSE "
            "is an error in cycles (lower is better), not an accuracy. Baseline: the *life prior*, which "
            "predicts from the current cycle only. Generated from `results/final/summary.json` by "
            "`engine-rul readme`.</sub>",
        ]
    )


def hero_figure(official: dict, path: str | Path) -> Path:
    """Two panels: predicted vs true RUL for the best model, and test RMSE (95% CI) for every model."""
    from engine_rul import plots as P
    from engine_rul.report_phase2 import COLORS

    P.style()
    plt = P.plt
    best = official["headline"]["best_family"]
    eng = official["engines"]
    t = np.array([e["y_true"] for e in eng])
    y = np.array([e["pred"][best] for e in eng])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    lim = 160
    a.plot([0, lim], [0, lim], color="#898781", lw=1.2, ls="--", zorder=1)
    a.scatter(t, y, s=22, color=COLORS[best], edgecolor="white", lw=0.6, zorder=2)
    a.set(
        xlim=(0, lim),
        ylim=(0, lim),
        xlabel="True remaining useful life (cycles)",
        ylabel="Predicted RUL (cycles)",
        title=f"{FAMILY_LABELS[best]}: {len(eng)} official test engines",
    )
    a.set_aspect("equal")
    a.text(
        6,
        lim - 8,
        "each dot: one engine\ndashed: perfect prediction",
        va="top",
        fontsize=9,
        color="#52514e",
    )
    fams = sorted(official["families"], key=lambda r: -r["rmse"]["mean"])
    for i, r in enumerate(fams):
        m, (lo, hi) = r["rmse"]["mean"], r["rmse_ci95"]
        b.barh(i, m, height=0.6, color=COLORS[r["id"]])
        b.errorbar(m, i, xerr=[[m - lo], [hi - m]], color="#0b0b0b", lw=1.2, capsize=3)
        b.text(hi + 1, i, f"{m:.2f}", va="center", fontsize=10, color="#0b0b0b")
    b.set_yticks(range(len(fams)), [FAMILY_LABELS[r["id"]] for r in fams])
    b.set(
        xlabel="Test RMSE (cycles, lower is better) with 95% bootstrap CI",
        title="Every model on the same 100 engines",
        xlim=(0, max(r["rmse_ci95"][1] for r in fams) * 1.15),
    )
    b.grid(axis="y", visible=False)
    b.set_axisbelow(True)
    fig.suptitle(
        "EngineRUL: remaining-useful-life predictions on NASA C-MAPSS FD001 (official test set)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, metadata={"Software": None})
    plt.close(fig)
    return path


SOCIAL_SIZE = (1200, 630)  # Open Graph / Twitter large-card size, pixels


def social_card(official: dict, path: str | Path) -> Path:
    """The dashboard's link-preview image: project, dataset and the headline official-test result."""
    from engine_rul import plots as P
    from engine_rul.report_phase2 import COLORS

    P.style()
    plt = P.plt
    h = official["headline"]
    best = h["best_family"]
    ink, ink2, page = "#0b0b0b", "#52514e", "#f9f9f7"
    fig = plt.figure(figsize=(SOCIAL_SIZE[0] / 100, SOCIAL_SIZE[1] / 100), dpi=100, facecolor=page)
    text = lambda x, y, s, **kw: fig.text(x, y, s, color=kw.pop("color", ink), **kw)  # noqa: E731
    text(0.055, 0.83, "EngineRUL", fontsize=40, fontweight="bold")
    text(
        0.055,
        0.745,
        "Remaining-useful-life prediction for turbofan engines",
        fontsize=17,
        color=ink2,
    )
    text(0.055, 0.53, f"{h['best_rmse']:.2f} cycles", fontsize=58, fontweight="bold")
    text(
        0.055,
        0.455,
        f"best official test RMSE ({FAMILY_LABELS[best].lower()})",
        fontsize=17,
        color=ink2,
    )
    text(
        0.055,
        0.33,
        f"\u2212{h['rmse_reduction_pct']:.1f}% RMSE vs the cycle-only baseline\n"
        f"{official['n_engines']} official test engines, evaluated once",
        fontsize=17,
        linespacing=1.5,
    )
    text(
        0.055,
        0.075,
        f"NASA C-MAPSS {official['dataset'].split()[-1]} \u00b7 simulated data, not validated for real aircraft",
        fontsize=13,
        color=ink2,
    )
    a = fig.add_axes((0.665, 0.14, 0.295, 0.295 * SOCIAL_SIZE[0] / SOCIAL_SIZE[1]), facecolor=page)
    t = np.array([e["y_true"] for e in official["engines"]])
    y = np.array([e["pred"][best] for e in official["engines"]])
    lim = 160
    a.plot([0, lim], [0, lim], color="#898781", lw=1.2, ls="--", zorder=1)
    a.scatter(t, y, s=26, color=COLORS[best], edgecolor=page, lw=0.6, zorder=2)
    a.set(xlim=(0, lim), ylim=(0, lim), xlabel="True RUL (cycles)", ylabel="Predicted RUL (cycles)")
    a.set_title("Each dot: one test engine", fontsize=13, loc="left")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"savefig.bbox": "standard"}):  # exact pixel size, no tight cropping
        fig.savefig(path, dpi=100, facecolor=page, metadata={"Software": None})
    plt.close(fig)
    return path


def _replace_block(text: str, name: str, block: str) -> str:
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    if start not in text or end not in text:
        raise DataValidationError(f"README has no {start} … {end} markers")
    return re.sub(
        re.escape(start) + r".*?" + re.escape(end),
        lambda _: f"{start}\n{block}\n{end}",
        text,
        flags=re.S,
    )


def readme_blocks(results: str | Path = "results") -> dict[str, str]:
    from engine_rul import report as R
    from engine_rul import report_phase2 as R2

    results = Path(results)
    p2 = results / "phase2"
    cv = _read_json(p2 / "cv" / "fd001_nested_cv" / "summary.json")
    ab = _read_json(p2 / "ablation" / "fd001_ablation" / "summary.json")
    conf = _read_json(p2 / "conformal" / "fd001_conformal" / "conformal_summary.json")
    return {
        "GLANCE": glance_block(official_test(results)),
        "RESULTS": R.readme_block(R.load_results(results), {}),
        "PHASE2": R2.readme_block(cv, ab, conf),
    }


def update_readme(
    results: str | Path = "results",
    readme: str | Path = REPO_ROOT / "README.md",
    hero: str | Path = REPO_ROOT / "reports" / "figures" / "readme" / "hero.png",
    social: str | Path = REPO_ROOT / "dashboard" / "public" / "og.png",
) -> Path:
    readme = Path(readme)
    text = readme.read_text()
    for name, block in readme_blocks(results).items():
        text = _replace_block(text, name, block)
    readme.write_text(text)
    official = official_test(Path(results))
    hero_figure(official, hero)
    social_card(official, social)
    return readme
