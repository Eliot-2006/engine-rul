"""Validation re-evaluation and the locked, logged final test evaluation (EVAL-02..33).

The official test labels are read only inside ``evaluate_test_final`` (EVAL-20/21)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul import data as D
from engine_rul.config import from_dict, load_config
from engine_rul.errors import TestSetLockError
from engine_rul.features import build_windows
from engine_rul.metrics import (
    METRICS,
    all_metrics,
    bootstrap_ci,
    mae,
    nasa_score_mean,
    paired_bootstrap_diff,
    rmse,
    stage_metrics,
)
from engine_rul.models import load_model
from engine_rul.preprocess import Preprocessor
from engine_rul.train import FINAL_DIR, prepare, val_metrics
from engine_rul.utils import git_info, sha256_file

log = logging.getLogger(__name__)

LEARNED = ("ridge", "hgb", "lstm")
CLASSICAL = ("ridge", "hgb")


def evaluate_val(run_dir: str | Path) -> dict:
    """Independently recompute validation metrics from a saved model + preprocessor."""
    run_dir = Path(run_dir)
    rec = json.loads((run_dir / "run.json").read_text())
    cfg = from_dict(rec["config"])
    prep = prepare(cfg)
    saved = Preprocessor.from_dict(json.loads((run_dir / "preprocessor.json").read_text()))
    if saved.to_dict() != prep.pre.to_dict():
        raise RuntimeError("re-fitted preprocessor differs from the saved one")
    model = load_model(run_dir)
    pred = np.clip(model.predict(prep.val_points), 0, None)
    return {f"val_points_{k}": v for k, v in val_metrics(cfg, prep.val_points.y_true, pred).items()}


def require_final(split: str, final: bool) -> None:
    if split == "test" and not final:
        raise TestSetLockError(
            "the official test set is locked: evaluate it only with frozen final models "
            "via `engine-rul evaluate --split test --final` (EVAL-21)"
        )


def _check_final_run(rec: dict, final_dir: Path) -> None:
    """EVAL-22: only stage=final runs of the committed configs/final/<family>.yaml."""
    if rec.get("stage") != "final" or rec.get("status") != "finished":
        raise TestSetLockError(f"run {rec['run_id']} is not a finished stage=final run")
    frozen = load_config(final_dir / f"{rec['family']}.yaml")
    if frozen.hash() != rec["config_hash"]:
        raise TestSetLockError(
            f"run {rec['run_id']} config differs from {final_dir}/{rec['family']}.yaml"
        )


def evaluate_test_final(
    families: list[str],
    results_dir: str | Path = "results",
    final_dir: str | Path = FINAL_DIR,
    note: str = "",
    log_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    results_dir = Path(results_dir)
    from engine_rul.provenance import refuse_overwrite

    refuse_overwrite([results_dir / "final" / "summary.json"], force)
    runs = json.loads((results_dir / "final_runs.json").read_text())
    log_path = Path(log_path or results_dir / "test_evaluations.jsonl")
    git = git_info()
    per_run = {}
    test_cache: dict = {}
    for family in families:
        for run_id in runs[family]:
            run_dir = results_dir / run_id
            rec = json.loads((run_dir / "run.json").read_text())
            _check_final_run(rec, Path(final_dir))
            cfg = from_dict(rec["config"])
            dc = cfg.data
            key = (dc.data_dir, dc.subset)
            if key not in test_cache:
                test_df = D.load_test_features(dc.data_dir, dc.subset)
                y_test = D.load_test_rul(dc.data_dir, dc.subset).astype(
                    np.float64
                )  # EVAL-20: only here
                inv = D.load_manifest()["subsets"].get(dc.subset, {}).get("invariants")
                if dc.check_invariants and inv:
                    D.check_test_invariants(test_df, y_test, inv)
                test_cache[key] = (
                    test_df,
                    y_test,
                    {
                        f"test_{dc.subset}.txt": sha256_file(
                            D.subset_path(dc.data_dir, dc.subset, "test")
                        ),
                        f"RUL_{dc.subset}.txt": sha256_file(
                            D.subset_path(dc.data_dir, dc.subset, "RUL")
                        ),
                    },
                )
            test_df, y_test, test_sha = test_cache[key]
            pre = Preprocessor.from_dict(json.loads((run_dir / "preprocessor.json").read_text()))
            ws = build_windows(
                pre.transform(test_df), pre.feature_names, dc, end_points=_last_points(test_df)
            )
            y = y_test[ws.unit - 1]  # label i <-> unit i+1 (DATA-21)
            pred = np.clip(load_model(run_dir).predict(ws), 0, None)
            capped = np.minimum(y, dc.r_max) if dc.r_max else y
            m = {"primary": all_metrics(y, pred), "capped": all_metrics(capped, pred)}
            pd.DataFrame(
                {
                    "unit": ws.unit,
                    "cycle": ws.cycle,
                    "y_true": y,
                    "y_capped": capped,
                    "y_pred": pred,
                }
            ).to_csv(run_dir / "predictions_test.csv", index=False)
            entry = {
                "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
                "run_id": run_id,
                "family": family,
                "seed": rec["seed"],
                "config_hash": rec["config_hash"],
                "data_sha256": {f"train_{dc.subset}.txt": rec["data_sha256"], **test_sha},
                "metrics": m,
                "note": note,
            }
            with open(log_path, "a") as f:  # EVAL-23: append-only
                f.write(json.dumps(entry) + "\n")
            per_run[run_id] = {
                "family": family,
                "seed": rec["seed"],
                "rec": rec,
                "metrics": m,
                "pred": pred,
                "y": y,
                "unit": ws.unit,
                "cycle": ws.cycle,
            }
            log.info(
                "TEST %s seed %d: RMSE %.3f MAE %.3f NASA %.1f",
                family,
                rec["seed"],
                m["primary"]["rmse"],
                m["primary"]["mae"],
                m["primary"]["nasa_sum"],
            )
    return summarize(per_run, families, results_dir, test_df)


def _last_points(df: pd.DataFrame) -> np.ndarray:
    last = df.groupby("unit")["cycle"].max()
    return np.column_stack([last.index.to_numpy(), last.to_numpy()])


def _agg(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "per_seed": v.tolist(),
    }


def summarize(per_run: dict, families: list[str], results_dir: Path, test_df: pd.DataFrame) -> Path:
    out_dir = results_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    fam_runs = {
        f: sorted((r for r in per_run.values() if r["family"] == f), key=lambda r: r["seed"])
        for f in families
    }
    summary_fams, median_run = {}, {}
    for f, rs in fam_runs.items():
        rmses = [r["metrics"]["primary"]["rmse"] for r in rs]
        med = rs[int(np.argsort(rmses)[len(rmses) // 2])]  # median-RMSE seed (display only)
        median_run[f] = med
        ci = {
            name: bootstrap_ci(med["y"], med["pred"], fn, n_boot=2000, seed=0)
            for name, fn in (("rmse", rmse), ("mae", mae), ("nasa_mean", nasa_score_mean))
        }
        val = [r["rec"]["metrics"]["val_points_capped"] for r in rs]
        summary_fams[f] = {
            "config_hash": rs[0]["rec"]["config_hash"],
            "params": rs[0]["rec"]["config"]["model"]["params"],
            "seeds": [r["seed"] for r in rs],
            "run_ids": [r["rec"]["run_id"] for r in rs],
            "median_seed": med["seed"],
            "test_primary": {k: _agg([r["metrics"]["primary"][k] for r in rs]) for k in METRICS},
            "test_capped": {k: _agg([r["metrics"]["capped"][k] for r in rs]) for k in METRICS},
            "test_ci95_median_seed": {k: list(v) for k, v in ci.items()},
            "val_points_capped": {k: _agg([v[k] for v in val]) for k in METRICS},
            "train_time_s": _agg([r["rec"]["train_time_s"] for r in rs]),
            "n_params": rs[0]["rec"]["n_params"],
            "fit_info": {k: v for k, v in med["rec"].get("fit_info", {}).items() if k != "history"},
        }
    # paired comparisons (EVAL-28); best classical chosen by *validation* score, not test
    pairs = []
    present = [f for f in CLASSICAL if f in fam_runs]
    if present:
        best_cls = min(present, key=lambda f: summary_fams[f]["val_points_capped"]["rmse"]["mean"])
        if "lstm" in fam_runs:
            pairs.append(("lstm", best_cls))
    if "life_prior" in fam_runs:
        pairs += [(f, "life_prior") for f in LEARNED if f in fam_runs]
        if "constant" in fam_runs:
            pairs.append(("life_prior", "constant"))
    paired = []
    for a, b in pairs:
        ra, rb = median_run[a], median_run[b]
        assert np.array_equal(ra["unit"], rb["unit"]), "evaluation points differ between models"
        d, (lo, hi) = paired_bootstrap_diff(
            ra["y"], ra["pred"], rb["pred"], rmse, n_boot=2000, seed=0
        )
        paired.append(
            {
                "a": a,
                "b": b,
                "delta_rmse": d,
                "ci95": [lo, hi],
                "ci_excludes_0": bool(lo > 0 or hi < 0),
            }
        )
    # stage analysis (EVAL-30/31) and worst engines (EVAL-33)
    stages_test, stages_val, worst = {}, {}, {}
    lengths = test_df.groupby("unit")["cycle"].max()
    for f, r in median_run.items():
        stages_test[f] = stage_metrics(r["y"], r["pred"], min_n=5)
        vd = pd.read_csv(results_dir / r["rec"]["run_id"] / "predictions_val_dense.csv")
        stages_val[f] = stage_metrics(vd["y_true"], vd["y_pred"])
        err = r["pred"] - r["y"]
        top = np.argsort(-np.abs(err))[:10]
        worst[f] = [
            {
                "unit": int(r["unit"][i]),
                "length": int(lengths[r["unit"][i]]),
                "true": float(r["y"][i]),
                "pred": float(r["pred"][i]),
                "error": float(err[i]),
            }
            for i in top
        ]
    any_rec = next(iter(per_run.values()))
    r_max = any_rec["rec"]["config"]["data"]["r_max"]
    y_any = any_rec["y"]
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_info()["commit"],
        "data_sha256": any_rec["rec"]["data_sha256"],
        "split_sha256": any_rec["rec"]["split_sha256"],
        "device": {f: median_run[f]["rec"]["env"]["device"] for f in median_run},
        "env": any_rec["rec"]["env"],
        "n_test_engines": int(len(y_any)),
        "r_max": r_max,
        "n_test_rul_above_r_max": int((y_any > r_max).sum()) if r_max else 0,
        "families": summary_fams,
        "paired_comparisons": paired,
        "stage_metrics_test": stages_test,
        "stage_metrics_val_dense": stages_val,
        "worst_engines_test": worst,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    # per-engine test predictions of the median seed (small; committed for plots)
    rows = []
    for f, r in median_run.items():
        rows += [
            {"family": f, "unit": int(u), "cycle": int(c), "y_true": float(y), "y_pred": float(p)}
            for u, c, y, p in zip(r["unit"], r["cycle"], r["y"], r["pred"], strict=True)
        ]
    pd.DataFrame(rows).to_csv(out_dir / "test_predictions_median_seed.csv", index=False)
    # small committed copies so `engine-rul report` works from results/final/ alone
    vd_frames = []
    for f, r in median_run.items():
        vd = pd.read_csv(results_dir / r["rec"]["run_id"] / "predictions_val_dense.csv")
        vd_frames.append(vd.assign(family=f))
    pd.concat(vd_frames).to_csv(out_dir / "val_dense_predictions_median_seed.csv", index=False)
    extras = {
        "lstm_histories": {
            str(r["seed"]): r["rec"]["fit_info"]["history"] for r in fam_runs.get("lstm", [])
        },
        "permutation_importance": {
            f: median_run[f]["rec"]["permutation_importance"]
            for f in median_run
            if "permutation_importance" in median_run[f]["rec"]
        },
        "ridge_coefficients": median_run["ridge"]["rec"].get("ridge_coefficients")
        if "ridge" in median_run
        else None,
    }
    (out_dir / "training_details.json").write_text(json.dumps(extras, indent=1) + "\n")
    return path
