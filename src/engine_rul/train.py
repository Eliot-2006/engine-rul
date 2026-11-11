"""Training harness: data preparation, single runs, tuning, sensitivity, final multi-seed runs.

Nothing in this module reads test files or test labels (EVAL-20, LEAK-06/07).
"""

from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine_rul import data as D
from engine_rul.config import Config, dump_config, load_config
from engine_rul.errors import ConfigError
from engine_rul.features import WindowSet, make_windows
from engine_rul.metrics import all_metrics
from engine_rul.models import build_model
from engine_rul.preprocess import Preprocessor
from engine_rul.split import load_split
from engine_rul.tracking import track_run
from engine_rul.utils import (
    REPO_ROOT,
    canonical_hash,
    environment_info,
    git_info,
    resolve_device,
    set_seed,
    sha256_file,
)

log = logging.getLogger(__name__)

FINAL_DIR = REPO_ROOT / "configs" / "final"
DETERMINISTIC = ("constant", "life_prior", "ridge")  # trained once in final (EVAL-26)
SELECTION_METRIC = ("val_points_capped", "rmse")  # EVAL-12
REFERENCE_CAP = 125.0  # fixed reference truth cap so different R_max settings stay comparable


@dataclass
class Prepared:
    pre: Preprocessor
    train: WindowSet  # all windows (t >= W) of train-split engines
    val_dense: WindowSet  # all windows of validation engines (early stopping, stage analysis)
    val_points: WindowSet  # 10 truncation points per validation engine (model selection)
    data_sha256: str
    split_sha256: str


def prepare(cfg: Config) -> Prepared:
    dc = cfg.data
    train_path = D.subset_path(dc.data_dir, dc.subset, "train")
    df = D.load_train(dc.data_dir, dc.subset)
    if dc.check_invariants:
        inv = D.load_manifest()["subsets"].get(dc.subset, {}).get("invariants")
        if inv:
            D.check_train_invariants(df, inv)  # DATA-10
    split = load_split(dc.split_file)
    units = set(df["unit"].unique().tolist())
    if (
        set(split.train_units) & set(split.val_units)
        or set(split.train_units) | set(split.val_units) != units
    ):
        raise ConfigError("split file units are not a disjoint partition of the training engines")
    tr_rows = df[df["unit"].isin(split.train_units)]
    va_rows = df[df["unit"].isin(split.val_units)]
    pre = Preprocessor(dc.drop_columns, dc.r_max, dc.expected_features).fit(tr_rows)  # LEAK-01/02
    tr, va = pre.transform(tr_rows), pre.transform(va_rows)
    feats = pre.feature_names
    return Prepared(
        pre=pre,
        train=make_windows(tr, feats, dc.window, pad_short=dc.pad_short),
        val_dense=make_windows(va, feats, dc.window, pad_short=dc.pad_short),
        val_points=make_windows(
            va, feats, dc.window, end_points=split.val_eval_points, pad_short=dc.pad_short
        ),
        data_sha256=sha256_file(train_path),
        split_sha256=sha256_file(dc.split_file),
    )


def val_metrics(cfg: Config, y_true: np.ndarray, pred: np.ndarray) -> dict:
    """Uncapped, config-capped and reference-capped (125) truth variants."""
    cap = cfg.data.r_max
    return {
        "uncapped": all_metrics(y_true, pred),
        "capped": all_metrics(np.minimum(y_true, cap) if cap else y_true, pred),
        "cap125": all_metrics(np.minimum(y_true, REFERENCE_CAP), pred),
    }


def _pred_frame(ws: WindowSet, pred: np.ndarray, cap) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": ws.unit,
            "cycle": ws.cycle,
            "y_true": ws.y_true,
            "y_capped": np.minimum(ws.y_true, cap) if cap else ws.y_true,
            "y_pred": pred,
        }
    )


def train_one(cfg: Config, stage: str = "dev", prepared: Prepared | None = None) -> dict:
    """Train one config/seed, evaluate on validation, save artifacts; returns run.json record."""
    device = resolve_device(cfg.device) if cfg.model.family == "lstm" else "cpu"
    set_seed(cfg.seed)
    prep = prepared or prepare(cfg)
    pre_dict = prep.pre.to_dict()
    record = {
        "config": cfg.to_dict(),
        "config_hash": cfg.hash(),
        "seed": cfg.seed,
        "git": git_info(),
        "data_sha256": prep.data_sha256,
        "split_sha256": prep.split_sha256,
        "preprocessor_sha256": canonical_hash(pre_dict),
        "env": environment_info(device),
        "n_train_windows": len(prep.train),
        "n_val_dense_windows": len(prep.val_dense),
        "n_val_points": len(prep.val_points),
    }
    model = build_model(cfg)
    with track_run(cfg, stage, record) as run:
        t0 = time.perf_counter()
        fit_info = model.fit(prep.train, prep.val_dense, cfg, workdir=run.dir, tracker=run)
        train_time = time.perf_counter() - t0
        p_points = np.clip(model.predict(prep.val_points), 0, None)  # ADR-13
        p_dense = np.clip(model.predict(prep.val_dense), 0, None)
        vm_points = val_metrics(cfg, prep.val_points.y_true, p_points)
        vm_dense = val_metrics(cfg, prep.val_dense.y_true, p_dense)
        metrics = {f"val_points_{k}": v for k, v in vm_points.items()}
        metrics |= {f"val_dense_{k}": v for k, v in vm_dense.items()}
        run.log_metrics({f"{k}.{m}": v for k, d in metrics.items() for m, v in d.items()})
        run.log_metrics({"train_time_s": train_time})
        model.save(run.dir)
        (run.dir / "preprocessor.json").write_text(json.dumps(pre_dict, indent=1))
        _pred_frame(prep.val_points, p_points, cfg.data.r_max).to_csv(
            run.dir / "predictions_val.csv", index=False
        )
        _pred_frame(prep.val_dense, p_dense, cfg.data.r_max).to_csv(
            run.dir / "predictions_val_dense.csv", index=False
        )
        extras = {}
        if cfg.model.family == "ridge":
            extras["ridge_coefficients"] = model.coefficients()
        if cfg.model.family in ("ridge", "hgb"):
            extras["permutation_importance"] = permutation_importance_val(model, prep.val_points)
        run.record.update(
            {
                "metrics": metrics,
                "train_time_s": train_time,
                "n_params": model.n_params(),
                "fit_info": fit_info,
                **extras,
            }
        )
        log.info(
            "%s seed=%d val RMSE (points, capped) %.3f | train %.1fs",
            run.run_id,
            cfg.seed,
            metrics["val_points_capped"]["rmse"],
            train_time,
        )
    return run.record


def permutation_importance_val(model, ws: WindowSet, n_repeats: int = 10) -> dict[str, dict]:
    """EVAL-34: permutation importance on validation eval points (never on test)."""
    from sklearn.inspection import permutation_importance

    X = model.features(ws)
    r = permutation_importance(
        model.model,
        X,
        ws.y,
        n_repeats=n_repeats,
        random_state=0,
        scoring="neg_root_mean_squared_error",
    )
    return {
        n: {"mean": float(m), "std": float(s)}
        for n, m, s in zip(model.feature_names, r.importances_mean, r.importances_std, strict=True)
    }


def selection_score(record: dict) -> float:
    k, m = SELECTION_METRIC
    return record["metrics"][k][m]


def _grid_combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [
        dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))
    ]


def tune(
    cfg: Config,
    grid_path: str | Path | None = None,
    out_dir: str | Path = FINAL_DIR,
    summary_dir: str | Path | None = None,
) -> Path:
    """Grid search on validation only; writes configs/final/<family>.yaml (EXP-01..03).

    Tie rule (EVAL-12): among configs within 0.01 RMSE of the best, the first in grid order
    wins; grids are listed simplest-first."""
    grid = yaml.safe_load(Path(grid_path).read_text()) if grid_path else {}
    combos = _grid_combos(grid) if grid else [{}]
    prep = prepare(cfg)
    rows = []
    for combo in combos:
        c = cfg.replace(model={"params": combo})
        rec = train_one(c, "dev", prepared=prep)
        rows.append(
            {
                "params": c.model.params,
                "run_id": rec["run_id"],
                "score": selection_score(rec),
                "val_points_capped": rec["metrics"]["val_points_capped"],
                "train_time_s": rec["train_time_s"],
                "n_params": rec["n_params"],
            }
        )
    best_score = min(r["score"] for r in rows)
    chosen = next(r for r in rows if r["score"] <= best_score + 0.01)
    best_cfg = cfg.replace(model={"params": chosen["params"]})
    out = Path(out_dir) / f"{cfg.model.family}.yaml"
    dump_config(
        best_cfg,
        out,
        header=(
            f"Selected by `engine-rul tune` on validation only ({len(rows)} configs).\n"
            f"Selection metric: val RMSE on truncated eval points vs capped truth = {chosen['score']:.4f}\n"
            f"Selected run: {chosen['run_id']}"
        ),
    )
    sdir = Path(summary_dir or Path(cfg.results_dir) / "final")
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / f"tuning_{cfg.model.family}.json").write_text(
        json.dumps(
            {
                "family": cfg.model.family,
                "grid": grid,
                "n_configs": len(rows),
                "selected": chosen,
                "rows": rows,
            },
            indent=1,
        )
        + "\n"
    )
    log.info(
        "tune %s: selected %s (score %.4f) -> %s",
        cfg.model.family,
        chosen["params"],
        chosen["score"],
        out,
    )
    return out


def sensitivity(spec_path: str | Path, results_dir: str | None = None) -> Path:
    """EXP-04: validation-only sensitivity to R_max and window length."""
    spec = yaml.safe_load(Path(spec_path).read_text())
    rows = []
    for family in spec["families"]:
        base = load_config(FINAL_DIR / f"{family}.yaml")
        if results_dir:
            base = base.replace(results_dir=results_dir)
        for r_max, window in itertools.product(spec["r_max"], spec["window"]):
            c = base.replace(data={"r_max": r_max, "window": window})
            rec = train_one(c, "dev")
            m = rec["metrics"]
            rows.append(
                {
                    "family": family,
                    "r_max": r_max,
                    "window": window,
                    "run_id": rec["run_id"],
                    "val_rmse_cap125": m["val_points_cap125"]["rmse"],
                    "val_rmse_uncapped": m["val_points_uncapped"]["rmse"],
                    "val_nasa_mean_cap125": m["val_points_cap125"]["nasa_mean"],
                }
            )
    out = Path(results_dir or base.results_dir) / "final" / "sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"spec": spec, "rows": rows}, indent=1) + "\n")
    return out


def final(
    families: list[str],
    seeds: list[int],
    results_dir: str | None = None,
    final_dir: str | Path = FINAL_DIR,
) -> Path:
    """EXP-05: train frozen configs from configs/final/ with several seeds (stage=final)."""
    runs: dict[str, list[str]] = {}
    out_root = None
    for family in families:
        path = Path(final_dir) / f"{family}.yaml"
        if not path.exists():
            raise ConfigError(f"no frozen config {path}; run `engine-rul tune` for {family} first")
        cfg = load_config(path)
        if results_dir:
            cfg = cfg.replace(results_dir=results_dir)
        out_root = Path(cfg.results_dir)
        prep = prepare(cfg)
        fam_seeds = [seeds[0]] if family in DETERMINISTIC else seeds
        runs[family] = [
            train_one(cfg.replace(seed=s), "final", prepared=prep)["run_id"] for s in fam_seeds
        ]
    out = out_root / "final_runs.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(runs)
    out.write_text(json.dumps(existing, indent=1) + "\n")
    return out
