"""Phase 2: repeated, engine-grouped, nested cross-validation and ablations (validation only).

Design (see docs/phase2.md):

* ``splits/cv_fd001.json`` holds R repeats x K lifetime-stratified folds of *training* engines.
* Each outer fold's held-out engines are scored only once, by models whose preprocessing,
  hyperparameters and early stopping were decided on the other (development) engines:
  development engines are split into inner-train (fit + preprocessing) and inner-val
  (hyperparameter selection and LSTM early stopping).
* Evaluation endpoints (truncation cycles) are drawn per engine by a named strategy with a
  deterministic seed, and every model/variant is scored on the identical endpoints.
* Nothing here reads the official test files or labels.
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
from engine_rul.config import Config, load_config
from engine_rul.errors import ConfigError
from engine_rul.features import WindowSet, build_windows
from engine_rul.metrics import (
    STAGE_BINS,
    cluster_bootstrap_rmse,
    engine_errors,
    engine_metrics,
    kendall_tau,
    paired_cluster_bootstrap,
)
from engine_rul.models import build_model
from engine_rul.preprocess import Preprocessor
from engine_rul.provenance import experiment_dir, source_provenance, write_manifest
from engine_rul.utils import canonical_hash, environment_info, set_seed, sha256_file

log = logging.getLogger(__name__)

STRATEGIES = ("uniform", "late", "stratified")
TIE_TOL = 0.01  # same tie rule as Phase 1 (EVAL-12)

# --------------------------------------------------------------------------- fold definitions


def make_cv_folds(
    train_df: pd.DataFrame,
    n_folds: int = 5,
    seeds=(101, 202, 303),
    subset: str = "FD001",
    train_file_sha256: str | None = None,
) -> dict:
    """Repeated grouped K-fold: every engine is in exactly one fold per repeat.

    Lifetime-stratified: engines sorted by lifetime are cut into consecutive blocks of
    ``n_folds`` and each block's engines are randomly assigned to distinct folds."""
    life = train_df.groupby("unit")["cycle"].max()
    order = life.sort_values(kind="stable").index.to_numpy()
    repeats = []
    for s in seeds:
        rng = np.random.default_rng(s)
        fold_of = {}
        for b in range(0, len(order), n_folds):
            block = order[b : b + n_folds]
            for u, f in zip(block, rng.permutation(n_folds)[: len(block)], strict=True):
                fold_of[int(u)] = int(f)
        repeats.append(
            {
                "seed": int(s),
                "folds": [sorted(u for u, f in fold_of.items() if f == k) for k in range(n_folds)],
            }
        )
    return {
        "subset": subset,
        "n_folds": n_folds,
        "method": "repeated lifetime-stratified grouped K-fold (engine = group)",
        "sha256_of_train_file": train_file_sha256,
        "repeats": repeats,
    }


def load_cv_folds(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"fold file {path} not found; run `engine-rul cv make-folds`")
    spec = json.loads(path.read_text())
    for r in spec["repeats"]:
        flat = [u for f in r["folds"] for u in f]
        if len(flat) != len(set(flat)):
            raise ConfigError(f"{path}: an engine appears in two folds of repeat seed {r['seed']}")
    return spec


def inner_roles(
    dev_units, life: pd.Series, fractions: dict[str, float], seed
) -> dict[str, list[int]]:
    """Split development engines into roles, e.g. {train: .8, val: .2} or {train: .6, val: .15,
    cal: .25}. The first role takes the remainder; every other role is drawn lifetime-stratified
    (one engine per lifetime bin)."""
    rng = np.random.default_rng(seed)
    names = list(fractions)
    n = len(dev_units)
    counts = {r: int(round(fractions[r] * n)) for r in names[1:]}
    remaining = sorted(dev_units, key=lambda u: (life[u], u))
    roles: dict[str, list[int]] = {}
    for r in names[1:]:
        k = counts[r]
        if k <= 0 or k >= len(remaining):
            raise ConfigError(f"role {r!r} would get {k} of {len(remaining)} engines")
        picked = [int(rng.choice(b)) for b in np.array_split(np.asarray(remaining), k)]
        roles[r] = sorted(picked)
        remaining = [u for u in remaining if u not in set(picked)]
    roles[names[0]] = sorted(int(u) for u in remaining)
    return {r: roles[r] for r in names}


# --------------------------------------------------------------------------- endpoints


@dataclass(frozen=True)
class EndpointSpec:
    n_per_engine: int = 10
    min_cycle: int = 40
    late_horizon: float = 125
    seed: int = 20261111
    strategies: tuple[str, ...] = STRATEGIES
    primary: str = "uniform"

    @classmethod
    def from_dict(cls, d: dict) -> EndpointSpec:
        d = dict(d)
        if "strategies" in d:
            d["strategies"] = tuple(d["strategies"])
        spec = cls(**d)
        bad = set(spec.strategies) - set(STRATEGIES)
        if bad or spec.primary not in spec.strategies:
            raise ConfigError(
                f"strategies must be within {STRATEGIES} and include primary {spec.primary!r}"
            )
        return spec


def sample_endpoints(
    life: pd.Series, units, strategy: str, spec: EndpointSpec, salt: int = 0
) -> np.ndarray:
    """Deterministic truncation cycles per engine -> int[N, 2] of (unit, cycle).

    * uniform: cycles uniform on [min_cycle, T].
    * late: cycles with true RUL <= late_horizon (late-life / near-failure focus).
    * stratified: spread across true-RUL stage bins (STAGE_BINS), round-robin over non-empty bins.
    Each engine's draw depends only on (seed, salt, strategy, unit), not on which other engines
    are in the set."""
    s_idx = STRATEGIES.index(strategy)
    out = []
    for u in sorted(int(x) for x in units):
        T = int(life[u])
        rng = np.random.default_rng([spec.seed, salt, s_idx, u])
        cand = np.arange(spec.min_cycle, T + 1)
        if len(cand) == 0:
            raise ConfigError(f"engine {u} (life {T}) is shorter than min_cycle {spec.min_cycle}")
        rul = T - cand
        k = spec.n_per_engine
        if strategy == "uniform":
            pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
        elif strategy == "late":
            pool = cand[rul <= spec.late_horizon]
            pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
        else:
            bins = [
                cand[(rul >= lo) & (rul < hi)]
                for lo, hi in zip(STAGE_BINS[:-1], STAGE_BINS[1:], strict=True)
            ]
            pools = [list(rng.permutation(b)) for b in bins if len(b)]
            order = rng.permutation(len(pools))
            pick, i = [], 0
            while len(pick) < min(k, len(cand)):
                pool = pools[order[i % len(order)]]
                if pool:
                    pick.append(pool.pop())
                i += 1
        out += [[u, int(c)] for c in sorted(pick)]
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


# --------------------------------------------------------------------------- fold data


@dataclass
class FoldData:
    pre: Preprocessor
    train: WindowSet
    val_dense: WindowSet
    inner: dict[str, WindowSet]  # strategy -> inner-val endpoints (selection / early stopping)
    outer: dict[str, WindowSet]  # strategy -> held-out endpoints (evaluation)
    extra: dict[str, dict[str, WindowSet]]  # other roles (e.g. "cal") -> strategy -> endpoints


def prepare_fold(
    df: pd.DataFrame, cfg: Config, roles: dict, eval_units, endpoints: dict
) -> FoldData:
    """Fit preprocessing on the fold's inner-train engines only, then build windows.

    ``endpoints`` maps role ("val", "outer", other roles) -> strategy -> int[N, 2]."""
    dc = cfg.data
    rows = {r: df[df["unit"].isin(u)] for r, u in {**roles, "outer": list(eval_units)}.items()}
    pre = Preprocessor(dc.drop_columns, dc.r_max, dc.expected_features).fit(rows["train"])
    t = {r: pre.transform(x) for r, x in rows.items()}
    f = pre.feature_names

    def ws(role):
        return {s: build_windows(t[role], f, dc, end_points=p) for s, p in endpoints[role].items()}

    return FoldData(
        pre=pre,
        train=build_windows(t["train"], f, dc),
        val_dense=build_windows(t["val"], f, dc),
        inner=ws("val"),
        outer=ws("outer"),
        extra={r: ws(r) for r in endpoints if r not in ("val", "outer")},
    )


def capped(y: np.ndarray, cap) -> np.ndarray:
    return np.minimum(y, cap) if cap else y


def rmse_capped(ws: WindowSet, pred: np.ndarray, cap) -> float:
    return float(np.sqrt(np.mean((pred - capped(ws.y_true, cap)) ** 2)))


def fit_model(cfg: Config, fd: FoldData, primary: str):
    """Train one config on a fold; LSTM early stopping on dense inner-val windows or on the
    inner-val selection endpoints (cfg.train.early_stopping)."""
    set_seed(cfg.seed)
    model = build_model(cfg)
    es = fd.inner[primary] if cfg.train.early_stopping == "endpoints" else fd.val_dense
    t0 = time.perf_counter()
    info = model.fit(fd.train, es, cfg)
    return (
        model,
        {k: v for k, v in (info or {}).items() if k != "history"},
        time.perf_counter() - t0,
    )


def save_fold_model(
    d: Path, model, cfg: Config, pre: Preprocessor, roles: dict, eval_units
) -> None:
    """Same layout as a Phase-1 run directory, so ``models.load_model`` and
    ``inference.predict_engines`` work on it."""
    d.mkdir(parents=True, exist_ok=True)
    model.save(d)
    (d / "preprocessor.json").write_text(json.dumps(pre.to_dict(), indent=1))
    (d / "run.json").write_text(
        json.dumps(
            {
                "run_id": d.name,
                "family": cfg.model.family,
                "seed": cfg.seed,
                "stage": "phase2",
                "config": cfg.to_dict(),
                "config_hash": cfg.hash(),
                "engines": {**roles, "outer": list(eval_units)},
            },
            indent=1,
        )
    )


def predict(model, ws: WindowSet) -> np.ndarray:
    return np.clip(model.predict(ws), 0, None)  # ADR-13, same post-processing as Phase 1


def grid_combos(grid_path) -> list[dict]:
    if not grid_path:
        return [{}]
    grid = yaml.safe_load(Path(grid_path).read_text())
    keys = list(grid)
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*(grid[k] for k in keys))]


def select(scores: list[float]) -> int:
    best = min(scores)
    return next(i for i, s in enumerate(scores) if s <= best + TIE_TOL)


# --------------------------------------------------------------------------- experiment config


def load_spec(path: str | Path, required=("fold_file", "repeats", "families", "endpoints")) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    for k in ("name", "category", *required):
        if k not in spec:
            raise ConfigError(f"{path}: missing key {k!r}")
    return spec


def family_config(fam_spec: dict, seed: int, early_stopping: str | None) -> Config:
    cfg = load_config(fam_spec["config"], {"seed": seed, "tracking": {"mlflow_uri": None}})
    if cfg.model.family == "lstm" and early_stopping:
        cfg = cfg.replace(train={"early_stopping": early_stopping})
    return cfg


def _common_manifest(spec: dict, spec_path, df_sha: str, folds_path) -> dict:
    return {
        "category": spec["category"],
        "name": spec["name"],
        "spec_file": str(spec_path),
        "spec": spec,
        "spec_sha256": canonical_hash(spec),
        "fold_file": str(folds_path),
        "fold_file_sha256": sha256_file(folds_path),
        "train_data_sha256": df_sha,
        "provenance": source_provenance(),
        "env": environment_info("cpu"),
        "official_test_set_used": False,
    }


def _log_mlflow(spec: dict, exp_dir: Path, metrics: dict) -> None:
    uri = spec.get("mlflow_uri", "sqlite:///mlflow.db")
    if not uri:
        return
    try:
        import mlflow

        from engine_rul.tracking import _flatten

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(f"fd001-phase2-{spec['category']}")
        with mlflow.start_run(run_name=spec["name"]):
            mlflow.log_params({k: str(v)[:500] for k, v in _flatten(spec).items()})
            mlflow.set_tags({"category": spec["category"], "official_test_set_used": "false"})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
            for name in ("summary.json", "manifest.json", "fold_metrics.csv"):
                if (exp_dir / name).exists():
                    mlflow.log_artifact(str(exp_dir / name))
    except Exception as e:  # noqa: BLE001 - tracking never fails an experiment (TRACK-04)
        log.warning("MLflow logging failed for %s: %s", spec["name"], e)


def _fold_iter(spec: dict, folds: dict, repeats):
    for r in repeats:
        rep = folds["repeats"][r]
        for k, eval_units in enumerate(rep["folds"]):
            dev = sorted(u for j, f in enumerate(rep["folds"]) if j != k for u in f)
            yield r, rep["seed"], k, eval_units, dev


def fold_endpoints(
    life, roles, eval_units, ep: EndpointSpec, rep_seed: int, k: int, role_names
) -> dict:
    """Outer endpoints depend only on (repeat seed, engine): each engine is outer exactly once
    per repeat. Inner-role endpoints are additionally salted by the outer fold."""
    out = {
        "outer": {
            s: sample_endpoints(life, eval_units, s, ep, salt=rep_seed) for s in ep.strategies
        }
    }
    for role in role_names:
        out[role] = {
            s: sample_endpoints(life, roles[role], s, ep, salt=rep_seed * 100 + k + 1)
            for s in ep.strategies
        }
    return out


# --------------------------------------------------------------------------- nested CV


def run_cv(
    spec_path: str | Path,
    root: str | Path = "results/phase2",
    force: bool = False,
    data_dir: str | None = None,
) -> Path:
    spec = load_spec(spec_path)
    exp = experiment_dir(root, spec["category"], spec["name"], force)
    folds_path = Path(spec["fold_file"])
    folds = load_cv_folds(folds_path)
    ep = EndpointSpec.from_dict(spec["endpoints"])
    fractions = spec.get("inner_fractions", {"train": 0.8, "val": 0.2})
    first = family_config(next(iter(spec["families"].values())), 0, None)
    ddir = data_dir or first.data.data_dir
    df = D.load_train(ddir, first.data.subset)
    df_sha = sha256_file(D.subset_path(ddir, first.data.subset, "train"))
    life = df.groupby("unit")["cycle"].max()
    preds, sel_rows, roles_log = [], [], []
    t_start = time.perf_counter()
    for r, rep_seed, k, eval_units, dev in _fold_iter(spec, folds, spec["repeats"]):
        roles = inner_roles(dev, life, fractions, seed=[rep_seed, k])
        roles_log.append({"repeat": r, "fold": k, "outer": eval_units, **roles})
        endpoints = fold_endpoints(
            life, roles, eval_units, ep, rep_seed, k, [x for x in roles if x != "train"]
        )
        cache: dict[str, FoldData] = {}
        for fam, fs in spec["families"].items():
            base = family_config(fs, seed=r, early_stopping=spec.get("early_stopping"))
            if data_dir:
                base = base.replace(data={"data_dir": data_dir})
            key = canonical_hash(base.to_dict()["data"])
            if key not in cache:
                cache[key] = prepare_fold(df, base, roles, eval_units, endpoints)
            fd = cache[key]
            cands = []
            for i, combo in enumerate(grid_combos(fs.get("grid"))):
                cfg = base.replace(model={"params": combo})
                model, info, secs = fit_model(cfg, fd, ep.primary)
                inner = {
                    s: rmse_capped(w, predict(model, w), cfg.data.r_max)
                    for s, w in fd.inner.items()
                }
                outer = {s: predict(model, w) for s, w in fd.outer.items()}
                cands.append((cfg, inner, outer, model))
                sel_rows.append(
                    {
                        "repeat": r,
                        "fold": k,
                        "family": fam,
                        "candidate": i,
                        "params": json.dumps(cfg.model.params, sort_keys=True),
                        **{f"inner_rmse_{s}": v for s, v in inner.items()},
                        "train_time_s": secs,
                        **{
                            f"fit_{a}": b
                            for a, b in info.items()
                            if isinstance(b, int | float | str)
                        },
                    }
                )
            chosen = {s: select([c[1][s] for c in cands]) for s in ep.strategies}
            for row in sel_rows[-len(cands) :]:
                row.update(
                    {f"selected_by_{s}": row["candidate"] == chosen[s] for s in ep.strategies}
                )
            cfg, _, outer, model = cands[chosen[ep.primary]]
            for s, w in fd.outer.items():
                preds.append(
                    pd.DataFrame(
                        {
                            "repeat": r,
                            "fold": k,
                            "strategy": s,
                            "family": fam,
                            "variant": "base",
                            "role": "outer",
                            "unit": w.unit,
                            "cycle": w.cycle,
                            "y_true": w.y_true,
                            "y_pred": outer[s],
                        }
                    )
                )
            if fd.extra or spec.get("save_models"):
                for role, sets in fd.extra.items():
                    for s, w in sets.items():
                        preds.append(
                            pd.DataFrame(
                                {
                                    "repeat": r,
                                    "fold": k,
                                    "strategy": s,
                                    "family": fam,
                                    "variant": "base",
                                    "role": role,
                                    "unit": w.unit,
                                    "cycle": w.cycle,
                                    "y_true": w.y_true,
                                    "y_pred": predict(model, w),
                                }
                            )
                        )
                if spec.get("save_models"):
                    save_fold_model(
                        exp / "models" / f"r{r}_f{k}_{fam}", model, cfg, fd.pre, roles, eval_units
                    )
            log.info(
                "cv r%d f%d %s: selected %s (inner %s RMSE %.3f)",
                r,
                k,
                fam,
                cfg.model.params,
                ep.primary,
                cands[chosen[ep.primary]][1][ep.primary],
            )
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(exp / "predictions.csv.gz", index=False)
    pd.DataFrame(sel_rows).to_csv(exp / "selection.csv", index=False)
    write_manifest(
        exp,
        {
            **_common_manifest(spec, spec_path, df_sha, folds_path),
            "roles": roles_log,
            "wall_time_s": time.perf_counter() - t_start,
            "r_max_truth_cap": first.data.r_max,
        },
    )
    summary = analyze_cv(exp)
    if spec["category"] == "conformal":
        from engine_rul.conformal import analyze_conformal

        analyze_conformal(exp)
    _log_mlflow(
        spec,
        exp,
        {
            f"{f}_rmse_capped": v["pooled"]["capped"]["rmse"]
            for f, v in summary["strategies"][ep.primary]["families"].items()
        },
    )
    return exp


# --------------------------------------------------------------------------- ablations


def run_ablation(
    spec_path: str | Path,
    root: str | Path = "results/phase2",
    force: bool = False,
    data_dir: str | None = None,
) -> Path:
    """One-dimension-at-a-time variants of the per-fold, nested-selected configs of a CV run.

    Hyperparameters come from the source CV experiment's inner-validation selection, so they never
    depend on the outer fold. Same roles, endpoints and early-stopping rule as the source run."""
    spec = load_spec(spec_path, required=("source_cv", "repeats", "variants"))
    exp = experiment_dir(root, spec["category"], spec["name"], force)
    src = Path(spec["source_cv"])
    src_manifest = json.loads((src / "manifest.json").read_text())
    sel = pd.read_csv(src / "selection.csv")
    cv_spec = src_manifest["spec"]
    folds_path = Path(cv_spec["fold_file"])
    folds = load_cv_folds(folds_path)
    if sha256_file(folds_path) != src_manifest["fold_file_sha256"]:
        raise ConfigError("fold file changed since the source CV experiment")
    ep = EndpointSpec.from_dict(cv_spec["endpoints"])
    fractions = cv_spec.get("inner_fractions", {"train": 0.8, "val": 0.2})
    fam_specs = cv_spec["families"]
    first = family_config(next(iter(fam_specs.values())), 0, None)
    ddir = data_dir or first.data.data_dir
    df = D.load_train(ddir, first.data.subset)
    df_sha = sha256_file(D.subset_path(ddir, first.data.subset, "train"))
    life = df.groupby("unit")["cycle"].max()
    max_window = max(
        [first.data.window] + [v.get("data", {}).get("window", 0) for v in spec["variants"]]
    )
    if max_window > ep.min_cycle:
        raise ConfigError(f"variant window {max_window} exceeds endpoint min_cycle {ep.min_cycle}")
    preds, info_rows = [], []
    t_start = time.perf_counter()
    for r, rep_seed, k, eval_units, dev in _fold_iter(
        spec, folds, sorted(set(spec["repeats"]) | set(spec.get("lstm_repeats", [])))
    ):
        roles = inner_roles(dev, life, fractions, seed=[rep_seed, k])
        endpoints = fold_endpoints(
            life, roles, eval_units, ep, rep_seed, k, [x for x in roles if x != "train"]
        )
        endpoints = {
            role: {ep.primary: e[ep.primary]} for role, e in endpoints.items()
        }  # primary only
        cache: dict[str, FoldData] = {}
        for v in spec["variants"]:
            for fam in v["families"]:
                reps = (
                    spec.get("lstm_repeats", spec["repeats"]) if fam == "lstm" else spec["repeats"]
                )
                if r not in reps:
                    continue
                row = sel[
                    (sel.repeat == r)
                    & (sel.fold == k)
                    & (sel.family == fam)
                    & sel[f"selected_by_{ep.primary}"]
                ]
                params = {**json.loads(row["params"].iloc[0]), **v.get("params", {})}
                base = family_config(
                    fam_specs[fam], seed=r, early_stopping=cv_spec.get("early_stopping")
                )
                cfg = base.replace(
                    model={"params": params}, data=v.get("data", {}), train=v.get("train", {})
                )
                if data_dir:
                    cfg = cfg.replace(data={"data_dir": data_dir})
                key = canonical_hash(cfg.to_dict()["data"])
                if key not in cache:
                    cache[key] = prepare_fold(df, cfg, roles, eval_units, endpoints)
                fd = cache[key]
                model, info, secs = fit_model(cfg, fd, ep.primary)
                w = fd.outer[ep.primary]
                preds.append(
                    pd.DataFrame(
                        {
                            "repeat": r,
                            "fold": k,
                            "strategy": ep.primary,
                            "family": fam,
                            "variant": v["name"],
                            "unit": w.unit,
                            "cycle": w.cycle,
                            "y_true": w.y_true,
                            "y_pred": predict(model, w),
                        }
                    )
                )
                info_rows.append(
                    {
                        "repeat": r,
                        "fold": k,
                        "family": fam,
                        "variant": v["name"],
                        "params": json.dumps(params, sort_keys=True),
                        "train_time_s": secs,
                        "n_inputs": len(fd.train.feature_names),
                        **{
                            f"fit_{a}": b
                            for a, b in info.items()
                            if isinstance(b, int | float | str)
                        },
                    }
                )
        log.info("ablation r%d f%d done", r, k)
    pred = pd.concat(preds, ignore_index=True)
    pred.to_csv(exp / "predictions.csv.gz", index=False)
    pd.DataFrame(info_rows).to_csv(exp / "runs.csv", index=False)
    write_manifest(
        exp,
        {
            **_common_manifest(spec, spec_path, df_sha, folds_path),
            "source_cv": str(src),
            "source_cv_spec_sha256": src_manifest["spec_sha256"],
            "wall_time_s": time.perf_counter() - t_start,
            "r_max_truth_cap": first.data.r_max,
        },
    )
    summary = analyze_ablation(exp)
    _log_mlflow(
        spec,
        exp,
        {
            f"{f}_{x['variant']}_delta_rmse": x["delta_rmse"]
            for f, rows in summary["families"].items()
            for x in rows
            if x["variant"] != "base"
        },
    )
    return exp


# --------------------------------------------------------------------------- analysis


def _engine_table(p: pd.DataFrame, cap) -> dict[str, dict]:
    """Per-engine errors for capped and uncapped truth (pooled over repeats)."""
    return {
        "capped": engine_errors(p["unit"], capped(p["y_true"].to_numpy(), cap), p["y_pred"]),
        "uncapped": engine_errors(p["unit"], p["y_true"], p["y_pred"]),
    }


def _fold_metrics(p: pd.DataFrame, cap) -> pd.DataFrame:
    rows = []
    keys = ["strategy", "family", "variant", "repeat", "fold"]
    for key, g in p.groupby(keys, sort=True):
        for truth, e in _engine_table(g, cap).items():
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "truth": truth,
                    **engine_metrics(e),
                    "n_points": len(g),
                }
            )
    return pd.DataFrame(rows)


def _ranks(scores: dict[str, float]) -> dict[str, int]:
    order = sorted(scores, key=lambda f: scores[f])
    return {f: order.index(f) + 1 for f in scores}


def analyze_cv(exp: str | Path, n_boot: int = 2000) -> dict:
    exp = Path(exp)
    man = json.loads((exp / "manifest.json").read_text())
    cap = man["r_max_truth_cap"]
    p = pd.read_csv(exp / "predictions.csv.gz")
    if "role" in p:
        p = p[p["role"] == "outer"]
    sel = pd.read_csv(exp / "selection.csv")
    fm = _fold_metrics(p, cap)
    fm.to_csv(exp / "fold_metrics.csv", index=False)
    fams = list(dict.fromkeys(p["family"]))
    primary = man["spec"]["endpoints"].get("primary", "uniform")
    out = {
        "experiment": man["name"],
        "category": man["category"],
        "primary_strategy": primary,
        "truth_cap": cap,
        "n_repeats": int(p["repeat"].nunique()),
        "n_outer_folds": int(fm[["repeat", "fold"]].drop_duplicates().shape[0]),
        "n_engines": int(p["unit"].nunique()),
        "strategies": {},
    }
    for s, ps in p.groupby("strategy", sort=False):
        eng = {f: _engine_table(ps[ps.family == f], cap) for f in fams}
        units = eng[fams[0]]["capped"]["unit"]
        assert all(np.array_equal(eng[f]["capped"]["unit"], units) for f in fams), (
            "engines differ between models"
        )
        sfm = fm[(fm.strategy == s) & (fm.truth == "capped")]
        res = {"families": {}, "n_points": int(len(ps) / len(fams))}
        for f in fams:
            ff = sfm[sfm.family == f]
            per_rep = ff.groupby("repeat")["rmse"].mean()
            res["families"][f] = {
                "pooled": {t: engine_metrics(e) for t, e in eng[f].items()},
                "rmse_ci95_engine_bootstrap": {
                    t: list(cluster_bootstrap_rmse(e["mse"], n_boot, 0)) for t, e in eng[f].items()
                },
                "fold_rmse_capped": {
                    "mean": float(ff["rmse"].mean()),
                    "std": float(ff["rmse"].std(ddof=1)),
                    "min": float(ff["rmse"].min()),
                    "max": float(ff["rmse"].max()),
                    "n_folds": int(len(ff)),
                },
                "repeat_mean_rmse_capped": per_rep.round(6).tolist(),
            }
        # ranking stability across outer folds (capped truth)
        piv = sfm.pivot_table(index=["repeat", "fold"], columns="family", values="rmse")
        ranks = piv.rank(axis=1, method="min")
        res["ranking"] = {
            f: {
                "mean_rank": float(ranks[f].mean()),
                "rank1_frac": float((ranks[f] == 1).mean()),
                "pooled_rank": _ranks(
                    {g: res["families"][g]["pooled"]["capped"]["rmse"] for g in fams}
                )[f],
            }
            for f in fams
        }
        # bootstrap rank-1 frequency over engines
        rng = np.random.default_rng(1)
        M = np.column_stack([eng[f]["capped"]["mse"] for f in fams])
        idx = rng.integers(0, len(M), size=(n_boot, len(M)))
        winners = np.argmin(M[idx].mean(axis=1), axis=1)
        for i, f in enumerate(fams):
            res["ranking"][f]["bootstrap_rank1_frac"] = float((winners == i).mean())
        # paired comparisons between all learned families + vs life prior
        pairs = [(a, b) for a, b in itertools.combinations(fams, 2) if a not in ("constant",)]
        res["paired"] = []
        for a, b in pairs:
            for t in ("capped", "uncapped"):
                d, (lo, hi) = paired_cluster_bootstrap(
                    eng[a][t]["mse"], eng[b][t]["mse"], n_boot, seed=2
                )
                res["paired"].append(
                    {
                        "a": a,
                        "b": b,
                        "truth": t,
                        "delta_rmse": d,
                        "ci95": [lo, hi],
                        "ci_excludes_0": bool(lo > 0 or hi < 0),
                    }
                )
        out["strategies"][s] = res
    # strategy effects on rankings
    strat = list(out["strategies"])
    out["strategy_rank_agreement"] = [
        {
            "a": a,
            "b": b,
            "kendall_tau": kendall_tau(
                [out["strategies"][a]["families"][f]["pooled"]["capped"]["rmse"] for f in fams],
                [out["strategies"][b]["families"][f]["pooled"]["capped"]["rmse"] for f in fams],
            ),
        }
        for a, b in itertools.combinations(strat, 2)
    ]
    # would inner-validation selection change under another strategy?
    agree = {}
    for s in strat:
        chosen = sel[sel[f"selected_by_{s}"]].set_index(["repeat", "fold", "family"])["candidate"]
        prim = sel[sel[f"selected_by_{primary}"]].set_index(["repeat", "fold", "family"])[
            "candidate"
        ]
        agree[s] = {
            f: float((chosen.xs(f, level="family") == prim.xs(f, level="family")).mean())
            for f in fams
            if sel[sel.family == f]["candidate"].max() > 0
        }
    out["selection_agreement_with_primary"] = agree
    out["selected_params"] = {
        f: sel[(sel.family == f) & sel[f"selected_by_{primary}"]]["params"].value_counts().to_dict()
        for f in fams
    }
    (exp / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def analyze_ablation(exp: str | Path, n_boot: int = 2000) -> dict:
    exp = Path(exp)
    man = json.loads((exp / "manifest.json").read_text())
    cap = man["r_max_truth_cap"]
    p = pd.read_csv(exp / "predictions.csv.gz")
    fm = _fold_metrics(p, cap)
    fm.to_csv(exp / "fold_metrics.csv", index=False)
    # determinism check: the recomputed base must reproduce the source CV's selected-model predictions
    keys = ["repeat", "fold", "strategy", "family", "unit", "cycle"]
    src = pd.read_csv(Path(man["source_cv"]) / "predictions.csv.gz")
    if "role" in src:
        src = src[src["role"] == "outer"].drop(columns="role")
    m = p[p.variant == "base"].merge(src, on=keys, suffixes=("", "_cv"), validate="1:1")
    check = {f: float(np.abs(g["y_pred"] - g["y_pred_cv"]).max()) for f, g in m.groupby("family")}
    out = {
        "experiment": man["name"],
        "category": man["category"],
        "truth": "capped",
        "truth_cap": cap,
        "base_vs_source_cv_max_abs_diff": check,
        "families": {},
    }
    for fam, pf in p.groupby("family", sort=False):
        base = _engine_table(pf[pf.variant == "base"], cap)["capped"]
        rows = []
        for v, pv in pf.groupby("variant", sort=False):
            e = _engine_table(pv, cap)["capped"]
            assert np.array_equal(e["unit"], base["unit"])
            d, (lo, hi) = paired_cluster_bootstrap(e["mse"], base["mse"], n_boot, seed=3)
            ff = fm[(fm.family == fam) & (fm.variant == v) & (fm.truth == "capped")]
            rows.append(
                {
                    "variant": v,
                    **engine_metrics(e),
                    "delta_rmse": d,
                    "ci95": [lo, hi],
                    "ci_excludes_0": bool(lo > 0 or hi < 0),
                    "n_folds": int(len(ff)),
                    "repeats": sorted(pv["repeat"].unique().tolist()),
                    "fold_rmse_mean": float(ff["rmse"].mean()),
                    "fold_rmse_std": float(ff["rmse"].std(ddof=1)),
                }
            )
        out["families"][fam] = rows
    (exp / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out
