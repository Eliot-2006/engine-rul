"""Phase 2: nested CV, endpoints, calibration, inference intervals, experiment isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from engine_rul import cv
from engine_rul.config import dump_config, load_config
from engine_rul.conformal import calibrate, conformal_quantile, interval
from engine_rul.data import load_train
from engine_rul.errors import ConfigError
from engine_rul.inference import predict_engines
from engine_rul.preprocess import Preprocessor
from engine_rul.provenance import experiment_dir, refuse_overwrite

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def train_df(synth_dir):
    return load_train(synth_dir)


@pytest.fixture
def life(train_df):
    return train_df.groupby("unit")["cycle"].max()


def test_cv_folds_partition_engines_and_are_deterministic(train_df):
    spec = cv.make_cv_folds(train_df, n_folds=5, seeds=(1, 2))
    units = sorted(train_df["unit"].unique().tolist())
    for rep in spec["repeats"]:
        flat = [u for f in rep["folds"] for u in f]
        assert sorted(flat) == units  # complete, each engine exactly once
        assert all(len(f) == 2 for f in rep["folds"])
    assert spec == cv.make_cv_folds(train_df, n_folds=5, seeds=(1, 2))
    assert spec["repeats"][0]["folds"] != spec["repeats"][1]["folds"]


def test_load_cv_folds_rejects_overlap(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"repeats": [{"seed": 1, "folds": [[1, 2], [2, 3]]}]}))
    with pytest.raises(ConfigError):
        cv.load_cv_folds(p)


def test_inner_roles_partition_development_engines(life):
    dev = list(range(1, 9))
    roles = cv.inner_roles(dev, life, {"train": 0.6, "val": 0.15, "cal": 0.25}, seed=[1, 0])
    flat = [u for r in roles.values() for u in r]
    assert sorted(flat) == dev and len(set(flat)) == len(flat)
    assert all(roles[r] for r in roles)


def test_endpoints_deterministic_valid_and_independent_of_other_engines(life):
    spec = cv.EndpointSpec(n_per_engine=4, min_cycle=12, late_horizon=20)
    for s in cv.STRATEGIES:
        a = cv.sample_endpoints(life, [1, 2, 3], s, spec, salt=7)
        b = cv.sample_endpoints(life, [3], s, spec, salt=7)
        np.testing.assert_array_equal(a[a[:, 0] == 3], b)
        np.testing.assert_array_equal(a, cv.sample_endpoints(life, [1, 2, 3], s, spec, salt=7))
        assert (a[:, 1] >= spec.min_cycle).all()
        assert (a[:, 1] <= life[a[:, 0]].to_numpy()).all()
        assert len(np.unique(a, axis=0)) == len(a)
    late = cv.sample_endpoints(life, life.index, "late", spec)
    assert (life[late[:, 0]].to_numpy() - late[:, 1] <= spec.late_horizon).all()


def test_prepare_fold_fits_preprocessing_on_inner_train_only(train_df, life, synth_cfg):
    cfg = synth_cfg("ridge")
    roles = {"train": [1, 2, 3, 4, 5, 6], "val": [7, 8]}
    ep = cv.EndpointSpec(n_per_engine=3, min_cycle=12)
    endpoints = cv.fold_endpoints(life, roles, [9, 10], ep, rep_seed=1, k=0, role_names=["val"])
    fd = cv.prepare_fold(train_df, cfg, roles, [9, 10], endpoints)
    dc = cfg.data
    expect = Preprocessor(dc.drop_columns, dc.r_max, dc.expected_features).fit(
        train_df[train_df.unit.isin(roles["train"])]
    )
    assert fd.pre.to_dict() == expect.to_dict()
    assert set(fd.train.unit) == set(roles["train"])
    assert set(fd.outer["uniform"].unit) == {9, 10}
    assert set(fd.inner["uniform"].unit) == {7, 8}


def _spec(tmp_path, synth_cfg, folds_path, category="cv", **extra) -> Path:
    fams = {}
    for fam in ("constant", "ridge"):
        p = tmp_path / f"{fam}.yaml"
        dump_config(synth_cfg(fam), p)
        fams[fam] = {"config": str(p)}
    grid = tmp_path / "ridge_grid.yaml"
    grid.write_text("alpha: [100.0, 1.0]\n")
    fams["ridge"]["grid"] = str(grid)
    spec = {
        "name": "t",
        "category": category,
        "fold_file": str(folds_path),
        "repeats": [0],
        "mlflow_uri": None,
        "families": fams,
        "endpoints": {"n_per_engine": 3, "min_cycle": 12, "late_horizon": 20, "seed": 5},
        **extra,
    }
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


@pytest.fixture
def folds_path(train_df, tmp_path):
    p = tmp_path / "folds.json"
    p.write_text(json.dumps(cv.make_cv_folds(train_df, seeds=(11,))))
    return p


def test_run_cv_shared_endpoints_and_determinism(tmp_path, synth_cfg, folds_path):
    spec = _spec(tmp_path, synth_cfg, folds_path)
    a = cv.run_cv(spec, root=tmp_path / "a")
    b = cv.run_cv(spec, root=tmp_path / "b")
    pa, pb = (pd.read_csv(x / "predictions.csv.gz") for x in (a, b))
    pd.testing.assert_frame_equal(pa, pb)
    for _, g in pa.groupby("strategy"):
        sets = [set(map(tuple, f[["unit", "cycle"]].to_numpy())) for _, f in g.groupby("family")]
        assert all(s == sets[0] for s in sets)  # identical endpoints for every model
    outer = pa[pa.role == "outer"].groupby(["repeat", "fold"])["unit"].unique()
    flat = [u for us in outer for u in set(us)]
    assert sorted(flat) == list(range(1, 11))  # every engine held out exactly once per repeat
    with pytest.raises(ConfigError):
        cv.run_cv(spec, root=tmp_path / "a")  # experiment isolation
    s = json.loads((a / "summary.json").read_text())
    assert s["strategies"]["uniform"]["families"]["ridge"]["pooled"]["capped"]["n_engines"] == 10
    assert json.loads((a / "manifest.json").read_text())["official_test_set_used"] is False


def test_conformal_calibration_engines_disjoint_and_intervals_served(
    tmp_path, synth_cfg, folds_path, synth_dir
):
    spec = _spec(
        tmp_path,
        synth_cfg,
        folds_path,
        category="conformal",
        save_models=True,
        inner_fractions={"train": 0.6, "val": 0.15, "cal": 0.25},
    )
    exp = cv.run_cv(spec, root=tmp_path / "r")
    p = pd.read_csv(exp / "predictions.csv.gz")
    for d in sorted((exp / "models").iterdir()):
        eng = json.loads((d / "run.json").read_text())["engines"]
        sets = [set(eng[r]) for r in ("train", "val", "cal", "outer")]
        assert sum(len(s) for s in sets) == len(set().union(*sets))  # pairwise disjoint
        r, k = int(d.name.split("_")[0][1:]), int(d.name.split("_")[1][1:])
        cal_units = set(p[(p.role == "cal") & (p.repeat == r) & (p.fold == k)]["unit"])
        assert cal_units == set(eng["cal"])
        assert (d / "conformal.json").exists()
    assert (exp / "conformal_summary.json").exists()
    model_dir = exp / "models" / "r0_f0_ridge"
    rows = load_train(synth_dir).drop(columns=["rul"])
    plain = predict_engines(model_dir, rows)
    with_iv = predict_engines(model_dir, rows, levels=(0.8, 0.95))
    pd.testing.assert_frame_equal(plain, with_iv[plain.columns])  # point predictions unchanged
    assert (with_iv["lower_0.95"] <= with_iv["lower_0.8"]).all()
    assert (with_iv["upper_0.8"] <= with_iv["upper_0.95"]).all()


def test_conformal_quantile_finite_sample_coverage():
    rng = np.random.default_rng(0)
    n, level, trials = 19, 0.9, 20000
    s = rng.normal(size=(trials, n + 1))
    q = np.array([conformal_quantile(row[:n], level) for row in s])
    # exchangeable scores: P(new <= q) = ceil((n+1)level)/(n+1) = 18/20 exactly
    assert abs((s[:, n] <= q).mean() - 0.9) < 0.01
    assert conformal_quantile([1.0, 2.0], 0.9) == float("inf")  # k > n
    assert conformal_quantile(np.arange(1, 10), 0.8) == 8.0  # k = ceil(10 * 0.8) = 8


def test_mondrian_interval_uses_predicted_rul_bin():
    cal = pd.DataFrame(
        {"unit": np.repeat(np.arange(30), 3), "y_pred": np.tile([10.0, 70.0, 120.0], 30)}
    )
    cal["y_target"] = cal["y_pred"] + np.tile([1.0, 5.0, 20.0], 30)
    c = calibrate(cal, levels=(0.8,))
    lo, hi = interval([20.0, 80.0, 200.0], c, 0.8, "mondrian_pooled")
    np.testing.assert_allclose(hi - [20.0, 80.0, 200.0], [1.0, 5.0, 20.0])
    assert (lo >= 0).all()


def test_predict_without_calibration_refuses_intervals(tmp_path, synth_cfg, synth_dir):
    from engine_rul.train import train_one

    rec = train_one(synth_cfg("ridge"))
    run = Path(synth_cfg("ridge").results_dir) / rec["run_id"]
    rows = load_train(synth_dir).drop(columns=["rul"])
    out = predict_engines(run, rows)
    assert list(out.columns) == ["unit", "cycle", "y_pred"] and len(out) == 10
    with pytest.raises(ConfigError):
        predict_engines(run, rows, levels=(0.9,))


def test_experiment_isolation(tmp_path):
    d = experiment_dir(tmp_path, "cv", "x")
    (d / "a.txt").write_text("1")
    with pytest.raises(ConfigError):
        experiment_dir(tmp_path, "cv", "x")
    assert experiment_dir(tmp_path, "cv", "x", force=True) == d
    with pytest.raises(ConfigError):
        experiment_dir(tmp_path, "not_a_category", "x")
    with pytest.raises(ConfigError):
        refuse_overwrite([d / "a.txt"])
    refuse_overwrite([d / "a.txt"], force=True)
    refuse_overwrite([d / "missing.txt"])


def test_published_test_summary_is_protected(tmp_path):
    from engine_rul.evaluate import evaluate_test_final

    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "summary.json").write_text("{}")
    with pytest.raises(ConfigError):  # refuses before any data (or test label) is read
        evaluate_test_final(["ridge"], results_dir=tmp_path)


def test_frozen_config_hashes_unchanged_by_phase2_fields():
    summary = json.loads((ROOT / "results" / "final" / "summary.json").read_text())
    for fam, v in summary["families"].items():
        assert load_config(ROOT / "configs" / "final" / f"{fam}.yaml").hash() == v["config_hash"]
