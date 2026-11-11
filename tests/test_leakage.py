"""TEST-LEAK-*: leakage controls (LEAK-01..07)."""

import numpy as np
import pandas as pd
import pytest

from engine_rul import data as D
from engine_rul.cli import main
from engine_rul.features import make_windows, tabular_features
from engine_rul.preprocess import Preprocessor


def test_scaler_uses_train_rows_only(synth_dir):  # TEST-LEAK-01
    df = D.load_train(synth_dir)
    train, val = df[df["unit"] <= 8].copy(), df[df["unit"] > 8].copy()
    val["s2"] += 50.0  # shifted validation distribution
    pre = Preprocessor(("op1", "op2", "op3"), 40).fit(train)
    i = pre.features.index("s2")
    assert pre.mean[i] == pytest.approx(train["s2"].mean())
    assert pre.mean[i] != pytest.approx(pd.concat([train, val])["s2"].mean())


def test_selection_uses_train_rows_only(synth_dir):  # TEST-LEAK-02
    df = D.load_train(synth_dir)
    train, val = df[df["unit"] <= 8].copy(), df[df["unit"] > 8].copy()
    val["s1"] = np.linspace(0, 1, len(val))  # varies only in validation
    pre = Preprocessor(("op1", "op2", "op3"), 40).fit(train)
    assert "s1" not in pre.features
    assert "s1" in Preprocessor(("op1", "op2", "op3"), 40).fit(pd.concat([train, val])).features


def test_windows_single_unit_and_causal(synth_dir):  # TEST-LEAK-03 / LEAK-05
    df = D.load_train(synth_dir)
    pre = Preprocessor(("op1", "op2", "op3"), 40).fit(df)
    t = pre.transform(df)
    t["unit_marker"] = t["unit"].astype(np.float32)
    t["cycle_marker"] = t["cycle"].astype(np.float32)
    ws = make_windows(t, ["unit_marker", "cycle_marker", "cycle_norm"], 10)
    assert (ws.X_seq[:, :, 0] == ws.unit[:, None]).all()  # one engine per window
    assert (ws.X_seq[:, :, 1].max(axis=1) == ws.cycle).all()  # nothing after the end cycle
    assert (np.diff(ws.X_seq[:, :, 1], axis=1) == 1).all()


def test_train_and_tune_never_touch_test_files(synth_cfg, monkeypatch, tmp_path):  # TEST-LEAK-04
    def boom(*a, **k):
        raise AssertionError("test data accessed during training")

    monkeypatch.setattr(D, "load_test_rul", boom)
    monkeypatch.setattr(D, "load_test_features", boom)
    real_subset_path = D.subset_path

    def guarded(data_dir, subset, kind):
        if kind != "train":
            boom()
        return real_subset_path(data_dir, subset, kind)

    monkeypatch.setattr(D, "subset_path", guarded)
    from engine_rul.train import train_one, tune

    for fam in ("constant", "life_prior", "ridge", "hgb", "lstm"):
        train_one(synth_cfg(fam))
    grid = tmp_path / "g.yaml"
    grid.write_text("alpha: [10.0, 1.0]\n")
    tune(synth_cfg("ridge"), grid, out_dir=tmp_path / "final")


def test_test_split_requires_final_flag():  # TEST-LEAK-05
    assert main(["evaluate", "--split", "test"]) == 3


def test_features_do_not_depend_on_targets(synth_dir):  # TEST-LEAK-06
    df = D.load_train(synth_dir)
    pre = Preprocessor(("op1", "op2", "op3"), 40).fit(df)
    ws = make_windows(pre.transform(df), pre.feature_names, 10)
    X1, _ = tabular_features(ws)
    ws.y = np.random.default_rng(0).permutation(ws.y)
    ws.y_true = np.random.default_rng(1).permutation(ws.y_true)
    X2, _ = tabular_features(ws)
    np.testing.assert_array_equal(X1, X2)
