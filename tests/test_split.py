"""TEST-SPLIT-*: engine-level split integrity, determinism, evaluation points."""

import json

import numpy as np
import pytest

from engine_rul import data as D
from engine_rul.errors import ConfigError
from engine_rul.split import load_split, make_split, save_split
from tests.conftest import WINDOW, requires_data


def test_split_disjoint_and_complete(synth_dir):  # TEST-SPLIT-01
    df = D.load_train(synth_dir)
    s = make_split(df, seed=3, window=WINDOW)
    units = set(df["unit"].unique().tolist())
    assert not set(s["train_units"]) & set(s["val_units"])
    assert set(s["train_units"]) | set(s["val_units"]) == units
    assert len(s["val_units"]) == round(0.2 * len(units))


def test_split_deterministic(synth_dir):  # TEST-SPLIT-02
    df = D.load_train(synth_dir)
    assert make_split(df, seed=7, window=WINDOW) == make_split(df, seed=7, window=WINDOW)
    assert make_split(df, seed=7, window=WINDOW) != make_split(df, seed=8, window=WINDOW)


def test_eval_points(synth_dir):  # TEST-SPLIT-03
    df = D.load_train(synth_dir)
    s = make_split(df, seed=3, n_points=10, window=WINDOW)
    life = df.groupby("unit")["cycle"].max()
    pts = np.asarray(s["val_eval_points"])
    assert set(pts[:, 0]) <= set(s["val_units"])
    for u in s["val_units"]:
        cyc = pts[pts[:, 0] == u, 1]
        assert len(cyc) == 10 and len(set(cyc)) == 10
        assert cyc.min() >= WINDOW and cyc.max() <= life[u]


def test_save_refuses_overwrite(tmp_path, synth_dir):
    s = make_split(D.load_train(synth_dir), seed=1, window=WINDOW)
    p = tmp_path / "s.json"
    save_split(s, p)
    with pytest.raises(ConfigError, match="--force"):
        save_split(s, p)
    save_split(s, p, force=True)
    assert load_split(p).val_eval_points.shape[1] == 2


@requires_data
@pytest.mark.data
def test_committed_split_matches_regenerated():  # TEST-SPLIT-02 (real data)
    committed = json.load(open("splits/fd001_split.json"))
    regen = make_split(
        D.load_train("data/raw"),
        seed=committed["seed"],
        window=30,
        train_file_sha256=committed["sha256_of_train_file"],
    )
    assert regen == committed
    assert len(committed["train_units"]) == 80 and len(committed["val_units"]) == 20
    assert len(committed["val_eval_points"]) == 200
