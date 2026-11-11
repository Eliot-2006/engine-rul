"""PREP-01..05: sensor selection, normalization, persistence."""

import numpy as np
import pytest

from engine_rul import data as D
from engine_rul.errors import ConfigError
from engine_rul.preprocess import Preprocessor
from tests.conftest import requires_data

FD001_SENSORS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]


def test_selection_and_scaling(synth_dir):
    df = D.load_train(synth_dir)
    pre = Preprocessor(("op1", "op2", "op3"), 40).fit(df)
    assert pre.features == FD001_SENSORS
    assert pre.dropped_constant == ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
    out = pre.transform(df)
    z = out[pre.features].to_numpy(np.float64)
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-4)
    assert out["cycle_norm"].max() == pytest.approx(1.0)
    assert (out["rul_target"] <= 40).all() and out["rul"].max() > 40


def test_expected_features_assertion(synth_dir):
    df = D.load_train(synth_dir)
    with pytest.raises(ConfigError, match="sensor selection"):
        Preprocessor(("op1", "op2", "op3"), 40, expected_features=["s2"]).fit(df)


def test_roundtrip(synth_dir):
    df = D.load_train(synth_dir)
    pre = Preprocessor(("op1", "op2", "op3"), 40).fit(df)
    pre2 = Preprocessor.from_dict(pre.to_dict())
    assert pre2.to_dict() == pre.to_dict()
    assert pre.transform(df).equals(pre2.transform(df))


@requires_data
@pytest.mark.data
def test_real_fd001_selection_on_train_split():
    from engine_rul.split import load_split

    df = D.load_train("data/raw")
    s = load_split("splits/fd001_split.json")
    pre = Preprocessor(("op1", "op2", "op3"), 125).fit(df[df["unit"].isin(s.train_units)])
    assert pre.features == FD001_SENSORS
