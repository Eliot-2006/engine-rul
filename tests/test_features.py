"""TEST-WIN-*: windowing and tabular feature correctness."""

import numpy as np
import pandas as pd
import pytest

from engine_rul.features import make_windows, tabular_features
from tests.conftest import requires_data


def _frame(units_lengths, n_feat=2):
    rows = []
    for u, T in units_lengths:
        for t in range(1, T + 1):
            rows.append(
                {
                    "unit": u,
                    "cycle": t,
                    **{f"f{i}": 1000 * u + t + 0.1 * i for i in range(n_feat)},
                    "cycle_norm": t / 100,
                    "rul": T - t,
                    "rul_target": min(T - t, 20),
                }
            )
    return pd.DataFrame(rows)


def test_window_count_and_content():  # TEST-WIN-01
    df = _frame([(1, 35)])
    ws = make_windows(df, ["f0", "f1", "cycle_norm"], 30)
    assert len(ws) == 6 and ws.cycle.tolist() == list(range(30, 36))
    for k in range(6):  # window k = cycles k+1..k+30, oldest first
        np.testing.assert_allclose(ws.X_seq[k, :, 0], 1000 + np.arange(k + 1, k + 31))
    assert ws.y_true.tolist() == [5, 4, 3, 2, 1, 0]


def test_short_sequence_padding():  # TEST-WIN-02
    df = _frame([(1, 20)])
    ws = make_windows(df, ["f0", "cycle_norm"], 30, end_points=[[1, 20]], pad_short=True)
    assert ws.X_seq.shape == (1, 30, 2)
    np.testing.assert_allclose(ws.X_seq[0, :11, 0], 1001)  # 10 pad rows + the real first row
    np.testing.assert_allclose(ws.X_seq[0, 10:, 0], 1000 + np.arange(1, 21))
    assert len(make_windows(df, ["f0", "cycle_norm"], 30, pad_short=True)) == 1


def test_short_sequence_without_padding_raises():  # TEST-WIN-03
    df = _frame([(1, 20)])
    with pytest.raises(ValueError, match="pad"):
        make_windows(df, ["f0", "cycle_norm"], 30)
    with pytest.raises(ValueError, match="padding"):
        make_windows(df, ["f0", "cycle_norm"], 30, end_points=[[1, 20]])


def test_windowset_contract():  # TEST-WIN-04
    ws = make_windows(_frame([(1, 40), (2, 33)]), ["f0", "f1", "cycle_norm"], 30)
    assert ws.X_seq.dtype == np.float32 and ws.X_seq.shape == (11 + 4, 30, 3)
    assert len(ws.y) == len(ws.y_true) == len(ws.unit) == len(ws.cycle) == len(ws)
    assert ws.unit.dtype == np.int32


def test_tabular_features_math():  # TEST-WIN-05
    W = 30
    a, b = 3.0, 0.5
    df = pd.DataFrame(
        {
            "unit": 1,
            "cycle": np.arange(1, W + 1),
            "lin": a + b * np.arange(W),
            "const": 7.0,
            "cycle_norm": np.arange(1, W + 1) / 100,
            "rul": 0.0,
            "rul_target": 0.0,
        }
    )
    ws = make_windows(df, ["lin", "const", "cycle_norm"], W)
    X, names = tabular_features(ws)
    f = dict(zip(names, X[0], strict=True))
    assert X.shape == (1, 2 * 4 + 1)
    assert f["lin__slope"] == pytest.approx(b, abs=1e-6)
    assert f["lin__last"] == pytest.approx(a + b * (W - 1), abs=1e-4)
    assert f["const__std"] == pytest.approx(0.0, abs=1e-9)
    assert f["const__slope"] == pytest.approx(0.0, abs=1e-9)
    assert f["cycle_norm"] == pytest.approx(0.30, abs=1e-6)


@requires_data
@pytest.mark.data
def test_real_window_count():  # TEST-WIN-06
    from engine_rul import data as D
    from engine_rul.preprocess import Preprocessor

    df = D.load_train("data/raw")
    pre = Preprocessor(("op1", "op2", "op3"), 125).fit(df)
    ws = make_windows(pre.transform(df), pre.feature_names, 30)
    assert len(ws) == 17731
    assert tabular_features(ws)[0].shape[1] == 57
