"""TEST-MET-*: hand-verified metric values, validation, bootstrap, stage bins."""

import math

import numpy as np
import pytest

from engine_rul import metrics as M


def test_hand_computed_values():  # TEST-MET-01
    assert M.nasa_terms([0], [0])[0] == 0.0
    assert M.nasa_terms([13], [0])[0] == pytest.approx(math.e - 1, abs=1e-9)  # d = -13 (early)
    assert M.nasa_terms([0], [10])[0] == pytest.approx(math.e - 1, abs=1e-9)  # d = +10 (late)
    assert M.nasa_terms([26], [0])[0] == pytest.approx(math.e**2 - 1, abs=1e-9)
    assert M.nasa_terms([0], [20])[0] == pytest.approx(math.e**2 - 1, abs=1e-9)
    y, p = np.array([10.0, 10.0]), np.array([13.0, 6.0])  # errors [3, -4]
    assert M.rmse(y, p) == pytest.approx(math.sqrt(12.5), abs=1e-9)
    assert M.mae(y, p) == pytest.approx(3.5, abs=1e-9)
    assert M.bias(y, p) == pytest.approx(-0.5)
    assert M.late_fraction(y, p) == pytest.approx(0.5)
    s = (math.exp(3 / 10) - 1) + (math.exp(4 / 13) - 1)
    assert M.nasa_score(y, p) == pytest.approx(s, abs=1e-9)
    assert M.nasa_score_mean(y, p) == pytest.approx(s / 2, abs=1e-9)


def test_asymmetry():  # TEST-MET-02
    for k in (1, 5, 20, 50):
        assert M.nasa_score([0], [k]) > M.nasa_score([k], [0])


@pytest.mark.parametrize(
    "a, b", [([1, 2], [1]), ([], []), ([1, np.nan], [1, 2]), ([1, 2], [1, np.inf])]
)
def test_input_validation(a, b):  # TEST-MET-03
    with pytest.raises(ValueError):
        M.rmse(a, b)


def test_bootstrap():  # TEST-MET-04
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 100, 100)
    p = y + rng.normal(0, 10, 100)
    ci1 = M.bootstrap_ci(y, p, M.rmse, n_boot=500, seed=0)
    assert ci1 == M.bootstrap_ci(y, p, M.rmse, n_boot=500, seed=0)
    assert ci1[0] < M.rmse(y, p) < ci1[1]
    d, (lo, hi) = M.paired_bootstrap_diff(y, p, p, n_boot=200)
    assert d == 0 and lo == 0 and hi == 0
    d, (lo, hi) = M.paired_bootstrap_diff(y, y + 30, p, n_boot=500)
    assert d > 0 and lo > 0


def test_stage_bins():  # TEST-MET-05
    y = np.array([0, 24, 25, 60, 99, 100, 124, 125, 140])
    rows = M.stage_metrics(y, y + 1.0, min_n=1)
    assert [r["n"] for r in rows] == [2, 1, 1, 1, 2, 2]
    assert rows[-1]["bin"] == "[125, inf)"
    assert all(r["rmse"] == pytest.approx(1.0) for r in rows)
    sparse = M.stage_metrics(y, y, min_n=2)
    assert sparse[1]["rmse"] is None and sparse[0]["rmse"] == 0.0
