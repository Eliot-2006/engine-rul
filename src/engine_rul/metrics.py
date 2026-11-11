"""Evaluation metrics (EVAL-01, EVAL-04) and bootstrap helpers (EVAL-27/28).

Pure NumPy; error convention d = y_pred - y_true, so d > 0 is a *late* (over-optimistic)
prediction.

NASA asymmetric score (Saxena et al., PHM08):
    s_i = exp(-d_i / 13) - 1   if d_i < 0   (early)
    s_i = exp( d_i / 10) - 1   if d_i >= 0  (late, penalized more)
Note: arXiv:1810.05644 prints the late branch as exp(-d/10); that variant is NOT used here.
"""

from __future__ import annotations

import numpy as np


def _check(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"length mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty input")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("non-finite values in metric input")
    return y_true, y_pred


def rmse(y_true, y_pred) -> float:
    t, p = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true, y_pred) -> float:
    t, p = _check(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def nasa_terms(y_true, y_pred) -> np.ndarray:
    t, p = _check(y_true, y_pred)
    d = p - t
    return np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)


def nasa_score(y_true, y_pred) -> float:
    """Sum over evaluation points (literature convention)."""
    return float(nasa_terms(y_true, y_pred).sum())


def nasa_score_mean(y_true, y_pred) -> float:
    return float(nasa_terms(y_true, y_pred).mean())


def bias(y_true, y_pred) -> float:
    t, p = _check(y_true, y_pred)
    return float(np.mean(p - t))


def late_fraction(y_true, y_pred) -> float:
    t, p = _check(y_true, y_pred)
    return float(np.mean(p - t > 0))


METRICS = {
    "rmse": rmse,
    "mae": mae,
    "nasa_sum": nasa_score,
    "nasa_mean": nasa_score_mean,
    "bias": bias,
    "late_fraction": late_fraction,
}


def all_metrics(y_true, y_pred) -> dict[str, float]:
    out = {k: f(y_true, y_pred) for k, f in METRICS.items()}
    out["n"] = int(np.asarray(y_true).size)
    return out


def bootstrap_ci(y_true, y_pred, metric=rmse, n_boot: int = 2000, seed: int = 0, alpha=0.05):
    """Percentile bootstrap CI resampling evaluation points (engines) with replacement."""
    t, p = _check(y_true, y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    stats = np.array([metric(t[i], p[i]) for i in idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric=rmse, n_boot=2000, seed=0, alpha=0.05):
    """CI of metric(a) - metric(b), resampling engines jointly for both models (EVAL-28)."""
    t, a = _check(y_true, pred_a)
    _, b = _check(y_true, pred_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    diffs = np.array([metric(t[i], a[i]) - metric(t[i], b[i]) for i in idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(metric(t, a) - metric(t, b)), (float(lo), float(hi))


STAGE_BINS = (0, 25, 50, 75, 100, 125, np.inf)


def stage_labels(bins=STAGE_BINS) -> list[str]:
    return [
        f"[{int(lo)}, {'inf' if np.isinf(hi) else int(hi)})"
        for lo, hi in zip(bins[:-1], bins[1:], strict=True)
    ]


def stage_metrics(y_true, y_pred, bins=STAGE_BINS, min_n: int = 1) -> list[dict]:
    """EVAL-30: per true-RUL bin N, RMSE, MAE, bias. Bins with N < min_n get None metrics."""
    t, p = _check(y_true, y_pred)
    rows = []
    for label, lo, hi in zip(stage_labels(bins), bins[:-1], bins[1:], strict=True):
        m = (t >= lo) & (t < hi)
        n = int(m.sum())
        ok = n >= max(min_n, 1)
        rows.append(
            {
                "bin": label,
                "n": n,
                "rmse": rmse(t[m], p[m]) if ok else None,
                "mae": mae(t[m], p[m]) if ok else None,
                "bias": bias(t[m], p[m]) if ok else None,
            }
        )
    return rows
