"""Sliding windows and window-derived tabular features (FEAT-10..21, SEQ-01..04, LEAK-03/05).

Every model - tabular or sequence - consumes the same W-cycle window ending at the evaluation
cycle; tabular features are computed from ``X_seq`` only (FEAT-21).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class WindowSet:
    X_seq: np.ndarray  # float32[N, W, F], oldest -> newest
    y: np.ndarray  # float32[N] training target (capped if r_max set); NaN if unknown
    y_true: np.ndarray  # float32[N] uncapped RUL; NaN if unknown
    unit: np.ndarray  # int32[N]
    cycle: np.ndarray  # int32[N] window end cycle
    feature_names: list[str]

    def __len__(self) -> int:
        return len(self.y)


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    end_points=None,
    pad_short: bool = False,
) -> WindowSet:
    """Build windows per engine; never crosses engine boundaries.

    ``df`` is a transformed frame (Preprocessor.transform). ``end_points`` is an int[N, 2] of
    (unit, cycle) pairs; if None, every end cycle t >= window is used (SEQ-02). Windows ending
    before cycle ``window`` are left edge-padded with the first observed row if ``pad_short``
    (SEQ-03), otherwise a ValueError is raised.
    """
    W = window
    groups = {int(u): g for u, g in df.groupby("unit", sort=True)}
    if end_points is None:
        ends = []
        for u, g in groups.items():
            T = int(g["cycle"].max())
            if T >= W:
                ends += [(u, c) for c in range(W, T + 1)]
            elif pad_short:
                ends.append((u, T))
            else:
                raise ValueError(f"unit {u} has {T} cycles < window {W} and pad_short=False")
        end_points = np.asarray(ends, dtype=np.int64).reshape(-1, 2)
    end_points = np.asarray(end_points, dtype=np.int64).reshape(-1, 2)

    has_y = "rul" in df
    X = np.empty((len(end_points), W, len(feature_cols)), dtype=np.float32)
    y = np.full(len(end_points), np.nan, dtype=np.float32)
    y_true = np.full(len(end_points), np.nan, dtype=np.float32)
    for u in np.unique(end_points[:, 0]):
        if u not in groups:
            raise ValueError(f"unit {u} not present in data")
        g = groups[int(u)]
        idx = np.where(end_points[:, 0] == u)[0]
        cyc = end_points[idx, 1]
        T = len(g)
        if cyc.min() < 1 or cyc.max() > T:
            raise ValueError(f"unit {u}: end cycle outside 1..{T}")
        if cyc.min() < W and not pad_short:
            raise ValueError(
                f"unit {u}: window ending at cycle {cyc.min()} needs padding (pad_short=False)"
            )
        arr = g[feature_cols].to_numpy(np.float32)  # rows are cycles 1..T (validated contiguous)
        padded = np.concatenate([np.repeat(arr[:1], W - 1, axis=0), arr])
        views = sliding_window_view(padded, W, axis=0)  # [T, F, W]; view i ends at cycle i+1
        X[idx] = views[cyc - 1].transpose(0, 2, 1)
        if has_y:
            y[idx] = g["rul_target"].to_numpy()[cyc - 1]
            y_true[idx] = g["rul"].to_numpy()[cyc - 1]
    return WindowSet(
        X_seq=X,
        y=y,
        y_true=y_true,
        unit=end_points[:, 0].astype(np.int32),
        cycle=end_points[:, 1].astype(np.int32),
        feature_names=list(feature_cols),
    )


def last_cycle_points(df: pd.DataFrame) -> np.ndarray:
    """One (unit, last observed cycle) pair per engine - the official test protocol (EVAL-02)."""
    last = df.groupby("unit")["cycle"].max()
    return np.column_stack([last.index.to_numpy(), last.to_numpy()]).astype(np.int64)


TAB_STATS = ("last", "mean", "std", "slope")


def tabular_features(ws: WindowSet) -> tuple[np.ndarray, list[str]]:
    """FEAT-20: last/mean/std/slope per sensor channel + cycle_norm at the end cycle.

    Assumes the final channel is ``cycle_norm`` (Preprocessor.feature_names)."""
    assert ws.feature_names[-1] == "cycle_norm"
    S = ws.X_seq[:, :, :-1].astype(np.float64)
    W = S.shape[1]
    t = np.arange(W, dtype=np.float64) - (W - 1) / 2
    denom = (t**2).sum()
    slope = (
        np.einsum("nwf,w->nf", S - S.mean(axis=1, keepdims=True), t) / denom
        if denom
        else np.zeros(S.shape[::2])
    )
    blocks = [S[:, -1, :], S.mean(axis=1), S.std(axis=1), slope]
    feats = np.stack(blocks, axis=2).reshape(len(S), -1)  # per sensor: last, mean, std, slope
    X = np.column_stack([feats, ws.X_seq[:, -1, -1].astype(np.float64)])
    names = [f"{s}__{st}" for s in ws.feature_names[:-1] for st in TAB_STATS] + ["cycle_norm"]
    return X, names
