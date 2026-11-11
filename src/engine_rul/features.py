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


def build_windows(
    df: pd.DataFrame, feature_cols: list[str], data_cfg, end_points=None
) -> WindowSet:
    """make_windows + input-channel selection driven by a DataConfig (window, pad_short,
    use_sensors, use_cycle). Every harness (training, CV, evaluation, benchmarks) goes through
    this so that train-time and inference-time inputs always match."""
    ws = make_windows(
        df, feature_cols, data_cfg.window, end_points=end_points, pad_short=data_cfg.pad_short
    )
    return select_channels(ws, data_cfg.use_sensors, data_cfg.use_cycle)


def last_cycle_points(df: pd.DataFrame) -> np.ndarray:
    """One (unit, last observed cycle) pair per engine - the official test protocol (EVAL-02)."""
    last = df.groupby("unit")["cycle"].max()
    return np.column_stack([last.index.to_numpy(), last.to_numpy()]).astype(np.int64)


TAB_STATS = ("last", "mean", "std", "slope")
CYCLE = "cycle_norm"


def select_channels(ws: WindowSet, use_sensors: bool = True, use_cycle: bool = True) -> WindowSet:
    """Phase 2 input ablations: keep sensor channels and/or the cycle_norm channel."""
    if use_sensors and use_cycle:
        return ws
    keep = [
        i
        for i, n in enumerate(ws.feature_names)
        if (n == CYCLE and use_cycle) or (n != CYCLE and use_sensors)
    ]
    return WindowSet(
        ws.X_seq[:, :, keep],
        ws.y,
        ws.y_true,
        ws.unit,
        ws.cycle,
        [ws.feature_names[i] for i in keep],
    )


def tabular_features(ws: WindowSet, stats=TAB_STATS) -> tuple[np.ndarray, list[str]]:
    """FEAT-20: per sensor channel the requested window statistics (default last, mean, std,
    slope, in that per-sensor order), plus cycle_norm at the end cycle if that channel exists."""
    unknown = set(stats) - set(TAB_STATS)
    if unknown:
        raise ValueError(
            f"unknown tabular stats {sorted(unknown)}; expected a subset of {TAB_STATS}"
        )
    has_cycle = bool(ws.feature_names) and ws.feature_names[-1] == CYCLE
    sensors = ws.feature_names[:-1] if has_cycle else ws.feature_names
    S = ws.X_seq[:, :, : len(sensors)].astype(np.float64)
    W = S.shape[1]
    t = np.arange(W, dtype=np.float64) - (W - 1) / 2
    denom = (t**2).sum()

    def slope():
        if not denom:
            return np.zeros((len(S), S.shape[2]))
        return np.einsum("nwf,w->nf", S - S.mean(axis=1, keepdims=True), t) / denom

    make = {
        "last": lambda: S[:, -1, :],
        "mean": lambda: S.mean(axis=1),
        "std": lambda: S.std(axis=1),
        "slope": slope,
    }
    cols = [np.stack([make[st]() for st in stats], axis=2).reshape(len(S), -1)] if sensors else []
    names = [f"{s}__{st}" for s in sensors for st in stats]
    if has_cycle:
        cols.append(ws.X_seq[:, -1, -1].astype(np.float64)[:, None])
        names.append(CYCLE)
    return np.column_stack(cols), names
