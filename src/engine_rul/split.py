"""Engine-level train/validation split and test-like validation evaluation points (DATA-30..33)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul.errors import ConfigError


@dataclass(frozen=True)
class Split:
    train_units: np.ndarray
    val_units: np.ndarray
    val_eval_points: np.ndarray  # int[N, 2] of (unit, cycle)


def make_split(
    train_df: pd.DataFrame,
    seed: int,
    n_val: int | None = None,
    n_points: int = 10,
    window: int = 30,
    subset: str = "FD001",
    train_file_sha256: str | None = None,
) -> dict:
    """Lifetime-stratified engine-level split (DATA-30/32) plus truncation points (DATA-33).

    Engines are sorted by lifetime and cut into ``n_val`` contiguous bins; one engine per bin is
    drawn for validation. For each validation engine, ``n_points`` distinct truncation cycles are
    drawn uniformly from [window, T_u]."""
    rng = np.random.default_rng(seed)
    life = train_df.groupby("unit")["cycle"].max().sort_index()
    n_val = n_val if n_val is not None else max(1, round(0.2 * len(life)))
    order = life.sort_values(kind="stable").index.to_numpy()  # stable: ties keep unit order
    val = sorted(int(rng.choice(b)) for b in np.array_split(order, n_val))
    train = sorted(int(u) for u in life.index if u not in set(val))
    points = []
    for u in val:
        T = int(life[u])
        if T < window:
            raise ConfigError(f"validation engine {u} (life {T}) shorter than window {window}")
        cand = np.arange(window, T + 1)
        cyc = np.sort(rng.choice(cand, size=min(n_points, len(cand)), replace=False))
        points += [[u, int(c)] for c in cyc]
    return {
        "subset": subset,
        "seed": seed,
        "window": window,
        "n_points_per_engine": n_points,
        "train_units": train,
        "val_units": val,
        "val_eval_points": points,
        "sha256_of_train_file": train_file_sha256,
    }


def save_split(split: dict, path: str | Path, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise ConfigError(f"{path} exists; pass --force to overwrite the committed split")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1) + "\n")


def load_split(path: str | Path) -> Split:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"split file {path} not found; run `engine-rul split make`")
    d = json.loads(path.read_text())
    return Split(
        train_units=np.asarray(d["train_units"], dtype=np.int32),
        val_units=np.asarray(d["val_units"], dtype=np.int32),
        val_eval_points=np.asarray(d["val_eval_points"], dtype=np.int32).reshape(-1, 2),
    )
