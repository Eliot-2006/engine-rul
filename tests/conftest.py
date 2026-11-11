"""Synthetic C-MAPSS-shaped fixtures so unit tests run without the real data (PRD 8.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine_rul.config import from_dict
from engine_rul.data import COLUMNS
from engine_rul.split import make_split

REAL_DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "FD001" / "train_FD001.txt"
TRAIN_LIVES = [40, 45, 50, 55, 60, 70, 48, 52, 58, 65]
TEST_LENS = [25, 30, 35, 22, 40]
TEST_RUL = [20, 12, 30, 41, 8]
WINDOW = 10
INFORMATIVE = [
    c for c in COLUMNS[2:] if c not in ("op3", "s1", "s5", "s6", "s10", "s16", "s18", "s19")
]


def _engine_rows(unit: int, n_rows: int, life: int, rng) -> list[list[float]]:
    rows = []
    for t in range(1, n_rows + 1):
        health = (life - t) / life  # 1 -> 0 at failure
        vals = {c: 0.0 for c in COLUMNS}
        vals["unit"], vals["cycle"] = unit, t
        for i, c in enumerate(INFORMATIVE):
            vals[c] = 100 + 10 * i + (5 + i) * (1 - health) ** 2 + rng.normal(0, 0.3)
        vals["op3"], vals["s1"], vals["s5"] = 100.0, 518.67, 14.62
        vals["s6"] = 21.61 if rng.random() < 0.98 else 21.6  # two distinct values like FD001
        vals["s10"], vals["s16"], vals["s18"], vals["s19"] = 1.3, 0.03, 2388.0, 100.0
        rows.append([vals[c] for c in COLUMNS])
    return rows


def _write(path: Path, rows) -> None:
    # trailing double space, like the real files
    path.write_text(
        "".join(
            " ".join(f"{v:.4f}" if i > 1 else str(int(v)) for i, v in enumerate(r)) + "  \n"
            for r in rows
        )
    )


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth")
    d = root / "raw" / "FD001"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    train = [r for u, L in enumerate(TRAIN_LIVES, 1) for r in _engine_rows(u, L, L, rng)]
    test = [
        r
        for u, (n, rul) in enumerate(zip(TEST_LENS, TEST_RUL, strict=True), 1)
        for r in _engine_rows(u, n, n + rul, rng)
    ]
    _write(d / "train_FD001.txt", train)
    _write(d / "test_FD001.txt", test)
    (d / "RUL_FD001.txt").write_text("".join(f"{r} \n" for r in TEST_RUL))
    return root / "raw"


@pytest.fixture(scope="session")
def synth_split(synth_dir, tmp_path_factory) -> Path:
    from engine_rul.data import load_train

    split = make_split(load_train(synth_dir), seed=1, n_val=2, n_points=5, window=WINDOW)
    p = tmp_path_factory.mktemp("split") / "split.json"
    p.write_text(json.dumps(split))
    return p


@pytest.fixture
def synth_cfg(synth_dir, synth_split, tmp_path):
    """Factory: synthetic config for a family with small, fast settings."""

    def make(family="ridge", params=None, **over):
        d = {
            "model": {"family": family, "params": params or {}},
            "data": {
                "data_dir": str(synth_dir),
                "split_file": str(synth_split),
                "window": WINDOW,
                "r_max": 40,
                "check_invariants": False,
                "pad_short": True,
            },
            "train": {"max_epochs": 3, "patience": 2, "batch_size": 32},
            "tracking": {"mlflow_uri": None},
            "device": "cpu",
            "results_dir": str(tmp_path / "results"),
        }
        for k, v in over.items():
            d[k] = {**d.get(k, {}), **v} if isinstance(v, dict) else v
        return from_dict(d)

    return make


requires_data = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="real FD001 not downloaded (engine-rul data download)"
)
