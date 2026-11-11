"""B0 constant-mean and B1 cycle-only life-prior baselines (PRD 4.4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine_rul.features import WindowSet


class ConstantMean:
    name = "constant"
    input_kind = "tabular"

    def __init__(self, params=None, seed: int = 0):
        self.value: float | None = None

    def fit(self, train: WindowSet, val=None, cfg=None, workdir=None, tracker=None) -> dict:
        self.value = float(np.mean(train.y))
        return {}

    def predict(self, ws: WindowSet) -> np.ndarray:
        return np.full(len(ws), self.value, dtype=np.float64)

    def n_params(self) -> int:
        return 1

    def save(self, d: str | Path) -> Path:
        p = Path(d) / "model.json"
        p.write_text(json.dumps({"family": self.name, "value": self.value}))
        return p

    @classmethod
    def load(cls, d: str | Path) -> ConstantMean:
        m = cls()
        m.value = json.loads((Path(d) / "model.json").read_text())["value"]
        return m


class LifePrior:
    """pred = min(max(L_bar - t, 0), R_max), L_bar = mean lifetime of training engines.

    Lifetimes are recovered from the training windows as cycle + uncapped RUL."""

    name = "life_prior"
    input_kind = "tabular"

    def __init__(self, params=None, seed: int = 0):
        self.mean_life: float | None = None
        self.r_max: float | None = None

    def fit(self, train: WindowSet, val=None, cfg=None, workdir=None, tracker=None) -> dict:
        life = {}
        for u, c, r in zip(train.unit, train.cycle, train.y_true, strict=True):
            life[int(u)] = float(c + r)
        self.mean_life = float(np.mean(list(life.values())))
        self.r_max = cfg.data.r_max if cfg is not None else None
        return {}

    def predict(self, ws: WindowSet) -> np.ndarray:
        pred = np.maximum(self.mean_life - ws.cycle.astype(np.float64), 0.0)
        return pred if self.r_max is None else np.minimum(pred, self.r_max)

    def n_params(self) -> int:
        return 1

    def save(self, d: str | Path) -> Path:
        p = Path(d) / "model.json"
        p.write_text(
            json.dumps({"family": self.name, "mean_life": self.mean_life, "r_max": self.r_max})
        )
        return p

    @classmethod
    def load(cls, d: str | Path) -> LifePrior:
        m = cls()
        j = json.loads((Path(d) / "model.json").read_text())
        m.mean_life, m.r_max = j["mean_life"], j["r_max"]
        return m
