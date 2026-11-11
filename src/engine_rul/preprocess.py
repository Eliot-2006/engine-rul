"""Leakage-safe preprocessing (PREP-01..05, LEAK-01/02).

``Preprocessor.fit`` must only ever see train-split rows; everything it learns (kept features,
means, stds, cycle scale) is frozen and applied unchanged to validation and test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine_rul.data import COLUMNS, cap_rul
from engine_rul.errors import ConfigError, DataValidationError

CANDIDATES = COLUMNS[2:]  # op1..op3, s1..s21


class Preprocessor:
    def __init__(self, drop_columns=(), r_max: float | None = 125, expected_features=None):
        self.drop_columns = tuple(drop_columns)
        self.r_max = r_max
        self.expected_features = tuple(expected_features) if expected_features else None
        self.features: list[str] | None = None
        self.dropped_constant: list[str] | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.cycle_scale: float | None = None

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        cands = [c for c in CANDIDATES if c not in self.drop_columns]
        nunique = df[cands].nunique()
        self.dropped_constant = [c for c in cands if nunique[c] <= 2]  # PREP-01
        self.features = [c for c in cands if c not in self.dropped_constant]
        if self.expected_features is not None and tuple(self.features) != self.expected_features:
            raise ConfigError(
                f"sensor selection {self.features} != expected {list(self.expected_features)}"
            )
        x = df[self.features].to_numpy(np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0, ddof=0)  # PREP-02
        if (self.std < 1e-8).any():
            raise DataValidationError("zero-variance feature survived selection")
        self.cycle_scale = float(df.groupby("unit")["cycle"].max().max())  # PREP-03
        return self

    @property
    def feature_names(self) -> list[str]:
        """Model input channels in order: normalized sensors then cycle_norm."""
        return [*self.features, "cycle_norm"]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.features is None:
            raise RuntimeError("Preprocessor not fitted")
        out = df[["unit", "cycle"]].copy()
        z = (df[self.features].to_numpy(np.float64) - self.mean) / self.std
        out[self.features] = z.astype(np.float32)
        out["cycle_norm"] = (df["cycle"] / self.cycle_scale).astype(np.float32)
        if "rul" in df:
            out["rul"] = df["rul"].to_numpy(np.float64)
            out["rul_target"] = cap_rul(out["rul"].to_numpy(), self.r_max)  # PREP-04 / DATA-24
        return out

    def to_dict(self) -> dict:
        return {
            "drop_columns": list(self.drop_columns),
            "r_max": self.r_max,
            "features": self.features,
            "dropped_constant": self.dropped_constant,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "cycle_scale": self.cycle_scale,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Preprocessor:
        p = cls(d["drop_columns"], d["r_max"])
        p.features = list(d["features"])
        p.dropped_constant = list(d["dropped_constant"])
        p.mean = np.asarray(d["mean"], dtype=np.float64)
        p.std = np.asarray(d["std"], dtype=np.float64)
        p.cycle_scale = float(d["cycle_scale"])
        return p
