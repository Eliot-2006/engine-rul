"""Predict RUL for engines from raw sensor rows, using any saved run directory
(``run.json`` + ``preprocessor.json`` + model file): Phase-1 runs or Phase-2 fold models.

Output columns: unit, cycle, y_pred (identical to the evaluation pipeline: ADR-13 clipping).
If the run directory has a ``conformal.json`` (Phase-2 calibrated fold models) and ``levels`` are
requested, ``lower_<level>`` / ``upper_<level>`` columns are added; point predictions never change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine_rul.config import from_dict
from engine_rul.conformal import DEFAULT_METHOD, interval
from engine_rul.errors import ConfigError
from engine_rul.features import build_windows
from engine_rul.models import load_model
from engine_rul.preprocess import Preprocessor


class Predictor:
    """Loaded once, reused for many calls (what the end-to-end benchmark times)."""

    def __init__(self, run_dir: str | Path, device: str = "cpu"):
        run_dir = Path(run_dir)
        self.cfg = from_dict(json.loads((run_dir / "run.json").read_text())["config"])
        self.pre = Preprocessor.from_dict(json.loads((run_dir / "preprocessor.json").read_text()))
        self.model = load_model(run_dir, device=device)
        cal = run_dir / "conformal.json"
        self.calibration = json.loads(cal.read_text()) if cal.exists() else None

    def predict(
        self,
        sensor_df: pd.DataFrame,
        levels=(),
        all_cycles: bool = False,
        method: str = DEFAULT_METHOD,
    ) -> pd.DataFrame:
        """Rows of the NASA sensor format (unit, cycle, op1-3, s1-s21). By default one prediction per
        engine at its last observed cycle; ``all_cycles`` gives the full trajectory."""
        if levels and self.calibration is None:
            raise ConfigError(
                "this run has no conformal.json; intervals need a calibrated Phase-2 model"
            )
        end = None
        if not all_cycles:
            last = sensor_df.groupby("unit")["cycle"].max()
            end = np.column_stack([last.index.to_numpy(), last.to_numpy()])
        ws = build_windows(
            self.pre.transform(sensor_df), self.pre.feature_names, self.cfg.data, end_points=end
        )
        pred = np.clip(self.model.predict(ws), 0, None)
        out = pd.DataFrame({"unit": ws.unit, "cycle": ws.cycle, "y_pred": pred})
        for lv in levels:
            lo, hi = interval(pred, self.calibration, lv, method)
            out[f"lower_{lv:g}"], out[f"upper_{lv:g}"] = lo, hi
        return out


def predict_engines(
    run_dir: str | Path,
    sensor_df: pd.DataFrame,
    levels=(),
    all_cycles: bool = False,
    method: str = DEFAULT_METHOD,
    device: str = "cpu",
) -> pd.DataFrame:
    return Predictor(run_dir, device).predict(sensor_df, levels, all_cycles, method)
