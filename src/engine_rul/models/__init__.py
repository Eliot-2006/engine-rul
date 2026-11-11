"""Model registry. All families share the RULModel protocol (PRD 4.4):

    name, input_kind, fit(train, val, cfg, workdir, tracker) -> dict, predict(ws) -> ndarray,
    save(dir), load(dir) (classmethod), n_params() -> int

Models never clip predictions; the harness applies clip(pred, 0, None) uniformly (ADR-13).
"""

from __future__ import annotations

import json
from pathlib import Path


def get_model_class(family: str):
    from engine_rul.models.baselines import ConstantMean, LifePrior
    from engine_rul.models.classical import HGBModel, RidgeModel
    from engine_rul.models.lstm import LSTMModel

    return {
        "constant": ConstantMean,
        "life_prior": LifePrior,
        "ridge": RidgeModel,
        "hgb": HGBModel,
        "lstm": LSTMModel,
    }[family]


def build_model(cfg):
    return get_model_class(cfg.model.family)(params=dict(cfg.model.params), seed=cfg.seed)


def load_model(run_dir: str | Path, device: str = "cpu"):
    """Load any family from a run directory (family read from run.json)."""
    run_dir = Path(run_dir)
    family = json.loads((run_dir / "run.json").read_text())["family"]
    cls = get_model_class(family)
    return cls.load(run_dir, device=device) if family == "lstm" else cls.load(run_dir)
