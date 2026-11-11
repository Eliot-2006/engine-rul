"""Ridge and HistGradientBoosting models on window-derived tabular features (PRD 4.4 M1/M2).

Artifacts are joblib pickles: only load files this project produced (ART-02)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engine_rul.features import TAB_STATS, WindowSet, tabular_features


class _SklearnModel:
    input_kind = "tabular"

    def __init__(self, params=None, seed: int = 0):
        self.params = dict(params or {})
        # Phase 2 ablation knob; not an estimator argument.
        self.stats = tuple(self.params.get("feature_stats", TAB_STATS))
        self.seed = seed
        self.model = None
        self.feature_names: list[str] | None = None

    def _make(self):
        raise NotImplementedError

    def _estimator_params(self) -> dict:
        return {k: v for k, v in self.params.items() if k != "feature_stats"}

    def fit(self, train: WindowSet, val=None, cfg=None, workdir=None, tracker=None) -> dict:
        X, self.feature_names = tabular_features(train, self.stats)
        self.model = self._make().fit(X, train.y)
        return {}

    def features(self, ws: WindowSet) -> np.ndarray:
        X, names = tabular_features(ws, self.stats)
        assert names == self.feature_names
        return X

    def predict(self, ws: WindowSet) -> np.ndarray:
        return self.model.predict(self.features(ws)).astype(np.float64)

    def save(self, d: str | Path) -> Path:
        p = Path(d) / "model.joblib"
        joblib.dump(
            {
                "family": self.name,
                "model": self.model,
                "feature_names": self.feature_names,
                "params": self.params,
                "seed": self.seed,
            },
            p,
        )
        return p

    @classmethod
    def load(cls, d: str | Path):
        obj = joblib.load(Path(d) / "model.joblib")
        m = cls(obj["params"], obj["seed"])
        m.model, m.feature_names = obj["model"], obj["feature_names"]
        return m


class RidgeModel(_SklearnModel):
    name = "ridge"

    def _make(self):
        return make_pipeline(StandardScaler(), Ridge(alpha=float(self.params.get("alpha", 1.0))))

    def n_params(self) -> int:
        r = self.model[-1]
        return int(r.coef_.size + 1)

    def coefficients(self) -> dict[str, float]:
        """Coefficients on standardized features."""
        return dict(zip(self.feature_names, self.model[-1].coef_.tolist(), strict=True))


class HGBModel(_SklearnModel):
    name = "hgb"

    def _make(self):
        # LEAK-10: internal early stopping would split *rows* (same engine on both sides).
        return HistGradientBoostingRegressor(
            loss="squared_error",
            early_stopping=False,
            random_state=self.seed,
            **self._estimator_params(),
        )

    def n_params(self) -> int:
        """Total tree nodes across all boosting iterations (BENCH-04)."""
        return int(sum(len(p.nodes) for it in self.model._predictors for p in it))
