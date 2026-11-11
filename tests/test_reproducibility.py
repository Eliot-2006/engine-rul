"""TEST-REPRO-*: deterministic retraining and config hashing."""

import numpy as np
import pytest

from engine_rul.config import from_dict, load_config
from engine_rul.models import build_model
from engine_rul.train import prepare
from engine_rul.utils import set_seed


@pytest.mark.parametrize("family, tol", [("ridge", 1e-6), ("hgb", 1e-6), ("lstm", 1e-3)])
def test_same_seed_same_predictions(synth_cfg, family, tol):  # TEST-REPRO-01 (CPU)
    cfg = synth_cfg(family)
    prep = prepare(cfg)
    preds = []
    for _ in range(2):
        set_seed(cfg.seed)
        m = build_model(cfg)
        m.fit(prep.train, prep.val_dense, cfg)
        preds.append(m.predict(prep.val_points))
    np.testing.assert_allclose(preds[0], preds[1], atol=tol)


def test_config_hash(tmp_path):  # TEST-REPRO-02
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("model: {family: ridge, params: {alpha: 1.0}}\ndata: {window: 30, r_max: 125}\n")
    b.write_text("data: {r_max: 125, window: 30}\nmodel: {params: {alpha: 1.0}, family: ridge}\n")
    ca, cb = load_config(a), load_config(b)
    assert ca.hash() == cb.hash()
    assert ca.replace(seed=5).hash() == ca.hash()  # seed excluded: multi-seed runs share a hash
    assert ca.replace(model={"params": {"alpha": 2.0}}).hash() != ca.hash()
    assert ca.replace(data={"window": 20}).hash() != ca.hash()


def test_config_validation():
    from engine_rul.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown keys"):
        from_dict({"modle": {}})
    with pytest.raises(ConfigError, match="family"):
        from_dict({"model": {"family": "xgboost"}})
    with pytest.raises(ConfigError, match="window"):
        from_dict({"data": {"window": 0}})
    with pytest.raises(ConfigError, match="r_max"):
        from_dict({"data": {"r_max": -1}})


def test_repo_configs_load():
    for name in ("baseline_constant", "baseline_life_prior", "ridge", "hgb", "lstm"):
        load_config(f"configs/{name}.yaml")
