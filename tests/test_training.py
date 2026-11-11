"""TEST-TRAIN-*: checkpointing, early stopping, artifact round trips, run records."""

import json

import numpy as np
import pytest

from engine_rul.models import load_model
from engine_rul.models.lstm import LSTMModel
from engine_rul.train import prepare, train_one


def test_lstm_checkpoint_roundtrip(synth_cfg, tmp_path):  # TEST-TRAIN-01
    cfg = synth_cfg("lstm", train={"max_epochs": 3, "patience": 5})
    prep = prepare(cfg)
    m = LSTMModel(cfg.model.params, 0)
    m.fit(prep.train, prep.val_dense, cfg, workdir=tmp_path)
    m.save(tmp_path)
    m2 = LSTMModel.load(tmp_path)  # torch.load(weights_only=True)
    np.testing.assert_allclose(m.predict(prep.val_points), m2.predict(prep.val_points), atol=1e-6)
    assert (tmp_path / "best.pt").exists()


def test_early_stopping_and_restore(synth_cfg, monkeypatch):  # TEST-TRAIN-02
    import engine_rul.models.lstm as L

    seq = iter([10.0, 8.0, 9.0, 9.5, 7.0, 7.0])  # best at epoch 2; patience 2 -> stop at epoch 4
    monkeypatch.setattr(L, "rmse", lambda *a: next(seq))
    cfg = synth_cfg("lstm", train={"max_epochs": 6, "patience": 2, "min_delta": 0.01})
    prep = prepare(cfg)
    m = LSTMModel(cfg.model.params, 0)
    states = []
    orig = m.predict

    def spy(ws, *a, **k):
        states.append({k2: v.clone() for k2, v in m.net.state_dict().items()})
        return orig(ws, *a, **k)

    monkeypatch.setattr(m, "predict", spy)
    info = m.fit(prep.train, prep.val_dense, cfg)
    assert info["epochs_run"] == 4 and info["best_epoch"] == 2
    restored = m.net.state_dict()
    assert all((restored[k] == states[1][k]).all() for k in restored)  # weights of epoch 2


@pytest.mark.parametrize("family", ["constant", "life_prior", "ridge", "hgb", "lstm"])
def test_artifact_roundtrip(synth_cfg, family):  # TEST-TRAIN-03
    cfg = synth_cfg(family)
    rec = train_one(cfg)
    run_dir = cfg.results_dir + "/" + rec["run_id"]
    prep = prepare(cfg)
    loaded = load_model(run_dir)
    saved = np.loadtxt(f"{run_dir}/predictions_val.csv", delimiter=",", skiprows=1)[:, -1]
    np.testing.assert_allclose(np.clip(loaded.predict(prep.val_points), 0, None), saved, atol=1e-5)


REQUIRED = {
    "run_id",
    "family",
    "stage",
    "status",
    "config",
    "config_hash",
    "seed",
    "git",
    "data_sha256",
    "split_sha256",
    "preprocessor_sha256",
    "env",
    "metrics",
    "train_time_s",
    "n_params",
}


def test_run_record_and_mlflow_failure(synth_cfg, monkeypatch, tmp_path):  # TEST-TRAIN-04
    import mlflow

    def broken(*a, **k):
        raise RuntimeError("tracking server down")

    monkeypatch.setattr(mlflow, "log_metrics", broken)
    cfg = synth_cfg("ridge", tracking={"mlflow_uri": f"sqlite:///{tmp_path}/m.db"})
    rec = train_one(cfg)
    on_disk = json.loads(open(f"{cfg.results_dir}/{rec['run_id']}/run.json").read())
    assert REQUIRED <= set(on_disk) and on_disk["status"] == "finished"
    assert on_disk["git"]["commit"]


def test_failed_run_is_marked(synth_cfg, monkeypatch):
    import engine_rul.models.classical as C

    monkeypatch.setattr(
        C.RidgeModel, "fit", lambda *a, **k: (_ for _ in ()).throw(FloatingPointError("nan"))
    )
    cfg = synth_cfg("ridge")
    with pytest.raises(FloatingPointError):
        train_one(cfg)
    from pathlib import Path

    recs = [json.loads(p.read_text()) for p in Path(cfg.results_dir).glob("*/run.json")]
    assert any(r["status"] == "failed" for r in recs)
