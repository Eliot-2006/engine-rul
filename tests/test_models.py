"""TEST-MOD-*: model I/O shapes, LSTM architecture, baselines, learning capability."""

import numpy as np
import pytest
import torch

from engine_rul.models.baselines import ConstantMean, LifePrior
from engine_rul.models.lstm import LSTMRegressor
from engine_rul.train import prepare


@pytest.mark.parametrize("family", ["constant", "life_prior", "ridge", "hgb", "lstm"])
def test_predict_shape(synth_cfg, family):  # TEST-MOD-01
    from engine_rul.models import build_model

    cfg = synth_cfg(family)
    prep = prepare(cfg)
    m = build_model(cfg)
    m.fit(prep.train, prep.val_dense, cfg)
    p = m.predict(prep.val_points)
    assert p.shape == (len(prep.val_points),) and p.dtype == np.float64 and np.isfinite(p).all()


@pytest.mark.parametrize("H, L, F", [(64, 2, 15), (32, 1, 15), (16, 3, 4)])
def test_lstm_forward_and_param_count(H, L, F):  # TEST-MOD-02
    net = LSTMRegressor(F, H, L, 0.1)
    assert net(torch.zeros(5, 30, F)).shape == (5,)
    lstm = sum(4 * (H * ((F if layer == 0 else H) + H) + 2 * H) for layer in range(L))
    h2 = H // 2
    head = H * h2 + h2 + h2 + 1
    assert sum(p.numel() for p in net.parameters()) == lstm + head


def test_baselines_closed_form(synth_cfg):  # TEST-MOD-03
    cfg = synth_cfg("life_prior")
    prep = prepare(cfg)
    c = ConstantMean()
    c.fit(prep.train)
    assert c.predict(prep.val_points) == pytest.approx(
        np.full(len(prep.val_points), prep.train.y.mean())
    )
    lp = LifePrior()
    lp.fit(prep.train, cfg=cfg)
    lives = {}
    for u, cyc, r in zip(prep.train.unit, prep.train.cycle, prep.train.y_true, strict=True):
        lives[u] = cyc + r
    L = np.mean(list(lives.values()))
    expected = np.minimum(np.maximum(L - prep.val_points.cycle, 0), cfg.data.r_max)
    np.testing.assert_allclose(lp.predict(prep.val_points), expected)


def test_lstm_can_overfit_small_batch():  # TEST-MOD-04
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = torch.tensor(rng.normal(size=(64, 10, 4)), dtype=torch.float32)
    y = X[:, -1, 0] * 0.5 + X[:, :, 1].mean(1)
    net = LSTMRegressor(4, 32, 1, 0.0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] <= 0.1 * losses[0]
