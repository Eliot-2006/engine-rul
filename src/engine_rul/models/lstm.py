"""Primary sequence model: stacked LSTM regressor (PRD 4.4 M3, ADR-08).

Training: Adam + MSE on targets scaled by S = R_max (or 125 if uncapped), seeded shuffling,
gradient clipping, per-epoch validation RMSE on the dense validation windows (capped truth),
early stopping with best-weight restore, ``best.pt`` checkpoint (ART-03), NaN guard (ERR-04).
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from engine_rul.features import WindowSet
from engine_rul.metrics import rmse
from engine_rul.utils import resolve_device

log = logging.getLogger(__name__)

DEFAULTS = {"hidden_size": 64, "num_layers": 2, "dropout": 0.2}


class LSTMRegressor(nn.Module):
    def __init__(
        self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        h2 = max(hidden_size // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, h2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, W, F] -> [B]
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class LSTMModel:
    name = "lstm"
    input_kind = "sequence"

    def __init__(self, params=None, seed: int = 0):
        self.params = {**DEFAULTS, **(params or {})}
        self.seed = seed
        self.net: LSTMRegressor | None = None
        self.target_scale = 125.0
        self.feature_names: list[str] | None = None
        self.device = "cpu"
        self.best_epoch: int | None = None
        self.best_val_rmse: float | None = None

    def _build(self, n_features: int) -> None:
        self.net = LSTMRegressor(n_features, **self.params).to(self.device)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def fit(self, train: WindowSet, val: WindowSet, cfg, workdir=None, tracker=None) -> dict:
        tc = cfg.train
        self.device = resolve_device(cfg.device)
        self.target_scale = float(cfg.data.r_max or 125.0)
        self.feature_names = list(train.feature_names)
        torch.manual_seed(cfg.seed)
        self._build(train.X_seq.shape[2])
        Xtr = torch.from_numpy(train.X_seq).to(self.device)
        ytr = torch.from_numpy((train.y / self.target_scale).astype(np.float32)).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=tc.lr)
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(cfg.seed)
        best, best_state, wait, history = np.inf, None, 0, []
        t_start = time.perf_counter()
        for epoch in range(1, tc.max_epochs + 1):
            self.net.train()
            perm = torch.randperm(len(Xtr), generator=gen).to(self.device)
            total, n = 0.0, 0
            for i in range(0, len(perm), tc.batch_size):
                b = perm[i : i + tc.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.net(Xtr[b]), ytr[b])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), tc.grad_clip)
                opt.step()
                total += loss.item() * len(b)
                n += len(b)
            train_loss = total / n
            val_rmse = rmse(val.y, np.clip(self.predict(val), 0, None))
            history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse})
            if tracker is not None:
                tracker.log_metrics(
                    {"train_loss": train_loss, "val_rmse_epoch": val_rmse}, step=epoch
                )
            if val_rmse < best - tc.min_delta:
                best, wait, self.best_epoch = val_rmse, 0, epoch
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
                )
                if workdir is not None:
                    self.best_val_rmse = best
                    self.save(workdir, filename="best.pt")
            else:
                wait += 1
                if wait >= tc.patience:
                    break
            log.debug("epoch %d loss %.5f val_rmse %.3f", epoch, train_loss, val_rmse)
        _sync(self.device)
        elapsed = time.perf_counter() - t_start
        self.net.load_state_dict(best_state)
        self.best_val_rmse = best
        log.info(
            "LSTM stopped after %d epochs; best epoch %d val RMSE %.3f",
            epoch,
            self.best_epoch,
            best,
        )
        return {
            "history": history,
            "epochs_run": epoch,
            "best_epoch": self.best_epoch,
            "best_val_rmse_dense": best,
            "seconds_per_epoch": elapsed / epoch,
            "device": self.device,
        }

    @torch.inference_mode()
    def predict(self, ws: WindowSet, batch_size: int = 4096) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(ws), batch_size):
            x = torch.from_numpy(ws.X_seq[i : i + batch_size]).to(self.device)
            out.append(self.net(x).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64) * self.target_scale

    def to(self, device: str) -> LSTMModel:
        self.device = device
        self.net.to(device)
        return self

    def save(self, d: str | Path, filename: str = "checkpoint.pt") -> Path:
        p = Path(d) / filename
        torch.save(
            {
                "model_state": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
                "model_params": self.params,
                "n_features": len(self.feature_names),
                "epoch": self.best_epoch,
                "best_val_rmse": self.best_val_rmse,
                "target_scale": self.target_scale,
                "feature_names": self.feature_names,
                "seed": self.seed,
                "torch_version": str(torch.__version__),
            },
            p,
        )
        return p

    @classmethod
    def load(cls, d: str | Path, device: str = "cpu", filename: str = "checkpoint.pt") -> LSTMModel:
        ck = torch.load(Path(d) / filename, map_location="cpu", weights_only=True)
        m = cls(ck["model_params"], ck["seed"])
        m.device = device
        m.target_scale = ck["target_scale"]
        m.feature_names = ck["feature_names"]
        m.best_epoch, m.best_val_rmse = ck["epoch"], ck["best_val_rmse"]
        m._build(ck["n_features"])
        m.net.load_state_dict(ck["model_state"])
        return m
