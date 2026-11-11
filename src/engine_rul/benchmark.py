"""Computational benchmarks: training time, inference latency, throughput, size (BENCH-01..06).

Latency is measured on test-set *sensor windows only*; no test labels are read here."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from engine_rul import data as D
from engine_rul.config import from_dict
from engine_rul.features import WindowSet, make_windows
from engine_rul.models import load_model
from engine_rul.models.lstm import _sync
from engine_rul.preprocess import Preprocessor
from engine_rul.utils import environment_info

log = logging.getLogger(__name__)


def _subset(ws: WindowSet, idx) -> WindowSet:
    return WindowSet(
        ws.X_seq[idx], ws.y[idx], ws.y_true[idx], ws.unit[idx], ws.cycle[idx], ws.feature_names
    )


def _time(fn, device: str, warmup: int, reps: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    _sync(device)
    out = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        out[i] = time.perf_counter() - t0
    return out


def benchmark_run(run_dir: Path, device: str = "cpu", warmup=20, reps=200, tp_reps=20) -> dict:
    rec = json.loads((run_dir / "run.json").read_text())
    cfg = from_dict(rec["config"])
    pre = Preprocessor.from_dict(json.loads((run_dir / "preprocessor.json").read_text()))
    test_df = D.load_test_features(cfg.data.data_dir, cfg.data.subset)
    last = test_df.groupby("unit")["cycle"].max()
    pts = np.column_stack([last.index.to_numpy(), last.to_numpy()])
    ws = make_windows(
        pre.transform(test_df),
        pre.feature_names,
        cfg.data.window,
        end_points=pts,
        pad_short=cfg.data.pad_short,
    )
    model = load_model(run_dir, device=device)
    one = _subset(ws, slice(0, 1))
    lat = _time(lambda: model.predict(one), device, warmup, reps)  # BENCH-02
    batch = _time(lambda: model.predict(ws), device, 3, tp_reps)  # BENCH-03
    artifact = next(
        run_dir / n
        for n in ("checkpoint.pt", "model.joblib", "model.json")
        if (run_dir / n).exists()
    )
    fi = rec.get("fit_info", {})
    return {
        "family": rec["family"],
        "run_id": rec["run_id"],
        "device": device,
        "train_time_s": rec["train_time_s"],
        "train_device": rec["env"]["device"],
        "epochs_run": fi.get("epochs_run"),
        "seconds_per_epoch": fi.get("seconds_per_epoch"),
        "latency_ms_median": float(np.median(lat) * 1e3),
        "latency_ms_p95": float(np.quantile(lat, 0.95) * 1e3),
        "throughput_windows_per_s": float(len(ws) / np.median(batch)),
        "n_params": rec["n_params"],
        "artifact": artifact.name,
        "artifact_bytes": artifact.stat().st_size,
    }


def run_benchmarks(
    families: list[str], results_dir: str | Path = "results", devices=("cpu",)
) -> Path:
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "final" / "summary.json").read_text())
    rows = []
    for f in families:
        fam = summary["families"][f]
        run_id = fam["run_ids"][fam["seeds"].index(fam["median_seed"])]
        for dev in (
            devices if f == "lstm" else ("cpu",)
        ):  # BENCH-06: accelerator only matters for torch
            log.info("benchmarking %s on %s", f, dev)
            rows.append(benchmark_run(results_dir / run_id, device=dev))
    out = results_dir / "final" / "benchmarks.json"
    out.write_text(json.dumps({"environment": environment_info(), "rows": rows}, indent=1) + "\n")
    return out
