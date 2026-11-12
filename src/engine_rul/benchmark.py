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
from engine_rul.errors import ConfigError
from engine_rul.features import WindowSet, build_windows
from engine_rul.models import load_model
from engine_rul.models.lstm import _sync
from engine_rul.preprocess import Preprocessor
from engine_rul.provenance import refuse_overwrite
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
    ws = build_windows(pre.transform(test_df), pre.feature_names, cfg.data, end_points=pts)
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


def benchmark_runs(results_dir: Path, families: list[str]) -> tuple[dict[str, str], Path]:
    """Run id to benchmark per family, and the file it came from.

    Published layout: the median-seed run recorded in ``final/summary.json``. An isolated
    reproduction (``make reproduce``) has no test evaluation and so no summary; it uses the first
    seed in ``final_runs.json`` instead (latency does not depend on the seed)."""
    summary, registry = results_dir / "final" / "summary.json", results_dir / "final_runs.json"
    if summary.exists():
        fams = json.loads(summary.read_text())["families"]
        return {
            f: fams[f]["run_ids"][fams[f]["seeds"].index(fams[f]["median_seed"])] for f in families
        }, summary
    if registry.exists():
        runs = json.loads(registry.read_text())
        return {f: runs[f][0] for f in families}, registry
    raise ConfigError(f"no {summary} or {registry}; run `engine-rul final` first")


def run_benchmarks(
    families: list[str],
    results_dir: str | Path = "results",
    devices=("cpu",),
    force: bool = False,
) -> Path:
    results_dir = Path(results_dir)
    out = results_dir / "final" / "benchmarks.json"
    refuse_overwrite([out], force)
    run_ids, _ = benchmark_runs(results_dir, families)
    rows = []
    for f in families:
        for dev in (
            devices if f == "lstm" else ("cpu",)
        ):  # BENCH-06: accelerator only matters for torch
            log.info("benchmarking %s on %s", f, dev)
            rows.append(benchmark_run(results_dir / run_ids[f], device=dev))
    out.write_text(json.dumps({"environment": environment_info(), "rows": rows}, indent=1) + "\n")
    return out


# --------------------------------------------------------------------------- Phase 2 benchmarks
MODEL_BATCHES = (1, 10, 100, 1000)
E2E_ENGINES = (1, 10, 100)
SCOPE = {
    "model": "model.predict on already-built windows (sequence tensor / tabular features computed "
    "inside predict for Ridge and HGB); excludes I/O, preprocessing and window construction",
    "end_to_end": "Predictor.predict on raw sensor rows: z-score preprocessing, window construction, "
    "feature extraction, prediction, clipping and the output DataFrame; excludes file parsing and "
    "model loading (Predictor is constructed once)",
}


def _measure(fn, device: str, n_items: int) -> dict:
    import tracemalloc

    reps = 50 if n_items <= 100 else 20
    t = _time(fn, device, warmup=5, reps=reps)
    tracemalloc.start()
    fn()
    _sync(device)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return {
        "n": n_items,
        "reps": reps,
        "latency_ms_median": float(np.median(t) * 1e3),
        "latency_ms_p95": float(np.quantile(t, 0.95) * 1e3),
        "throughput_per_s": float(n_items / np.median(t)),
        "python_peak_alloc_mb": peak / 2**20,
    }


def run_phase2_benchmarks(
    families: list[str],
    results_dir: str | Path = "results",
    devices=("cpu",),
    root: str | Path = "results/phase2",
    name: str = "fd001_latency",
    force: bool = False,
) -> Path:
    """Model-only vs end-to-end latency for the Phase-1 median-seed models. Inputs are *training*
    file sensor rows (labels dropped), so the official test files are not read."""
    import resource

    from engine_rul.inference import Predictor
    from engine_rul.provenance import experiment_dir, source_provenance, write_manifest

    results_dir = Path(results_dir)
    exp = experiment_dir(root, "benchmark", name, force)
    run_ids, source = benchmark_runs(results_dir, families)
    rows = []
    for f in families:
        run_dir = results_dir / run_ids[f]
        for dev in devices if f == "lstm" else ("cpu",):
            pr = Predictor(run_dir, device=dev)
            raw = D.load_train(pr.cfg.data.data_dir, pr.cfg.data.subset).drop(
                columns=["rul"], errors="ignore"
            )
            units = np.sort(raw["unit"].unique())
            ws = build_windows(pr.pre.transform(raw), pr.pre.feature_names, pr.cfg.data)
            idx = np.random.default_rng(0).permutation(len(ws))
            log.info("phase-2 benchmark %s on %s", f, dev)
            for b in MODEL_BATCHES:
                sub = _subset(ws, idx[:b])
                rows.append(
                    {
                        "family": f,
                        "device": dev,
                        "scope": "model",
                        "unit": "windows",
                        **_measure(lambda s=sub, m=pr.model: m.predict(s), dev, b),
                    }
                )
            for n in E2E_ENGINES:
                df = raw[raw["unit"].isin(units[:n])]
                rows.append(
                    {
                        "family": f,
                        "device": dev,
                        "scope": "end_to_end",
                        "unit": "engines",
                        "n_raw_rows": int(len(df)),
                        **_measure(lambda d=df, p=pr: p.predict(d), dev, n),
                    }
                )
    out = {
        "scope_definitions": SCOPE,
        "rows": rows,
        "environment": environment_info(),
        "process_max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**20,  # bytes on macOS
        "memory_note": "python_peak_alloc_mb is tracemalloc (Python allocations, incl. NumPy buffers; "
        "excludes PyTorch's native allocator); process_max_rss_mb is the whole process peak",
    }
    (exp / "benchmarks.json").write_text(json.dumps(out, indent=1) + "\n")
    write_manifest(
        exp,
        {
            "category": "benchmark",
            "name": name,
            "families": families,
            "devices": list(devices),
            "source_runs": str(source),
            "provenance": source_provenance(),
            "official_test_set_used": False,
        },
    )
    return exp
