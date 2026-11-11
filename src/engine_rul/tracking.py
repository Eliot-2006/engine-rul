"""Experiment tracking: MLflow + JSON source of truth (ADR-07, TRACK-01..05).

``track_run`` always writes ``<results_dir>/<run_id>/run.json``; MLflow logging is best effort
and never fails a run (TRACK-04)."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


class Run:
    def __init__(self, run_id: str, run_dir: Path, record: dict, mlflow_mod=None):
        self.run_id = run_id
        self.dir = run_dir
        self.record = record
        self._mlflow = mlflow_mod
        self._mlflow_root = mlflow_mod  # kept so the run is always ended, even after a failure

    def _safe(self, fn, *a, **k):
        if self._mlflow is None:
            return
        try:
            fn(*a, **k)
        except Exception as e:  # noqa: BLE001 - tracking must never fail a run
            log.warning("MLflow logging failed (%s); continuing with JSON only", e)
            self._mlflow = None

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        clean = {
            k: float(v) for k, v in metrics.items() if isinstance(v, int | float) and v is not None
        }
        if self._mlflow is not None:
            self._safe(self._mlflow.log_metrics, clean, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self._mlflow is not None:
            self._safe(self._mlflow.log_artifact, str(path))

    def write(self) -> Path:
        p = self.dir / "run.json"
        p.write_text(json.dumps(self.record, indent=1, default=str) + "\n")
        return p


def new_run_id(family: str, config_hash: str, seed: int) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{family}-{config_hash[:8]}-s{seed}-{ts}"


@contextmanager
def track_run(cfg, stage: str, record: dict):
    """Context manager yielding a Run; marks status finished/failed and always writes run.json."""
    run_id = new_run_id(cfg.model.family, cfg.hash(), cfg.seed)
    run_dir = Path(cfg.results_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "run_id": run_id,
        "family": cfg.model.family,
        "stage": stage,
        "status": "running",
        **record,
    }
    mlflow_mod = None
    if cfg.tracking.mlflow_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(cfg.tracking.mlflow_uri)
            mlflow.set_experiment("fd001-final" if stage == "final" else cfg.tracking.experiment)
            mlflow.start_run(run_name=run_id)
            mlflow.log_params({k: str(v)[:500] for k, v in _flatten(cfg.to_dict()).items()})
            mlflow.set_tags(
                {
                    "run_id": run_id,
                    "family": cfg.model.family,
                    "stage": stage,
                    "config_hash": cfg.hash(),
                    "git_commit": record.get("git", {}).get("commit"),
                    "git_dirty": record.get("git", {}).get("dirty"),
                    "data_sha256": record.get("data_sha256"),
                    "split_sha256": record.get("split_sha256"),
                }
            )
            mlflow_mod = mlflow
        except Exception as e:  # noqa: BLE001
            log.warning("MLflow unavailable (%s); writing JSON only", e)
    run = Run(run_id, run_dir, record, mlflow_mod)
    try:
        yield run
        run.record["status"] = "finished"
    except BaseException:
        run.record["status"] = "failed"
        raise
    finally:
        run.write()
        if run._mlflow is not None:
            run._safe(run._mlflow.log_artifact, str(run_dir / "run.json"))
            for name in (
                "preprocessor.json",
                "predictions_val.csv",
                "model.json",
                "model.joblib",
                "checkpoint.pt",
            ):
                if (run_dir / name).exists():
                    run._safe(run._mlflow.log_artifact, str(run_dir / name))
        if run._mlflow_root is not None:
            try:
                run._mlflow_root.end_run(
                    "FINISHED" if run.record["status"] == "finished" else "FAILED"
                )
            except Exception as e:  # noqa: BLE001
                log.warning("MLflow end_run failed: %s", e)
