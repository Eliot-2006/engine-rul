"""Seeds, devices, hashing, git info, logging (REPRO-01..04, HW-01, ERR-06)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int, deterministic: bool = True) -> None:
    """REPRO-01/02: seed python, numpy (legacy global) and torch; enable CPU determinism."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> str:
    """HW-01: auto -> cuda > mps > cpu. Raises ConfigError if an explicit device is unavailable."""
    import torch

    from engine_rul.errors import ConfigError

    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if device == "auto":
        return "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
    if device not in available:
        raise ConfigError(f"unknown device {device!r}; expected auto|cpu|mps|cuda")
    if not available[device]:
        raise ConfigError(f"device {device!r} requested but not available on this machine")
    return device


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    """SHA-256 of canonical JSON (sorted keys) - order-insensitive config hash (REPRO-03)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD") or "unknown",
        "dirty": bool(run("status", "--porcelain")),
    }


def environment_info(device: str | None = None) -> dict:
    """BENCH-05 / REPRO-04: software + hardware summary."""
    import matplotlib
    import pandas
    import sklearn
    import torch

    try:
        import mlflow

        mlflow_version = mlflow.__version__
    except Exception:  # noqa: BLE001 - optional at report time
        mlflow_version = None
    cpu = platform.processor()
    if platform.system() == "Darwin":
        try:
            cpu = (
                subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
                ).stdout.strip()
                or cpu
            )
        except OSError:
            pass
    ram = None
    try:
        ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "ram_gb": ram,
        "device": device,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "matplotlib": matplotlib.__version__,
        "mlflow": mlflow_version,
    }
