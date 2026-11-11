"""Experiment provenance (Phase 2 §5): git state, source snapshots, experiment directories.

If the working tree is dirty, ``source_provenance`` stores a content-addressed snapshot
(``git diff HEAD`` plus untracked project files) under ``results/snapshots/<sha256>.tar.gz`` and
returns its hash, so a result produced from uncommitted code can still be traced to exact source.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from engine_rul.errors import ConfigError
from engine_rul.utils import REPO_ROOT, git_info

SNAPSHOT_PATHS = ("src", "configs", "tests", "splits", "pyproject.toml", "uv.lock", "Makefile")
CATEGORIES = (
    "cv",
    "ablation",
    "validation_strategy",
    "conformal",
    "benchmark",
    "repro",
    "exploratory",
)


def _git(*args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return b""


def source_provenance(snapshot_dir: str | Path | None = None) -> dict:
    """Git commit + dirty flag; for a dirty tree also a deduplicated source snapshot."""
    info = git_info()
    out = {"git_commit": info["commit"], "git_dirty": info["dirty"], "source_snapshot": None}
    if not info["dirty"]:
        return out
    diff = _git("diff", "HEAD", "--binary", "--", *SNAPSHOT_PATHS)
    untracked = [
        p
        for p in _git("ls-files", "--others", "--exclude-standard", "--", *SNAPSHOT_PATHS)
        .decode()
        .splitlines()
        if p
    ]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:

        def add(name: str, data: bytes) -> None:
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime = (
                len(data),
                0,
            )  # mtime 0: identical content -> identical archive bytes
            tar.addfile(ti, io.BytesIO(data))

        add("tracked.diff", diff)
        for p in sorted(untracked):
            add(f"untracked/{p}", (REPO_ROOT / p).read_bytes())
    # hash the logical content (not the gzip bytes) so the id is stable
    h = hashlib.sha256(diff)
    for p in sorted(untracked):
        h.update(p.encode() + b"\0" + (REPO_ROOT / p).read_bytes())
    sha = h.hexdigest()
    sdir = Path(snapshot_dir or REPO_ROOT / "results" / "snapshots")
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"{sha[:16]}.tar.gz"
    if not path.exists():
        path.write_bytes(buf.getvalue())
    rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    out["source_snapshot"] = {"sha256": sha, "path": str(rel), "n_untracked": len(untracked)}
    return out


def experiment_dir(root: str | Path, category: str, name: str, force: bool = False) -> Path:
    """results/phase2/<category>/<name>/ - refuses to reuse an existing directory unless forced,
    so published or earlier experiment outputs are never silently overwritten."""
    if category not in CATEGORIES:
        raise ConfigError(f"unknown experiment category {category!r}; expected one of {CATEGORIES}")
    d = Path(root) / category / name
    if d.exists() and any(d.iterdir()) and not force:
        raise ConfigError(f"{d} already exists; choose a new experiment name or pass --force")
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(d: Path, manifest: dict) -> Path:
    manifest = {"created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"), **manifest}
    p = d / "manifest.json"
    p.write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return p


def refuse_overwrite(paths, force: bool = False) -> None:
    """Fail fast (before any training) if a published artifact would be overwritten.

    Published artifacts (frozen configs, tuning/sensitivity/summary JSON, final run registry)
    always embed run ids or timestamps, so any rewrite changes them; reruns must target another
    --results-dir/--final-dir, or pass --force deliberately."""
    if force:
        return
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise ConfigError(
            f"refusing to overwrite published artifact(s) {existing}; use a different "
            "--results-dir/--final-dir (e.g. `make repro-check`) or pass --force"
        )
