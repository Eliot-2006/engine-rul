"""Dataset acquisition, verification, parsing, schema validation and RUL labels.

Requirements: DATA-01..13, DATA-20..25, ERR-02, ERR-05.
Test labels are only reachable through ``load_test_rul`` (DATA-25 / EVAL-20).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine_rul import __version__
from engine_rul.errors import ChecksumError, DataValidationError
from engine_rul.utils import REPO_ROOT, sha256_file

log = logging.getLogger(__name__)

MANIFEST_PATH = REPO_ROOT / "configs" / "dataset_manifest.yaml"

OP_SETTINGS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OP_SETTINGS, *SENSORS]  # DATA-11

SENSOR_DESCRIPTIONS = {  # DATA-11 (Saxena et al. 2008)
    "s1": "T2 – total temperature at fan inlet (°R)",
    "s2": "T24 – total temperature at LPC outlet (°R)",
    "s3": "T30 – total temperature at HPC outlet (°R)",
    "s4": "T50 – total temperature at LPT outlet (°R)",
    "s5": "P2 – pressure at fan inlet (psia)",
    "s6": "P15 – total pressure in bypass duct (psia)",
    "s7": "P30 – total pressure at HPC outlet (psia)",
    "s8": "Nf – physical fan speed (rpm)",
    "s9": "Nc – physical core speed (rpm)",
    "s10": "epr – engine pressure ratio (P50/P2)",
    "s11": "Ps30 – static pressure at HPC outlet (psia)",
    "s12": "phi – fuel flow / Ps30 (pps/psi)",
    "s13": "NRf – corrected fan speed (rpm)",
    "s14": "NRc – corrected core speed (rpm)",
    "s15": "BPR – bypass ratio",
    "s16": "farB – burner fuel-air ratio",
    "s17": "htBleed – bleed enthalpy",
    "s18": "Nf_dmd – demanded fan speed (rpm)",
    "s19": "PCNfR_dmd – demanded corrected fan speed (rpm)",
    "s20": "W31 – HPT coolant bleed (lbm/s)",
    "s21": "W32 – LPT coolant bleed (lbm/s)",
}


# --------------------------------------------------------------------------- manifest / download


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f)


def _check_file(path: Path, expected: str, delete_on_mismatch: bool) -> None:
    actual = sha256_file(path)
    if actual != expected:
        if delete_on_mismatch:
            path.unlink(missing_ok=True)
        raise ChecksumError(
            f"checksum mismatch for {path.name}: expected {expected}, got {actual}"
            + (" (file deleted)" if delete_on_mismatch else "")
        )


def download(
    dest: str | Path = "data/raw",
    subsets: tuple[str, ...] = ("FD001",),
    manifest: dict | None = None,
    fetch=_fetch,
    retries: int = 3,
) -> Path:
    """DATA-01/02/06: download pinned raw files, verify SHA-256, write PROVENANCE.json."""
    manifest = manifest or load_manifest()
    src = manifest["source"]
    dest = Path(dest)
    files_meta = {}
    for subset in subsets:
        sub_dir = dest / subset
        sub_dir.mkdir(parents=True, exist_ok=True)
        for name, sha in manifest["subsets"][subset]["files"].items():
            path = sub_dir / name
            url = src["url_template"].format(repo=src["repo"], revision=src["revision"], file=name)
            if path.exists() and sha256_file(path) == sha:
                log.info("already present and verified: %s", path)
            else:
                for attempt in range(1, retries + 1):  # ERR-05
                    try:
                        log.info("downloading %s", url)
                        fetch(url, path)
                        break
                    except OSError as e:
                        if attempt == retries:
                            raise DataValidationError(f"download failed for {url}: {e}") from e
                        time.sleep(2**attempt)
                _check_file(path, sha, delete_on_mismatch=True)
            files_meta[f"{subset}/{name}"] = {"sha256": sha, "url": url}
    prov_path = dest / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {"files": {}}
    prov.update(
        {
            "source_repo": src["repo"],
            "revision": src["revision"],
            "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "tool": f"engine-rul {__version__}",
        }
    )
    prov["files"].update(files_meta)
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    return dest


def verify(dest: str | Path = "data/raw", subsets=("FD001",), manifest: dict | None = None) -> dict:
    """DATA-02: return {relative_path: ok}. Missing files count as not ok."""
    manifest = manifest or load_manifest()
    out = {}
    for subset in subsets:
        for name, sha in manifest["subsets"][subset]["files"].items():
            path = Path(dest) / subset / name
            out[f"{subset}/{name}"] = path.exists() and sha256_file(path) == sha
    return out


def verify_zip(zip_path: str | Path, manifest: dict | None = None) -> dict:
    """DATA-04: compare NASA's original CMAPSSData.zip against the pinned mirror checksums."""
    manifest = manifest or load_manifest()
    expected = {n: s for sub in manifest["subsets"].values() for n, s in sub["files"].items()}
    out = {}
    with zipfile.ZipFile(zip_path) as zf, tempfile.TemporaryDirectory() as tmp:
        members = {Path(m).name: m for m in zf.namelist()}
        for name, sha in expected.items():
            if name not in members:
                out[name] = None
                continue
            p = Path(zf.extract(members[name], tmp))
            out[name] = sha256_file(p) == sha
    return out


# --------------------------------------------------------------------------- parsing / validation


def subset_path(data_dir: str | Path, subset: str, kind: str) -> Path:
    """kind in {'train', 'test', 'RUL'}. ERR-02: a missing file tells the user what to run."""
    path = Path(data_dir) / subset / f"{kind}_{subset}.txt"
    if not path.exists():
        raise DataValidationError(
            f"missing data file {path}. Run `engine-rul data download` first."
        )
    return path


def _fail(path, check: str, bad=None):
    detail = f"; offending: {list(bad)[:10]}" if bad is not None else ""
    raise DataValidationError(f"{Path(path).name}: {check}{detail}")


def read_sensor_file(path: str | Path) -> pd.DataFrame:
    """Parse a train_/test_ file and validate the generic schema (DATA-10 c,d,h; DATA-12/13).

    Lines end with trailing whitespace, so the delimiter must be any-whitespace."""
    try:
        raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    except ValueError as e:
        _fail(path, f"non-numeric value ({e})")
    if raw.shape[1] != len(COLUMNS):
        _fail(path, f"expected {len(COLUMNS)} columns, found {raw.shape[1]}")
    raw.columns = COLUMNS
    vals = raw.to_numpy()
    if not np.isfinite(vals).all():
        rows = np.where(~np.isfinite(vals).all(axis=1))[0]
        _fail(path, "missing or non-finite values", rows)
    for c in ("unit", "cycle"):
        frac = raw[c] % 1 != 0
        if frac.any():
            _fail(path, f"{c} must be integer-valued", raw.index[frac])
    df = raw.astype({"unit": "int32", "cycle": "int32"})
    if df.duplicated(["unit", "cycle"]).any():
        _fail(
            path,
            "duplicate (unit, cycle) rows",
            df.loc[df.duplicated(["unit", "cycle"]), "unit"].unique(),
        )
    df = df.sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)
    first = df.groupby("unit")["cycle"].transform("min")
    steps = df.groupby("unit")["cycle"].diff().fillna(1)
    bad = df.loc[(first != 1) | (steps != 1), "unit"].unique()
    if len(bad):
        _fail(path, "cycles must start at 1 and increase by exactly 1 per unit", bad)
    return df


def load_train(data_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Training trajectories with uncapped RUL labels (DATA-20)."""
    return add_train_rul(read_sensor_file(subset_path(data_dir, subset, "train")))


def load_test_features(data_dir: str | Path, subset: str = "FD001") -> pd.DataFrame:
    """Test sensor trajectories only - no labels (safe for EDA and benchmarks)."""
    return read_sensor_file(subset_path(data_dir, subset, "test"))


def load_test_rul(data_dir: str | Path, subset: str = "FD001") -> np.ndarray:
    """DATA-21/25: true RUL after the last observed cycle, index i <-> unit i+1.

    Only the final test evaluation may call this (EVAL-20); tests monkeypatch it to raise."""
    path = subset_path(data_dir, subset, "RUL")
    vals = pd.read_csv(path, sep=r"\s+", header=None, dtype=np.float64)
    if vals.shape[1] != 1:
        _fail(path, f"expected 1 column, found {vals.shape[1]}")
    y = vals[0].to_numpy()
    if not np.isfinite(y).all() or (y % 1 != 0).any() or (y < 0).any():
        _fail(path, "labels must be finite non-negative integers")
    return y.astype(np.int64)


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """DATA-20: RUL(u, t) = T_u - t (0 at the last observed cycle)."""
    df = df.copy()
    df["rul"] = (df.groupby("unit")["cycle"].transform("max") - df["cycle"]).astype(np.float64)
    return df


def dense_test_rul(test_df: pd.DataFrame, final_rul: np.ndarray) -> pd.Series:
    """DATA-22 (analysis only): RUL(u, t) = T_u_obs - t + RUL_test(u)."""
    last = test_df.groupby("unit")["cycle"].transform("max")
    final = test_df["unit"].map(dict(zip(range(1, len(final_rul) + 1), final_rul, strict=True)))
    return (last - test_df["cycle"] + final).astype(np.float64)


def cap_rul(rul, r_max: float | None):
    """DATA-23: piecewise-linear target min(RUL, R_max); None means uncapped."""
    return rul if r_max is None else np.minimum(rul, r_max)


# --------------------------------------------------------------------------- invariants (DATA-10)


def check_train_invariants(train: pd.DataFrame, inv: dict, name: str = "train") -> None:
    life = train.groupby("unit")["cycle"].max()
    if len(train) != inv["train_rows"]:
        _fail(name, f"expected {inv['train_rows']} rows, found {len(train)}")
    if sorted(train["unit"].unique()) != list(range(1, inv["n_train_units"] + 1)):
        _fail(name, f"unit ids must be exactly 1..{inv['n_train_units']}")
    if [life.min(), life.max()] != inv["train_life"]:
        _fail(name, f"lifetime range {[int(life.min()), int(life.max())]} != {inv['train_life']}")


def check_test_invariants(test: pd.DataFrame, rul: np.ndarray, inv: dict) -> None:
    length = test.groupby("unit")["cycle"].max()
    if len(test) != inv["test_rows"]:
        _fail("test", f"expected {inv['test_rows']} rows, found {len(test)}")
    if sorted(test["unit"].unique()) != list(range(1, inv["n_test_units"] + 1)):
        _fail("test", f"unit ids must be exactly 1..{inv['n_test_units']}")
    if [length.min(), length.max()] != inv["test_len"]:
        _fail(
            "test",
            f"trajectory length range {[int(length.min()), int(length.max())]} != {inv['test_len']}",
        )
    if len(rul) != inv["n_test_units"]:
        _fail("RUL", f"expected {inv['n_test_units']} labels, found {len(rul)}")
    if [rul.min(), rul.max()] != inv["test_rul"]:
        _fail("RUL", f"label range {[int(rul.min()), int(rul.max())]} != {inv['test_rul']}")
