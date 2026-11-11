"""Typed experiment configuration (ADR-09, ERR-03, REPRO-03).

YAML files may contain ``extends: <other.yaml>`` (resolved relative to the file); the child is
deep-merged over the parent. Unknown keys raise ConfigError.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from engine_rul.errors import ConfigError
from engine_rul.utils import canonical_hash

FAMILIES = ("constant", "life_prior", "ridge", "hgb", "lstm")


@dataclass(frozen=True)
class DataConfig:
    subset: str = "FD001"
    data_dir: str = "data/raw"
    split_file: str = "splits/fd001_split.json"
    window: int = 30
    r_max: float | None = 125
    drop_columns: tuple[str, ...] = ("op1", "op2", "op3")
    expected_features: tuple[str, ...] | None = None  # PREP-01 assertion
    pad_short: bool = False
    check_invariants: bool = True  # DATA-10 against the dataset manifest


@dataclass(frozen=True)
class ModelConfig:
    family: str = "constant"
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 0.01
    grad_clip: float = 1.0


@dataclass(frozen=True)
class TrackingConfig:
    mlflow_uri: str | None = "sqlite:///mlflow.db"
    experiment: str = "fd001-dev"


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    seed: int = 0
    device: str = "auto"
    results_dir: str = "results"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data"]["drop_columns"] = list(d["data"]["drop_columns"])
        if d["data"]["expected_features"] is not None:
            d["data"]["expected_features"] = list(d["data"]["expected_features"])
        return d

    def hash(self) -> str:
        """Hash of everything that defines the experiment. Seed, device and output/tracking
        locations are excluded so that multi-seed runs of one config share a hash."""
        d = self.to_dict()
        for k in ("seed", "device", "results_dir", "tracking"):
            d.pop(k)
        d["data"].pop("data_dir")
        return canonical_hash(d)

    def replace(self, **changes) -> Config:
        """Return a copy with nested overrides, e.g. replace(seed=1, model={'params': {...}})."""
        return from_dict(_deep_merge(self.to_dict(), changes))


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "params":
            out[k] = _deep_merge(out[k], v)
        elif isinstance(v, dict) and k == "params" and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _build(cls, d: dict, where: str):
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(d).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    return d


def from_dict(d: dict) -> Config:
    d = _build(Config, dict(d), "config")
    data = dict(_build(DataConfig, d.get("data", {}), "data"))
    for k in ("drop_columns", "expected_features"):
        if data.get(k) is not None:
            data[k] = tuple(data[k])
    cfg = Config(
        model=ModelConfig(**_build(ModelConfig, d.get("model", {}), "model")),
        data=DataConfig(**data),
        train=TrainConfig(**_build(TrainConfig, d.get("train", {}), "train")),
        tracking=TrackingConfig(**_build(TrackingConfig, d.get("tracking", {}), "tracking")),
        **{k: d[k] for k in ("seed", "device", "results_dir") if k in d},
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.model.family not in FAMILIES:
        raise ConfigError(f"unknown model family {cfg.model.family!r}; expected one of {FAMILIES}")
    if not isinstance(cfg.model.params, dict):
        raise ConfigError("model.params must be a mapping")
    if not isinstance(cfg.data.window, int) or cfg.data.window < 1:
        raise ConfigError(f"data.window must be a positive int, got {cfg.data.window!r}")
    if cfg.data.r_max is not None and not cfg.data.r_max > 0:
        raise ConfigError(f"data.r_max must be > 0 or null, got {cfg.data.r_max!r}")
    if cfg.device not in ("auto", "cpu", "mps", "cuda"):
        raise ConfigError(f"device must be auto|cpu|mps|cuda, got {cfg.device!r}")
    t = cfg.train
    if t.lr <= 0 or t.batch_size < 1 or t.max_epochs < 1 or t.patience < 1:
        raise ConfigError("train: lr > 0, batch_size >= 1, max_epochs >= 1, patience >= 1 required")
    if not isinstance(cfg.seed, int):
        raise ConfigError("seed must be an int")


def _load_raw(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = d.pop("extends", None)
    if parent:
        d = _deep_merge(_load_raw(path.parent / parent), d)
    return d


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    d = _load_raw(Path(path))
    if overrides:
        d = _deep_merge(d, overrides)
    return from_dict(d)


def dump_config(cfg: Config, path: str | Path, header: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + text
    path.write_text(text)
