"""Configuration: nested dataclasses, loaded from YAML and overridable from the CLI."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    root: str = "/data/users/cs24s008/Datasets/ilsvrc2012_256"
    splits_dir: str = "splits"
    # Per-GPU batch size. The global batch is batch_size * world_size, so a
    # single-GPU and a 2-GPU run give each GPU identical work per step.
    batch_size: int = 256
    workers: int = 16
    image_size: int = 224
    resize_size: int = 256
    # Limit to the first N classes (sorted by wnid) for fast smoke runs. 0 = all.
    subset_classes: int = 0
    pca_lighting: float = 0.0  # std of the paper's "fancy PCA" colour noise; 0 disables


@dataclass
class ModelConfig:
    # "bn"           -> AlexNetBN, single-device, BatchNorm in place of LRN
    # "two_column"   -> AlexNetTwoColumn, the paper's layer-wise split over 2 GPUs
    variant: str = "bn"
    num_classes: int = 1000
    dropout: float = 0.5


@dataclass
class OptimConfig:
    # lr = None means "derive from global batch": base_lr * global_batch / base_batch.
    lr: float | None = None
    base_lr: float = 0.01
    base_batch: int = 256
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    min_lr: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 90
    # "single" | "ddp" | "model_parallel"
    parallel_mode: str = "ddp"
    amp_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "none"
    channels_last: bool = True
    compile: bool = False
    sync_bn: bool = False
    bf16_grad_compress: bool = True  # halves the DDP all-reduce over PCIe
    seed: int = 42
    print_freq: int = 50
    resume: str = ""


@dataclass
class LoggingConfig:
    out_dir: str = "runs/default"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "alexnet-again"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass, rejecting unknown keys loudly."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    # `from __future__ import annotations` makes Field.type a string, so resolve
    # the real types here rather than reading them off the fields.
    hints = get_type_hints(cls)
    kwargs = {}
    for name, value in data.items():
        ftype = hints[name]
        if isinstance(value, dict) and is_dataclass(ftype):
            kwargs[name] = _from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(Config, raw)


def _coerce(current: Any, text: str) -> Any:
    """Coerce a CLI string to the type of the value it is replacing."""
    if isinstance(current, bool):
        return text.lower() in {"1", "true", "yes", "y"}
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if current is None:
        # Optional fields in this config are all numeric (e.g. optim.lr).
        try:
            return float(text)
        except ValueError:
            return text
    return text


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply `section.key=value` overrides in place, e.g. `train.epochs=3`."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"unknown config section {part!r} in {key!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"unknown config key {key!r}")
        setattr(target, leaf, _coerce(getattr(target, leaf), value))
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="section.key=value",
        help="override config values, e.g. --set train.epochs=3 data.batch_size=64",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return apply_overrides(load_config(args.config), args.overrides)
