"""Seeding, logging and metric sinks."""

from __future__ import annotations

import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, rank: int = 0) -> None:
    """Seed every RNG. Ranks are offset so they don't draw identical streams."""
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def setup_logging(out_dir: str | Path, rank: int = 0) -> logging.Logger:
    """Console + file logging; non-zero ranks stay quiet."""
    logger = logging.getLogger("alexnet")
    logger.handlers.clear()
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if rank == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(out_dir) / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class MetricLogger:
    """Writes per-epoch metrics to CSV, and to TensorBoard/wandb if enabled."""

    def __init__(self, cfg, enabled: bool = True) -> None:
        self.enabled = enabled
        self.writer = None
        self.wandb = None
        self.csv_path = Path(cfg.logging.out_dir) / "metrics.csv"
        self._csv_header_written = self.csv_path.exists()
        if not enabled:
            return
        Path(cfg.logging.out_dir).mkdir(parents=True, exist_ok=True)
        if cfg.logging.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(Path(cfg.logging.out_dir) / "tb"))
        if cfg.logging.wandb:
            import wandb

            wandb.init(
                project=cfg.logging.wandb_project,
                config=cfg.to_dict(),
                dir=cfg.logging.out_dir,
            )
            self.wandb = wandb

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.enabled:
            return
        row = {"epoch": step, **metrics}
        with open(self.csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if not self._csv_header_written:
                w.writeheader()
                self._csv_header_written = True
            w.writerow(row)
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
        if self.wandb is not None:
            self.wandb.log(row, step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"
