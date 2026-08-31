"""Checkpoint save/load with resume support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def unwrap(model: nn.Module) -> nn.Module:
    """Strip a DistributedDataParallel / compile wrapper."""
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_acc1: float,
    cfg,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_acc1": best_acc1,
        # Store on CPU so a checkpoint can be reloaded under any device layout.
        "model": {k: v.cpu() for k, v in unwrap(model).state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": cfg.to_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash mid-write can't corrupt the checkpoint


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    target = unwrap(model)
    # assign=False keeps each parameter on the device the model already placed
    # it on, which matters for the two-column model split across two GPUs.
    target.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
