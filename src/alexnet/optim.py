"""Optimizer construction and the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def scaled_lr(cfg, world_size: int) -> float:
    """Learning rate for the effective global batch.

    ``data.batch_size`` is per-GPU, so the global batch grows with world size.
    The rate follows it linearly from ``base_lr`` at ``base_batch``, keeping a
    single-GPU and a 2-GPU run consistently tuned without a config edit. An
    explicit ``optim.lr`` overrides the rule.
    """
    if cfg.optim.lr is not None:
        return float(cfg.optim.lr)
    global_batch = cfg.data.batch_size * world_size
    return cfg.optim.base_lr * global_batch / cfg.optim.base_batch


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters so decay applies only to conv/linear weights.

    Decaying BatchNorm scales/shifts and biases costs accuracy and is standard
    practice to avoid.
    """
    decay, no_decay = [], []
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if name == "bias" or isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                no_decay.append(p)
            else:
                decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg, world_size: int) -> torch.optim.Optimizer:
    lr = scaled_lr(cfg, world_size)
    return torch.optim.SGD(
        param_groups(model, cfg.optim.weight_decay),
        lr=lr,
        momentum=cfg.optim.momentum,
        nesterov=cfg.optim.nesterov,
    )


def build_scheduler(optimizer, cfg, steps_per_epoch: int) -> LambdaLR:
    """Per-iteration linear warmup then cosine decay to ``optim.min_lr``.

    Stepping per iteration rather than per epoch gives a smooth curve and makes
    the warmup meaningful on short runs.
    """
    warmup_steps = max(0, cfg.optim.warmup_epochs * steps_per_epoch)
    total_steps = max(1, cfg.train.epochs * steps_per_epoch)
    base = scaled_lr(cfg, 1) if cfg.optim.lr is None else float(cfg.optim.lr)
    floor = cfg.optim.min_lr / base if base > 0 else 0.0

    def fn(step: int) -> float:
        if step < warmup_steps:
            # +1 so the first step has a non-zero rate.
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=fn)
