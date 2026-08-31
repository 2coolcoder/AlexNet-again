"""Training and evaluation loops."""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .distributed import DistInfo, all_reduce_sum
from .metrics import AverageMeter, accuracy

_AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "none": None}


def amp_context(cfg, device: torch.device):
    """Autocast context for the configured precision.

    bfloat16 is the default on Ada: it has the same exponent range as fp32, so
    no GradScaler is needed and a whole class of overflow bugs disappears.
    """
    dtype = _AMP_DTYPES[cfg.train.amp_dtype]
    if dtype is None or device.type != "cuda":
        return torch.autocast("cuda", enabled=False)
    return torch.autocast("cuda", dtype=dtype)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    scaler,
    cfg,
    info: DistInfo,
    epoch: int,
    logger,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing)
    loss_meter = AverageMeter()
    seen = correct1 = correct5 = 0
    start = time.time()
    n_batches = len(loader)

    # The two-column model distributes the input to both GPUs itself, so it is
    # handed a host tensor; moving it to one GPU first would force a peer copy.
    to_device = cfg.train.parallel_mode != "model_parallel"

    for it, (images, target) in enumerate(loader):
        if to_device:
            images = images.to(info.device, non_blocking=True)
        if cfg.train.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        target = target.to(info.device, non_blocking=True)

        with amp_context(cfg, info.device):
            output = model(images)
            # The two-column model returns logits on cuda:0, which may differ
            # from the loader's device.
            loss = criterion(output, target.to(output.device))

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        bs = images.size(0)
        loss_meter.update(loss.item(), bs)
        c1, c5 = accuracy(output.detach(), target.to(output.device), (1, 5))
        correct1 += int(c1)
        correct5 += int(c5)
        seen += bs

        if info.is_main and (it % cfg.train.print_freq == 0 or it == n_batches - 1):
            rate = seen / max(1e-9, time.time() - start)
            logger.info(
                f"epoch {epoch:3d} [{it:>5d}/{n_batches}] "
                f"loss {loss_meter.avg:.4f}  top1 {100*correct1/seen:.2f}%  "
                f"lr {optimizer.param_groups[0]['lr']:.5f}  "
                f"{rate * info.world_size:.0f} img/s"
            )

    elapsed = time.time() - start
    return {
        "train/loss": loss_meter.avg,
        "train/top1": 100.0 * correct1 / max(1, seen),
        "train/top5": 100.0 * correct5 / max(1, seen),
        "train/epoch_time_s": elapsed,
        "train/img_per_s": seen * info.world_size / max(1e-9, elapsed),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, cfg, info: DistInfo) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = torch.zeros((), device=info.device)
    correct1 = torch.zeros((), device=info.device)
    correct5 = torch.zeros((), device=info.device)
    total = torch.zeros((), device=info.device)

    to_device = cfg.train.parallel_mode != "model_parallel"
    for images, target in loader:
        if to_device:
            images = images.to(info.device, non_blocking=True)
        if cfg.train.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        target = target.to(info.device, non_blocking=True)
        with amp_context(cfg, info.device):
            output = model(images)
            loss = criterion(output, target.to(output.device))
        c1, c5 = accuracy(output, target.to(output.device), (1, 5))
        bs = images.size(0)
        loss_sum += loss.float() * bs
        correct1 += c1.to(info.device)
        correct5 += c5.to(info.device)
        total += bs

    # Sum across ranks before dividing, so the mean is over the whole val set
    # rather than an average of per-rank averages.
    stats = torch.stack([loss_sum, correct1, correct5, total])
    stats = all_reduce_sum(stats, info.device, info)
    loss_sum, correct1, correct5, total = stats.tolist()
    return {
        "val/loss": loss_sum / max(1.0, total),
        "val/top1": 100.0 * correct1 / max(1.0, total),
        "val/top5": 100.0 * correct5 / max(1.0, total),
        "val/samples": total,
    }
