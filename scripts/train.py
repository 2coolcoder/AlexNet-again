#!/usr/bin/env python
"""Train AlexNet on ILSVRC-2012.

Single GPU:
    python scripts/train.py --config configs/default.yaml --set train.parallel_mode=single

Two GPUs (data parallel):
    torchrun --nproc_per_node=2 scripts/train.py --config configs/default.yaml

Two GPUs (the paper's layer-wise model parallelism, single process):
    python scripts/train.py --config configs/default.yaml \
        --set train.parallel_mode=model_parallel model.variant=two_column
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alexnet.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from alexnet.config import add_config_args, config_from_args  # noqa: E402
from alexnet.data import build_loaders  # noqa: E402
from alexnet.distributed import (  # noqa: E402
    barrier,
    cleanup,
    init_distributed,
    wrap_ddp,
)
from alexnet.engine import evaluate, train_one_epoch  # noqa: E402
from alexnet.model import build_model, count_parameters  # noqa: E402
from alexnet.optim import build_optimizer, build_scheduler, scaled_lr  # noqa: E402
from alexnet.utils import MetricLogger, format_duration, set_seed, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    cfg = config_from_args(ap.parse_args())

    if cfg.train.parallel_mode == "model_parallel":
        cfg.model.variant = "two_column"

    info = init_distributed(cfg.train.parallel_mode)
    set_seed(cfg.train.seed, info.rank)
    logger = setup_logging(cfg.logging.out_dir, info.rank)

    if info.is_main:
        Path(cfg.logging.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.logging.out_dir) / "config.json").write_text(
            json.dumps(cfg.to_dict(), indent=2)
        )
        logger.info(f"mode={cfg.train.parallel_mode} world_size={info.world_size}")
        logger.info(
            f"batch/gpu={cfg.data.batch_size} "
            f"global batch={cfg.data.batch_size * info.world_size} "
            f"lr={scaled_lr(cfg, info.world_size):.4f}"
        )

    train_loader, val_loader, train_sampler = build_loaders(cfg, info)
    if info.is_main:
        logger.info(
            f"train={len(train_loader.dataset):,} val={len(val_loader.dataset):,} "
            f"steps/epoch={len(train_loader)}"
        )

    model = build_model(cfg, info.device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if info.is_main:
        logger.info(f"model={cfg.model.variant} params={count_parameters(model):,}")

    optimizer = build_optimizer(model, cfg, info.world_size)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    # float16 needs loss scaling; bfloat16 and fp32 do not.
    scaler = (
        torch.amp.GradScaler("cuda") if cfg.train.amp_dtype == "float16" else None
    )

    start_epoch, best_acc1 = 0, 0.0
    if cfg.train.resume:
        ckpt = load_checkpoint(cfg.train.resume, model, optimizer, scheduler,
                               map_location=str(info.device))
        start_epoch = ckpt["epoch"] + 1
        best_acc1 = ckpt.get("best_acc1", 0.0)
        logger.info(f"resumed from {cfg.train.resume} at epoch {start_epoch}")

    model = wrap_ddp(model, info, cfg)
    if cfg.train.compile:
        model = torch.compile(model)

    metrics = MetricLogger(cfg, enabled=info.is_main)
    out_dir = Path(cfg.logging.out_dir)
    t0 = time.time()

    for epoch in range(start_epoch, cfg.train.epochs):
        if train_sampler is not None:
            # Reshuffles differently each epoch; without this every epoch sees
            # the same per-rank ordering.
            train_sampler.set_epoch(epoch)

        tr = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                             cfg, info, epoch, logger)
        va = evaluate(model, val_loader, cfg, info)

        if info.is_main:
            logger.info(
                f"epoch {epoch:3d} done  train_loss {tr['train/loss']:.4f}  "
                f"val_top1 {va['val/top1']:.2f}%  val_top5 {va['val/top5']:.2f}%  "
                f"({format_duration(tr['train/epoch_time_s'])}, "
                f"{tr['train/img_per_s']:.0f} img/s)"
            )
            metrics.log(epoch, {**tr, **va, "lr": optimizer.param_groups[0]["lr"]})

            is_best = va["val/top1"] > best_acc1
            best_acc1 = max(best_acc1, va["val/top1"])
            save_checkpoint(out_dir / "last.pth", model, optimizer, scheduler,
                            epoch, best_acc1, cfg)
            if is_best:
                save_checkpoint(out_dir / "best.pth", model, optimizer, scheduler,
                                epoch, best_acc1, cfg)
        barrier(info)

    if info.is_main:
        logger.info(
            f"finished in {format_duration(time.time() - t0)}  "
            f"best val top1 {best_acc1:.2f}%  (top-1 error {100 - best_acc1:.2f}%)"
        )
        metrics.close()
    cleanup(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
