#!/usr/bin/env python
"""Measure training throughput for each parallelism mode.

Per-GPU batch size is held constant, so the 2-GPU data-parallel run does twice
the work per step (weak scaling) and images/sec stays the fair comparison.

Model parallelism is the odd case: the *model* is split rather than the batch,
so a per-GPU batch is not meaningful. It is run at the same global batch as the
single-GPU baseline, and again at the DDP global batch, so both comparisons are
available.

Synthetic data is used so the measurement reflects compute and inter-GPU
transfer rather than JPEG decode.

    python scripts/benchmark.py --modes single model_parallel
    torchrun --nproc_per_node=2 scripts/benchmark.py --modes ddp
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alexnet.config import Config, add_config_args, config_from_args  # noqa: E402
from alexnet.distributed import init_distributed, cleanup, wrap_ddp  # noqa: E402
from alexnet.engine import amp_context  # noqa: E402
from alexnet.model import build_model  # noqa: E402
from alexnet.optim import build_optimizer  # noqa: E402


def run_mode(mode: str, cfg: Config, steps: int, warmup: int) -> dict:
    cfg.train.parallel_mode = mode
    cfg.model.variant = "two_column" if mode == "model_parallel" else "bn"

    info = init_distributed(mode)
    model = build_model(cfg, info.device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = wrap_ddp(model, info, cfg)
    model.train()

    optimizer = build_optimizer(model, cfg, info.world_size)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing)

    bs = cfg.data.batch_size
    images = torch.randn(bs, 3, cfg.data.image_size, cfg.data.image_size,
                         device=info.device)
    if cfg.train.channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    target = torch.randint(0, cfg.model.num_classes, (bs,), device=info.device)

    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)

    elapsed = 0.0
    for step in range(steps + warmup):
        if step == warmup:
            torch.cuda.synchronize()
            elapsed = time.time()
        with amp_context(cfg, info.device):
            out = model(images)
            loss = criterion(out, target.to(out.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - elapsed

    # Global batch: DDP processes batch_size per rank; model parallel and
    # single process batch_size in total.
    global_batch = bs * info.world_size
    imgs = steps * global_batch
    peak = max(
        torch.cuda.max_memory_allocated(d) / 2**30
        for d in range(torch.cuda.device_count())
    )
    result = {
        "mode": mode,
        "world_size": info.world_size,
        "batch_per_gpu": bs if mode != "model_parallel" else None,
        "global_batch": global_batch,
        "img_per_s": imgs / elapsed,
        "ms_per_step": 1000 * elapsed / steps,
        "peak_mem_gib": peak,
    }
    if info.is_main:
        print(json.dumps(result))
    cleanup(info)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--modes", nargs="+", default=["single", "model_parallel"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = config_from_args(args)
    results = [run_mode(m, cfg, args.steps, args.warmup) for m in args.modes]
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
