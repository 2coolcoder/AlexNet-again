"""Distributed setup helpers for torchrun-launched DDP jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .p2p import configure_nccl_for_broken_p2p


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(parallel_mode: str) -> DistInfo:
    """Initialize the process group when launched under torchrun.

    ``parallel_mode`` of ``"ddp"`` uses one process per GPU. ``"single"`` and
    ``"model_parallel"`` run in a single process and skip the process group.
    """
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if parallel_mode != "ddp" or not launched or int(os.environ["WORLD_SIZE"]) < 2:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return DistInfo(False, 0, 0, 1, device)

    # NCCL deadlocks on this machine if it attempts P2P, so this must run
    # before the communicator is built.
    configure_nccl_for_broken_p2p()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device("cuda", local_rank),
    )
    return DistInfo(True, rank, local_rank, world_size, torch.device("cuda", local_rank))


def wrap_ddp(model, info: DistInfo, cfg):
    """Wrap in DDP, with a bf16 gradient compression hook when requested.

    AlexNet is ~62M parameters, so each step all-reduces ~250 MB in fp32 over
    PCIe with no NVLink. bf16 compression halves that, which is the single
    most effective inter-GPU optimization available on this hardware.
    """
    if not info.enabled:
        return model
    if cfg.train.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp = DistributedDataParallel(
        model,
        device_ids=[info.local_rank],
        output_device=info.local_rank,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    if cfg.train.bf16_grad_compress:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        ddp.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
    return ddp


def all_reduce_sum(value, device: torch.device, info: DistInfo) -> torch.Tensor:
    """Sum a scalar or tensor across ranks; a no-op when not distributed."""
    t = value if torch.is_tensor(value) else torch.tensor(value, device=device)
    t = t.to(device)
    if info.enabled:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def barrier(info: DistInfo) -> None:
    if info.enabled:
        dist.barrier()


def cleanup(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()
