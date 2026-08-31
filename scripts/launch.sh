#!/usr/bin/env bash
# Two-GPU DDP launcher.
#
# NCCL_P2P_DISABLE=1 is required on this machine: peer-to-peer DMA between the
# two GPUs is broken (IOMMU active, GPUs under separate PCIe root complexes),
# and NCCL hangs indefinitely if it tries to use it. The training code also
# auto-detects this, but exporting it here covers any tool that skips that path.
set -euo pipefail
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
NPROC=${NPROC:-2}
exec torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29500}" \
    scripts/train.py "$@"
