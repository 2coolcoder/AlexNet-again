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

# This machine is shared, so a fixed rendezvous port often collides. Pick a
# free one unless the caller pinned MASTER_PORT explicitly.
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
fi

echo "launching ${NPROC} ranks on port ${MASTER_PORT}"
exec torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    scripts/train.py "$@"
