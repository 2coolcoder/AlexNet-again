"""Cross-GPU transfer that survives a broken peer-to-peer DMA path.

On this machine (2x RTX 6000 Ada under separate PCIe root complexes, IOMMU
active) direct device-to-device copies silently produce all-zero tensors, and
NCCL collectives hang when they try to use P2P. `cudaDeviceCanAccessPeer`
reports True, so nothing errors -- the data is just wrong.

Rather than hard-code the workaround, we probe once at runtime and pick the
transfer path from the result. On a healthy machine the direct copy is used and
this module costs a single small copy at startup.

The permanent fix is a kernel cmdline change (`iommu=pt` or `amd_iommu=off`),
which needs root and a reboot; see the README.
"""

from __future__ import annotations

import os
import warnings

import torch

_probe_cache: dict[tuple[int, int], bool] = {}


def peer_copy_is_healthy(src: torch.device, dst: torch.device) -> bool:
    """Return True if a direct ``src -> dst`` copy actually moves the data.

    Result is cached per device pair. Non-CUDA pairs are always healthy.
    """
    if src.type != "cuda" or dst.type != "cuda":
        return True
    si, di = src.index or 0, dst.index or 0
    if si == di:
        return True
    key = (si, di)
    if key in _probe_cache:
        return _probe_cache[key]

    # The probe pattern must be unpredictable. A deterministic one (e.g.
    # arange) can pass spuriously: the destination may reuse a cached
    # allocator block that already holds an identical pattern, so a copy that
    # moved nothing still compares equal.
    probe = torch.randn(1 << 20, device=src)
    got = probe.to(dst)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    ok = bool(torch.equal(got.cpu(), probe.cpu()))
    _probe_cache[key] = ok
    if not ok:
        warnings.warn(
            f"direct GPU peer copy cuda:{si} -> cuda:{di} is broken (returns "
            "corrupt data); falling back to host-staged transfers. Fix "
            "permanently with `iommu=pt` on the kernel cmdline.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ok


def any_peer_broken() -> bool:
    """Probe every ordered pair of visible CUDA devices."""
    n = torch.cuda.device_count()
    if n < 2:
        return False
    return any(
        not peer_copy_is_healthy(torch.device("cuda", i), torch.device("cuda", j))
        for i in range(n)
        for j in range(n)
        if i != j
    )


def xfer(t: torch.Tensor, dst: torch.device) -> torch.Tensor:
    """Autograd-safe move of ``t`` to ``dst``, routing via host if P2P is broken.

    Both directions of a pair are probed: a peer path that works one way and
    not the other is treated as unusable, since a half-working DMA route is a
    liability rather than an optimization.

    Both ``Tensor.to`` calls are differentiable, so gradients flow back along
    the same (host-staged) route.
    """
    dst = torch.device(dst)
    if t.device == dst:
        return t
    if peer_copy_is_healthy(t.device, dst) and peer_copy_is_healthy(dst, t.device):
        return t.to(dst, non_blocking=True)
    return t.to("cpu").to(dst)


def configure_nccl_for_broken_p2p() -> bool:
    """Set ``NCCL_P2P_DISABLE=1`` if peer DMA is broken and the user hasn't chosen.

    Must run before the NCCL communicator is created. Returns True if it set
    the variable. Without this, NCCL hangs on this machine instead of failing.
    """
    if "NCCL_P2P_DISABLE" in os.environ:
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    if any_peer_broken():
        os.environ["NCCL_P2P_DISABLE"] = "1"
        return True
    return False
