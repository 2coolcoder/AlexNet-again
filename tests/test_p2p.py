import pytest
import torch

from alexnet.p2p import peer_copy_is_healthy, xfer


def test_same_device_is_a_noop():
    t = torch.randn(4)
    assert xfer(t, torch.device("cpu")) is t
    assert peer_copy_is_healthy(torch.device("cpu"), torch.device("cpu"))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 CUDA devices")
def test_xfer_moves_data_correctly_whatever_the_p2p_state():
    """xfer must deliver correct bytes even where direct peer DMA is broken."""
    src, dst = torch.device("cuda:0"), torch.device("cuda:1")
    a = torch.randn(1 << 20, device=src)
    b = xfer(a, dst)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    assert b.device.index == 1
    assert torch.equal(a.cpu(), b.cpu())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 CUDA devices")
def test_xfer_is_differentiable_across_devices():
    x = torch.randn(4, 4, device="cuda:0", requires_grad=True)
    xfer(x * 2, torch.device("cuda:1")).sum().backward()
    torch.cuda.synchronize()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.full_like(x, 2.0))
