import warnings

import pytest
import torch
import torch.nn as nn

from alexnet.config import Config
from alexnet.model import (
    COLUMN_WIDTHS,
    FULL_WIDTHS,
    AlexNetBN,
    AlexNetTwoColumn,
    build_model,
    count_parameters,
)

needs_2gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires 2 CUDA devices"
)


def test_forward_shape_and_size():
    m = AlexNetBN(num_classes=1000).eval()
    x = torch.randn(2, 3, 224, 224)
    assert m(x).shape == (2, 1000)
    # Roughly the paper's 60M parameters.
    assert 55e6 < count_parameters(m) < 70e6


def test_conv5_feature_map_matches_paper():
    """The paper's conv5 output is 256 channels at 6x6 after the final pool."""
    m = AlexNetBN().eval()
    feats = m.features(torch.randn(1, 3, 224, 224))
    assert feats.shape[1:] == (FULL_WIDTHS[4], 6, 6)


def test_columns_are_half_width():
    assert COLUMN_WIDTHS == tuple(w // 2 for w in FULL_WIDTHS)


def test_he_init_scales_with_fan_out():
    """He init gives conv weights std ~ sqrt(2 / fan_out)."""
    m = AlexNetBN()
    conv = m.features[2][0]  # conv2: 48*... -> checked against its own fan_out
    fan_out = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]
    assert conv.weight.std().item() == pytest.approx((2.0 / fan_out) ** 0.5, rel=0.15)


def test_bn_replaces_lrn_and_conv_bias_is_absorbed():
    m = AlexNetBN()
    assert not any(isinstance(mod, nn.LocalResponseNorm) for mod in m.modules())
    assert any(isinstance(mod, nn.BatchNorm2d) for mod in m.modules())
    # Every conv followed by BN should carry no bias of its own.
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            assert mod.bias is None


def test_build_model_rejects_unknown_variant():
    cfg = Config()
    cfg.model.variant = "nope"
    with pytest.raises(ValueError):
        build_model(cfg)


def test_two_column_cpu_forward():
    cpu = torch.device("cpu")
    m = AlexNetTwoColumn(devices=(cpu, cpu)).eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 1000)


@needs_2gpu
def test_two_column_places_params_on_both_gpus():
    m = AlexNetTwoColumn()
    devs = {p.device.index for p in m.parameters()}
    assert devs == {0, 1}
    assert next(m.col_a.parameters()).device.index == 0
    assert next(m.col_b.parameters()).device.index == 1


@needs_2gpu
def test_two_column_matches_single_device_reference():
    """The split model must compute the same function as one on a single device.

    This is the regression guard for the broken peer-copy path: a silently
    corrupt cross-GPU transfer shows up here as a large mismatch.
    """
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.manual_seed(0)
            gpu = AlexNetTwoColumn()
            cpu = AlexNetTwoColumn(devices=(torch.device("cpu"), torch.device("cpu")))
            cpu.load_state_dict({k: v.cpu() for k, v in gpu.state_dict().items()})
            gpu.eval()
            cpu.eval()
            x = torch.randn(4, 3, 224, 224)
            with torch.no_grad():
                assert torch.allclose(gpu(x).cpu(), cpu(x), atol=1e-4)
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32


@needs_2gpu
def test_two_column_gradients_reach_both_columns():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = AlexNetTwoColumn().train()
        out = m(torch.randn(4, 3, 224, 224))
        nn.functional.cross_entropy(
            out, torch.randint(0, 1000, (4,), device=out.device)
        ).backward()
        torch.cuda.synchronize()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
    assert m.col_b.stem[0][0].weight.grad.abs().sum() > 0


@needs_2gpu
def test_two_column_handles_gpu_resident_input():
    """Feeding an input that already lives on one GPU must still reach column B.

    The naive `x.to(dev_b)` is a peer copy, which on this machine silently
    yields zeros -- column B would train on blank images with no error raised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = AlexNetTwoColumn().eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            from_host = m(x)
            from_gpu = m(x.to("cuda:0"))
        torch.cuda.synchronize()
    # Same input, so the two must agree regardless of where it started.
    assert torch.allclose(from_host, from_gpu, atol=1e-4)
    # And column B must genuinely contribute: zeroing it must change the output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            for p_ in m.col_b.parameters():
                p_.zero_()
            zeroed = m(x.to("cuda:0"))
    assert not torch.allclose(from_gpu, zeroed, atol=1e-4)
