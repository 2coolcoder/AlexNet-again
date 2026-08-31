"""AlexNet architectures.

Two variants, both using the paper's layer geometry:

* ``AlexNetBN``        - single-device, full-width, BatchNorm in place of LRN.
* ``AlexNetTwoColumn`` - the paper's two-column split laid across two GPUs,
  reproducing the original layer-wise model parallelism.

Deviation from Krizhevsky et al. (2012): Local Response Normalization is
replaced by BatchNorm. LRN is slow and contributes little; BatchNorm trains
faster and more stably. This is a deliberate, documented change.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .p2p import xfer

# Paper widths for the full (both-columns-combined) network.
FULL_WIDTHS = (96, 256, 384, 384, 256)
# Per-column widths for the two-column split: exactly half of the above.
COLUMN_WIDTHS = tuple(w // 2 for w in FULL_WIDTHS)


def he_init(module: nn.Module) -> None:
    """He (Kaiming) initialization, as called for by the modern recipe."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def _conv_bn_relu(cin: int, cout: int, **kw) -> nn.Sequential:
    # bias=False: the BatchNorm immediately after supplies the shift.
    return nn.Sequential(
        nn.Conv2d(cin, cout, bias=False, **kw),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class AlexNetBN(nn.Module):
    """Full-width AlexNet on a single device.

    Input is 224x224. conv1 uses padding=2 with an 11x11 stride-4 kernel so the
    first feature map is 55x55, matching the paper (which effectively assumed a
    227x227 input).
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = FULL_WIDTHS
        self.features = nn.Sequential(
            _conv_bn_relu(3, c1, kernel_size=11, stride=4, padding=2),   # -> 55x55
            nn.MaxPool2d(kernel_size=3, stride=2),                        # -> 27x27
            _conv_bn_relu(c1, c2, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),                        # -> 13x13
            _conv_bn_relu(c2, c3, kernel_size=3, padding=1),
            _conv_bn_relu(c3, c4, kernel_size=3, padding=1),
            _conv_bn_relu(c4, c5, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),                        # -> 6x6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c5 * 6 * 6, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        he_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


class _Column(nn.Module):
    """One of the paper's two columns, resident on a single GPU.

    Split at conv3: ``stem`` runs conv1-conv2 on this column's own input, and
    ``head`` runs conv3-conv5 on the *concatenation* of both columns' conv2
    outputs. That concat is the cross-GPU hop the paper describes.
    """

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = COLUMN_WIDTHS
        self.stem = nn.Sequential(
            _conv_bn_relu(3, c1, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _conv_bn_relu(c1, c2, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # conv3 sees both columns (2 * c2 input channels); conv4/conv5 stay local.
        self.head = nn.Sequential(
            _conv_bn_relu(2 * c2, c3, kernel_size=3, padding=1),
            _conv_bn_relu(c3, c4, kernel_size=3, padding=1),
            _conv_bn_relu(c4, c5, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )


class AlexNetTwoColumn(nn.Module):
    """The paper's layer-wise model parallelism across two GPUs.

    Column A lives on ``devices[0]``, column B on ``devices[1]``. Following the
    paper, conv3 and the fully-connected layers see both columns while
    conv4/conv5 stay within a column. Each column's stem runs on its own CUDA
    stream so the two halves genuinely overlap rather than serializing.

    The output is returned on ``devices[0]``; compute the loss there.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
        devices: tuple[torch.device, torch.device] | None = None,
    ) -> None:
        super().__init__()
        if devices is None:
            devices = (torch.device("cuda:0"), torch.device("cuda:1"))
        self.dev_a, self.dev_b = torch.device(devices[0]), torch.device(devices[1])

        self.col_a = _Column()
        self.col_b = _Column()

        c5 = COLUMN_WIDTHS[4]
        flat = 2 * c5 * 6 * 6  # both columns' conv5 outputs feed the FC stack
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        he_init(self)

        self.col_a.to(self.dev_a)
        self.col_b.to(self.dev_b)
        self.classifier.to(self.dev_a)

        self._stream_a: torch.cuda.Stream | None = None
        self._stream_b: torch.cuda.Stream | None = None
        if self.dev_a.type == "cuda" and self.dev_b.type == "cuda":
            self._stream_a = torch.cuda.Stream(device=self.dev_a)
            self._stream_b = torch.cuda.Stream(device=self.dev_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each column loads its own copy of the input. Feeding this model a
        # tensor that already sits on one GPU forces a peer copy to the other,
        # which is exactly the path that is broken here -- `xfer` keeps that
        # correct. Feeding it a host tensor (see engine.py) avoids the detour
        # entirely, giving both columns a direct host-to-device copy.
        xa = xfer(x, self.dev_a)
        xb = xfer(x, self.dev_b)

        if self._stream_a is not None:
            # Run both stems concurrently, then join before the cross-column concat.
            cur_a = torch.cuda.current_stream(self.dev_a)
            cur_b = torch.cuda.current_stream(self.dev_b)
            self._stream_a.wait_stream(cur_a)
            self._stream_b.wait_stream(cur_b)
            with torch.cuda.stream(self._stream_a):
                sa = self.col_a.stem(xa)
            with torch.cuda.stream(self._stream_b):
                sb = self.col_b.stem(xb)
            cur_a.wait_stream(self._stream_a)
            cur_b.wait_stream(self._stream_b)
            # Keep the tensors alive until the consuming stream is done with them.
            sa.record_stream(cur_a)
            sb.record_stream(cur_b)
        else:
            sa = self.col_a.stem(xa)
            sb = self.col_b.stem(xb)

        # The cross-GPU hop: each column's conv3 needs both stems' feature maps.
        # `xfer` routes via the host when direct peer DMA is broken (see p2p.py).
        sa_on_b = xfer(sa, self.dev_b)
        sb_on_a = xfer(sb, self.dev_a)

        ha = self.col_a.head(torch.cat([sa, sb_on_a], dim=1))
        hb = self.col_b.head(torch.cat([sa_on_b, sb], dim=1))

        # Gather both columns onto device A for the fully-connected stack.
        feats = torch.cat(
            [torch.flatten(ha, 1), xfer(torch.flatten(hb, 1), self.dev_a)], dim=1
        )
        return self.classifier(feats)


def build_model(cfg, device: torch.device | None = None) -> nn.Module:
    """Construct the model named by ``cfg.model.variant``."""
    variant = cfg.model.variant
    if variant == "bn":
        model = AlexNetBN(num_classes=cfg.model.num_classes, dropout=cfg.model.dropout)
        if device is not None:
            model.to(device)
        return model
    if variant == "two_column":
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                "model.variant='two_column' needs 2 visible CUDA devices, found "
                f"{torch.cuda.device_count()}"
            )
        return AlexNetTwoColumn(
            num_classes=cfg.model.num_classes, dropout=cfg.model.dropout
        )
    raise ValueError(f"unknown model.variant {variant!r} (expected 'bn' or 'two_column')")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
