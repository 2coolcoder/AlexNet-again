#!/usr/bin/env python
"""Evaluate a checkpoint.

    python scripts/evaluate.py --checkpoint runs/default/best.pth [--ten-crop]

``--ten-crop`` averages predictions over the four corners, the centre, and
their horizontal mirrors, as the original paper did at test time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alexnet.checkpoint import load_checkpoint  # noqa: E402
from alexnet.config import Config, add_config_args, config_from_args  # noqa: E402
from alexnet.data import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageNetListDataset,
    build_transforms,
)
from alexnet.engine import amp_context  # noqa: E402
from alexnet.metrics import accuracy  # noqa: E402
from alexnet.model import build_model  # noqa: E402


def ten_crop_transform(cfg):
    return transforms.Compose(
        [
            transforms.Resize(cfg.data.resize_size),
            transforms.TenCrop(cfg.data.image_size),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
                            transforms.functional.to_tensor(c)
                        )
                        for c in crops
                    ]
                )
            ),
        ]
    )


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ten-crop", action="store_true")
    ap.add_argument("--split", default="val.txt")
    args = ap.parse_args()

    cfg = config_from_args(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prefer the config the checkpoint was trained with, so architecture always
    # matches the weights.
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" in raw:
        from alexnet.config import _from_dict

        saved = _from_dict(Config, raw["config"])
        cfg.model = saved.model
        cfg.data.image_size = saved.data.image_size
        cfg.data.resize_size = saved.data.resize_size

    model = build_model(cfg, device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    tf = ten_crop_transform(cfg) if args.ten_crop else build_transforms(cfg, False)
    ds = ImageNetListDataset(
        cfg.data.root, Path(cfg.data.splits_dir) / args.split, tf,
        cfg.data.subset_classes,
    )
    loader = DataLoader(
        ds, batch_size=cfg.data.batch_size if not args.ten_crop else 64,
        shuffle=False, num_workers=cfg.data.workers, pin_memory=True,
    )

    correct1 = correct5 = total = 0
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with amp_context(cfg, device):
            if args.ten_crop:
                b, n, c, h, w = images.shape
                logits = model(images.view(-1, c, h, w))
                # Average probabilities, not logits, across the ten crops.
                output = logits.softmax(dim=1).view(b, n, -1).mean(dim=1)
            else:
                output = model(images)
        c1, c5 = accuracy(output.float(), target.to(output.device), (1, 5))
        correct1 += int(c1); correct5 += int(c5); total += target.size(0)

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    mode = "10-crop" if args.ten_crop else "center-crop"
    print(f"{mode} over {total:,} images")
    print(f"  top-1 accuracy {top1:.2f}%   top-1 error {100-top1:.2f}%")
    print(f"  top-5 accuracy {top5:.2f}%   top-5 error {100-top5:.2f}%")
    print("  AlexNet paper (single net, ILSVRC-2012 val): "
          "top-1 error 40.7%, top-5 error 18.2%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
