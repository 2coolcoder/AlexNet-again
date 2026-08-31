"""Dataset, augmentation and loader construction."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Principal components of ImageNet RGB pixel values, from the AlexNet paper's
# "fancy PCA" colour augmentation.
_PCA_EIGVAL = torch.tensor([0.2175, 0.0188, 0.0045])
_PCA_EIGVEC = torch.tensor(
    [
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ]
)


class PCALighting:
    """The paper's PCA colour noise: add a random multiple of the RGB eigenvectors."""

    def __init__(self, alphastd: float) -> None:
        self.alphastd = alphastd

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.alphastd == 0:
            return tensor
        alpha = torch.randn(3) * self.alphastd
        rgb = (_PCA_EIGVEC * alpha.view(1, 3) * _PCA_EIGVAL.view(1, 3)).sum(dim=1)
        return tensor + rgb.view(3, 1, 1)


class UnpaddedDistributedSampler(torch.utils.data.Sampler):
    """Shard a dataset across ranks without padding.

    ``DistributedSampler`` pads the last partial batch by repeating samples so
    every rank sees the same count. That is harmless for training but wrong for
    evaluation: the repeated images get counted twice, skewing the reported
    accuracy. This shards contiguously instead, so ranks may see slightly
    different counts and every sample is counted exactly once.
    """

    def __init__(self, dataset, num_replicas: int, rank: int) -> None:
        self.total = len(dataset)
        # Contiguous slice per rank; the remainder spreads over the first ranks.
        per, extra = divmod(self.total, num_replicas)
        start = rank * per + min(rank, extra)
        self.indices = list(range(start, start + per + (1 if rank < extra else 0)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class ImageNetListDataset(Dataset):
    """Reads a `relative/path label` index file.

    Using an index file avoids walking 1.28M directory entries on every start,
    and pins the train/val split so it cannot drift between runs.
    """

    def __init__(
        self,
        root: str | Path,
        list_file: str | Path,
        transform=None,
        subset_classes: int = 0,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        with open(list_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rel, label = line.rsplit(" ", 1)
                label = int(label)
                if subset_classes and label >= subset_classes:
                    continue
                self.samples.append((rel, label))
        if not self.samples:
            raise ValueError(f"no samples loaded from {list_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel, label = self.samples[index]
        with Image.open(self.root / rel) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(cfg, train: bool):
    if train:
        ops = [
            transforms.RandomResizedCrop(cfg.data.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
        if cfg.data.pca_lighting > 0:
            ops.append(PCALighting(cfg.data.pca_lighting))
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
        return transforms.Compose(ops)
    return transforms.Compose(
        [
            transforms.Resize(cfg.data.resize_size),
            transforms.CenterCrop(cfg.data.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_datasets(cfg):
    splits = Path(cfg.data.splits_dir)
    train_ds = ImageNetListDataset(
        cfg.data.root, splits / "train.txt", build_transforms(cfg, True),
        cfg.data.subset_classes,
    )
    val_ds = ImageNetListDataset(
        cfg.data.root, splits / "val.txt", build_transforms(cfg, False),
        cfg.data.subset_classes,
    )
    return train_ds, val_ds


def build_loaders(cfg, info):
    """Create train/val loaders. ``cfg.data.batch_size`` is per-GPU."""
    train_ds, val_ds = build_datasets(cfg)

    train_sampler = val_sampler = None
    if info.enabled:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        # Evaluate without padding so every image is counted exactly once.
        val_sampler = UnpaddedDistributedSampler(
            val_ds, num_replicas=info.world_size, rank=info.rank
        )

    common = dict(
        num_workers=cfg.data.workers,
        pin_memory=True,
        persistent_workers=cfg.data.workers > 0,
    )
    if cfg.data.workers > 0:
        common["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler
