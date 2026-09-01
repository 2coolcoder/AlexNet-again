import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from alexnet.config import Config
from alexnet.data import ImageNetListDataset, build_transforms

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_dataset(tmp_path):
    """Three classes with a handful of images each, in ImageNet layout."""
    root = tmp_path / "data"
    for ci, cls in enumerate(["n00000001", "n00000002", "n00000003"]):
        (root / cls).mkdir(parents=True)
        for i in range(6):
            Image.new("RGB", (40 + i, 32), (ci * 40, i * 10, 20)).save(
                root / cls / f"{cls}_{i}.JPEG"
            )
    return root


def _make_splits(root, out, val_per_class=2):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "make_splits.py"),
         "--root", str(root), "--out", str(out),
         "--val-per-class", str(val_per_class)],
        capture_output=True, text=True, check=True,
    )


def test_splits_are_disjoint_and_complete(tiny_dataset, tmp_path):
    out = tmp_path / "splits"
    _make_splits(tiny_dataset, out)
    train = (out / "train.txt").read_text().split()
    val = (out / "val.txt").read_text().split()
    train_paths = set(train[0::2])
    val_paths = set(val[0::2])

    assert not (train_paths & val_paths), "train and val overlap"
    assert len(train_paths) + len(val_paths) == 18  # 3 classes x 6 images
    assert len(val_paths) == 6                      # 2 per class
    # Every class appears in both splits.
    assert {p.split("/")[0] for p in val_paths} == {
        "n00000001", "n00000002", "n00000003"
    }
    assert (out / "classes.txt").read_text().split() == [
        "n00000001", "n00000002", "n00000003"
    ]


def test_splits_are_deterministic(tiny_dataset, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _make_splits(tiny_dataset, a)
    _make_splits(tiny_dataset, b)
    for name in ["train.txt", "val.txt", "classes.txt"]:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_labels_match_class_index(tiny_dataset, tmp_path):
    out = tmp_path / "splits"
    _make_splits(tiny_dataset, out)
    classes = (out / "classes.txt").read_text().split()
    for line in (out / "train.txt").read_text().splitlines():
        rel, label = line.rsplit(" ", 1)
        assert classes[int(label)] == rel.split("/")[0]


def test_dataset_loads_and_transforms(tiny_dataset, tmp_path):
    out = tmp_path / "splits"
    _make_splits(tiny_dataset, out)
    cfg = Config()
    ds = ImageNetListDataset(tiny_dataset, out / "train.txt", build_transforms(cfg, True))
    img, label = ds[0]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert isinstance(label, int)


def test_subset_classes_filters(tiny_dataset, tmp_path):
    out = tmp_path / "splits"
    _make_splits(tiny_dataset, out)
    cfg = Config()
    full = ImageNetListDataset(tiny_dataset, out / "train.txt", build_transforms(cfg, False))
    sub = ImageNetListDataset(
        tiny_dataset, out / "train.txt", build_transforms(cfg, False), subset_classes=2
    )
    assert len(sub) < len(full)
    assert all(label < 2 for _, label in sub.samples)


def test_eval_transform_is_deterministic(tiny_dataset, tmp_path):
    out = tmp_path / "splits"
    _make_splits(tiny_dataset, out)
    cfg = Config()
    ds = ImageNetListDataset(tiny_dataset, out / "val.txt", build_transforms(cfg, False))
    assert torch.equal(ds[0][0], ds[0][0])


def test_preprocess_resizes_shortest_side(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "n00000001").mkdir(parents=True)
    Image.new("RGB", (800, 400)).save(src / "n00000001" / "a.JPEG")
    Image.new("L", (300, 900)).save(src / "n00000001" / "b.JPEG")  # greyscale
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "preprocess_imagenet.py"),
         "--src", str(src), "--dst", str(dst), "--size", "256", "--workers", "2"],
        capture_output=True, text=True, check=True,
    )
    a = Image.open(dst / "n00000001" / "a.JPEG")
    b = Image.open(dst / "n00000001" / "b.JPEG")
    assert min(a.size) == 256 and a.size == (512, 256)
    assert min(b.size) == 256 and b.size == (256, 768)
    assert b.mode == "RGB", "greyscale input must be converted to RGB"


def test_unpadded_eval_sampler_covers_every_sample_once():
    """Evaluation must not double-count: DistributedSampler pads, this must not."""
    from alexnet.data import UnpaddedDistributedSampler

    class _DS:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    for total, world in [(50000, 2), (4999, 2), (10, 3), (7, 4)]:
        shards = [
            list(UnpaddedDistributedSampler(_DS(total), world, r)) for r in range(world)
        ]
        flat = [i for s in shards for i in s]
        assert sorted(flat) == list(range(total))       # every index exactly once
        assert max(map(len, shards)) - min(map(len, shards)) <= 1  # balanced


def test_non_wnid_directories_are_not_treated_as_classes(tiny_dataset, tmp_path):
    """Only wnid directories count as classes.

    Without this filter, any stray directory in the dataset root becomes a
    1001st class and every label after it shifts.
    """
    (tiny_dataset / "256px").mkdir()
    (tiny_dataset / "256px" / "n00000001").mkdir()
    Image.new("RGB", (40, 32)).save(tiny_dataset / "256px" / "n00000001" / "x.JPEG")
    out = tmp_path / "splits"
    _make_splits(tiny_dataset, out)
    classes = (out / "classes.txt").read_text().split()
    assert classes == ["n00000001", "n00000002", "n00000003"]
    assert "256px" not in classes
    assert not any(line.startswith("256px/") for line in
                   (out / "train.txt").read_text().splitlines())


def test_preprocess_skips_non_wnid_directories(tmp_path):
    """A destination nested in the source must not be re-processed on a re-run."""
    src, dst = tmp_path / "src", tmp_path / "src" / "256px"
    (src / "n00000001").mkdir(parents=True)
    Image.new("RGB", (800, 400)).save(src / "n00000001" / "a.JPEG")
    for _ in range(2):  # second run must be a no-op, not a recursive re-process
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "preprocess_imagenet.py"),
             "--src", str(src), "--dst", str(dst), "--workers", "2"],
            capture_output=True, text=True, check=True,
        )
    assert sorted(p.name for p in dst.iterdir()) == ["n00000001"]
    assert not (dst / "256px").exists()
