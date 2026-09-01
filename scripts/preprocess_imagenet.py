#!/usr/bin/env python
"""Resize ILSVRC-2012 so the shortest side is 256px.

AlexNet is ~0.7 GFLOPs, so training on these GPUs is bound by JPEG decode
rather than compute. Decoding full-resolution images every epoch wastes most of
the CPU budget; a one-off resize shrinks the set to roughly 38 GB and cuts
per-epoch decode cost several-fold. The source directory is never modified --
the resized copy goes to its own directory.

Idempotent: files already present at the destination are skipped, so an
interrupted run can simply be restarted.

    python scripts/preprocess_imagenet.py \
        --src /data/users/cs24s008/Datasets/ilsvrc2012 \
        --dst /data/users/cs24s008/Datasets/ilsvrc2012_256
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

# ImageNet contains a handful of very large images; Pillow's decompression-bomb
# guard would otherwise refuse them.
Image.MAX_IMAGE_PIXELS = None

# Only WordNet-id directories are treated as classes, so a stray non-class
# directory in the dataset root is ignored rather than silently becoming a
# 1001st class (and a destination nested in the source is never re-processed).
WNID = re.compile(r"^n\d{8}$")


def resize_one(job: tuple[str, str, int, int]) -> tuple[str, str]:
    """Resize a single image. Returns (status, detail)."""
    src, dst, size, quality = job
    try:
        if os.path.exists(dst):
            return ("skipped", dst)
        with Image.open(src) as im:
            # ImageNet ships CMYK JPEGs, greyscale images, and at least one PNG
            # with a .JPEG extension. Normalising to RGB handles all of them.
            im = im.convert("RGB")
            w, h = im.size
            if min(w, h) != size:
                if w < h:
                    new = (size, max(1, round(h * size / w)))
                else:
                    new = (max(1, round(w * size / h)), size)
                im = im.resize(new, Image.BICUBIC)
            tmp = dst + ".tmp"
            im.save(tmp, "JPEG", quality=quality)
            os.replace(tmp, dst)  # never leave a partial file behind
        return ("ok", dst)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return ("failed", f"{src}: {type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=256, help="target shortest side")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    args = ap.parse_args()

    src_root, dst_root = Path(args.src), Path(args.dst)
    if not src_root.is_dir():
        print(f"source not found: {src_root}", file=sys.stderr)
        return 1

    classes = sorted(
        p.name for p in src_root.iterdir() if p.is_dir() and WNID.match(p.name)
    )
    print(f"found {len(classes)} class directories under {src_root}")

    jobs = []
    for cls in classes:
        (dst_root / cls).mkdir(parents=True, exist_ok=True)
        for img in sorted((src_root / cls).iterdir()):
            if img.is_file():
                jobs.append(
                    (str(img), str(dst_root / cls / img.name), args.size, args.quality)
                )
    print(f"{len(jobs):,} images -> {dst_root} using {args.workers} workers")

    counts = {"ok": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    with Pool(args.workers) as pool:
        for i, (status, detail) in enumerate(
            pool.imap_unordered(resize_one, jobs, chunksize=64), 1
        ):
            counts[status] += 1
            if status == "failed":
                failures.append(detail)
            if i % 50_000 == 0 or i == len(jobs):
                print(f"  {i:,}/{len(jobs):,}  {counts}", flush=True)

    print(f"done: {counts}")
    if failures:
        print(f"{len(failures)} failures; first 10:")
        for f in failures[:10]:
            print("  ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
