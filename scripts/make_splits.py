#!/usr/bin/env python
"""Build deterministic train/val index files.

Only the ILSVRC-2012 *train* split is available locally, so validation is a
held-out slice of it: 50 images per class (50,000 total), matching the size of
the official validation set. Because it comes from the same collection as the
training data, accuracy measured on it reads slightly optimistic compared with
the official val set -- see the README.

The split is a pure function of the class name and file list, so re-running
this reproduces byte-identical files.

    python scripts/make_splits.py --root <resized-root> --out splits
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

SEED = 1234

# ImageNet class directories are WordNet ids: 'n' followed by 8 digits. Matching
# on that rather than "any subdirectory" means a stray directory in the dataset
# root is never mistaken for a 1001st class, which would shift every label.
WNID = re.compile(r"^n\d{8}$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="dataset root of wnid directories")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--val-per-class", type=int, default=50)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(
        p.name for p in root.iterdir() if p.is_dir() and WNID.match(p.name)
    )
    if not classes:
        raise SystemExit(f"no wnid class directories (nXXXXXXXX) under {root}")

    (out / "classes.txt").write_text("\n".join(classes) + "\n")

    train_rows, val_rows = [], []
    for idx, cls in enumerate(classes):
        files = sorted(p.name for p in (root / cls).iterdir() if p.is_file())
        # Seed per class so adding a class never reshuffles the others.
        rng = random.Random(SEED + idx)
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_val = min(args.val_per_class, max(0, len(shuffled) - 1))
        for name in shuffled[:n_val]:
            val_rows.append(f"{cls}/{name} {idx}")
        for name in shuffled[n_val:]:
            train_rows.append(f"{cls}/{name} {idx}")

    # Sort so the files are stable regardless of shuffle order.
    (out / "train.txt").write_text("\n".join(sorted(train_rows)) + "\n")
    (out / "val.txt").write_text("\n".join(sorted(val_rows)) + "\n")
    print(f"classes={len(classes)}  train={len(train_rows):,}  val={len(val_rows):,}")
    print(f"wrote {out}/train.txt, {out}/val.txt, {out}/classes.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
