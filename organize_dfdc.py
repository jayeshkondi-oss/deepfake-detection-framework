"""
organize_dfdc.py
Sort a raw DFDC folder (a pile of .mp4 files + metadata.json) into the
real/ and fake/ layout DeepfakeDataset expects.

DFDC's metadata.json looks like:
    {
        "aaaaaa.mp4": {"label": "FAKE", "split": "train", "original": "bbbbbb.mp4"},
        "bbbbbb.mp4": {"label": "REAL", "split": "train"}
    }

DFDC is naturally very imbalanced (many more FAKE than REAL videos,
since each real video has several fakes derived from it) -- taking
just the first N entries in metadata.json order can easily produce a
dataset that's 80%+ fake, which makes both training and validation
metrics unreliable on small subsets. By default, this script now
balances real vs fake counts (use --no_balance to get the old
first-N-in-order behavior instead).

Usage:
    python organize_dfdc.py --source dfdc_train_part_0 --dest data/raw

    # Combine multiple DFDC parts into one organized folder:
    python organize_dfdc.py --source dfdc_train_part_0 --dest data/raw
    python organize_dfdc.py --source dfdc_train_part_1 --dest data/raw
    ...

    # Balanced subset for a quick smoke test (default behavior):
    python organize_dfdc.py --source dfdc_train_part_0 --dest data/smoke_test --max_videos 120

    # Old behavior: first N in metadata order, no balancing:
    python organize_dfdc.py --source dfdc_train_part_0 --dest data/raw --no_balance
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm import tqdm


def _select_balanced(metadata: dict, max_videos: int | None, seed: int = 42):
    """
    Split metadata into real/fake lists and, if max_videos is given,
    cap each class at max_videos // 2 -- so the result is as close to
    50/50 as the source data allows, not just whatever order
    metadata.json happens to list things in.
    """
    real_items = [
        (name, info) for name, info in metadata.items()
        if info.get("label") == "REAL"
    ]
    fake_items = [
        (name, info) for name, info in metadata.items()
        if info.get("label") != "REAL"
    ]

    rng = random.Random(seed)
    rng.shuffle(real_items)
    rng.shuffle(fake_items)

    if max_videos is not None:
        per_class = max_videos // 2
        real_items = real_items[:per_class]
        fake_items = fake_items[:per_class]

    return real_items + fake_items


def organize_dfdc(
    source_dir: Path,
    dest_dir: Path,
    max_videos: int | None = None,
    balance: bool = True,
):
    metadata_path = source_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {source_dir}. "
            "Make sure --source points to a DFDC part folder "
            "(e.g. dfdc_train_part_0)."
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    (dest_dir / "real").mkdir(parents=True, exist_ok=True)
    (dest_dir / "fake").mkdir(parents=True, exist_ok=True)

    real_count = 0
    fake_count = 0
    skipped = 0

    if balance:
        items = _select_balanced(metadata, max_videos)
    else:
        items = list(metadata.items())
        if max_videos is not None:
            items = items[:max_videos]

    for filename, info in tqdm(items, desc=f"Sorting {source_dir.name}"):
        src_path = source_dir / filename

        if not src_path.exists():
            skipped += 1
            continue

        label = "real" if info.get("label") == "REAL" else "fake"
        dest_path = dest_dir / label / filename

        if dest_path.exists():
            continue  # already copied (e.g. re-running after a partial run)

        shutil.copy2(src_path, dest_path)

        if label == "real":
            real_count += 1
        else:
            fake_count += 1

    print(f"Done. Copied {real_count} real, {fake_count} fake "
          f"({skipped} listed in metadata.json but missing on disk).")

    if balance and max_videos is not None and real_count < max_videos // 2:
        print(
            f"NOTE: only found {real_count} real videos available in this "
            f"source folder (wanted {max_videos // 2}). DFDC has far fewer "
            "real videos than fake ones per part -- if you need more real "
            "videos, run this script again against additional DFDC parts "
            "with the same --dest, and it will keep adding to what's "
            "already there without duplicating."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, required=True,
        help="A DFDC part folder containing metadata.json and .mp4 files.",
    )
    parser.add_argument(
        "--dest", type=str, required=True,
        help="Destination folder -- real/ and fake/ subfolders are created here.",
    )
    parser.add_argument(
        "--max_videos", type=int, default=None,
        help="Cap the total number of videos copied. With balancing on "
        "(the default), this is split evenly between real and fake.",
    )
    parser.add_argument(
        "--no_balance", action="store_true",
        help="Disable balancing -- just take the first --max_videos entries "
        "in metadata.json order (the old behavior). Not recommended: DFDC's "
        "natural order is heavily skewed toward FAKE videos.",
    )
    args = parser.parse_args()

    organize_dfdc(
        Path(args.source),
        Path(args.dest),
        max_videos=args.max_videos,
        balance=not args.no_balance,
    )


if __name__ == "__main__":
    main()