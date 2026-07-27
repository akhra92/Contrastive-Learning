"""
Compute per-channel mean and std of the NIH Chest X-ray14 dataset.

Uses Welford's online algorithm to avoid loading all images into memory.
Computes stats on the train/val images only — the official test split
(test_list.txt) is excluded so no test information leaks into normalisation.
Images are resized to the target image_size first.

Saves results to data/processed/norm_stats.json so training scripts can
load them automatically instead of using borrowed ImageNet values.

Usage:
    python -m src.data.compute_norm_stats
    python -m src.data.compute_norm_stats --raw_dir data/raw/nih-chest-xrays
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_norm_stats(
    image_dir: str,
    image_size: int = 224,
    max_samples: int = 0,
    exclude_files: set | None = None,
) -> dict:
    """
    Compute mean and std over .png images in image_dir.

    Uses Welford's online algorithm for numerical stability.

    Args:
        image_dir     : directory containing .png images.
        image_size    : resize images to this size before computing stats.
        max_samples   : if > 0, subsample this many images (faster for large datasets).
        exclude_files : filenames (basenames) to skip, e.g. the official test
                        split, so no test information leaks into normalisation.

    Returns:
        dict with keys 'mean' and 'std' (each a list with one float for grayscale).
    """
    paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.png"), recursive=True))
    if exclude_files:
        n_before = len(paths)
        paths = [p for p in paths if os.path.basename(p) not in exclude_files]
        print(f"Excluded {n_before - len(paths)} images (official test split)")
    if not paths:
        raise FileNotFoundError(f"No .png images found under {image_dir}")

    if max_samples > 0 and max_samples < len(paths):
        rng = np.random.RandomState(42)
        indices = rng.choice(len(paths), size=max_samples, replace=False)
        paths = [paths[i] for i in sorted(indices)]

    # Welford's online algorithm
    n_pixels = 0
    mean = 0.0
    m2 = 0.0

    for path in tqdm(paths, desc="Computing normalization stats"):
        img = Image.open(path).convert("L").resize(
            (image_size, image_size), Image.BICUBIC
        )
        pixels = np.asarray(img, dtype=np.float64) / 255.0
        flat = pixels.ravel()

        for val in [flat.mean()]:
            # Batch update: treat each image as a batch of pixels
            batch_size = flat.size
            batch_mean = flat.mean()
            batch_var = flat.var()

            delta = batch_mean - mean
            total = n_pixels + batch_size
            mean = mean + delta * batch_size / total
            m2 = m2 + batch_var * batch_size + delta**2 * n_pixels * batch_size / total
            n_pixels = total
            break  # single pass per image

    std = np.sqrt(m2 / n_pixels)

    stats = {
        "mean": [round(float(mean), 6)],
        "std": [round(float(std), 6)],
        "n_images": len(paths),
        "n_pixels": int(n_pixels),
        "image_size": image_size,
    }

    print(f"\nDataset normalization stats ({len(paths)} images, {image_size}x{image_size}):")
    print(f"  mean = {stats['mean']}")
    print(f"  std  = {stats['std']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset-specific normalization stats for NIH Chest X-ray14"
    )
    parser.add_argument("--raw_dir", default="data/raw/nih-chest-xrays")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="Max images to sample (0 = use all). 5000-10000 gives accurate estimates.",
    )
    args = parser.parse_args()

    # Find image directory
    from src.training.utils import find_image_dir
    image_dir = find_image_dir(args.raw_dir)

    # Exclude the official test split so its pixels don't leak into the stats
    exclude_files = None
    test_list = os.path.join(args.raw_dir, "test_list.txt")
    if os.path.isfile(test_list):
        with open(test_list) as f:
            exclude_files = {line.strip() for line in f if line.strip()}
    else:
        print(f"Warning: {test_list} not found — stats will include test images.")

    stats = compute_norm_stats(
        image_dir=image_dir,
        image_size=args.image_size,
        max_samples=args.max_samples,
        exclude_files=exclude_files,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "norm_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to {out_path}")
    print(f"To use in configs, set:\n"
          f"  normalize_mean: {stats['mean']}\n"
          f"  normalize_std:  {stats['std']}")


if __name__ == "__main__":
    main()
