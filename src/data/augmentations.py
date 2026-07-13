"""
SimCLR augmentation pipeline adapted for chest X-rays (grayscale medical images).

Key differences from standard natural-image SimCLR:
  - Single-channel (L-mode) grayscale input — no saturation/hue jitter
  - Gaussian blur kernel size adjusted for 224x224 images
  - No RandomGrayscale (already grayscale)
  - Lighter crop scale preserved to keep pathological regions visible

Normalization stats:
  Set normalize_mean/normalize_std to "auto" in your config to load
  dataset-specific stats from data/processed/norm_stats.json (computed by
  src.data.compute_norm_stats). Falls back to ImageNet values if the file
  doesn't exist.
"""

import json
import os
import random

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# ImageNet grayscale fallbacks
_IMAGENET_MEAN = [0.485]
_IMAGENET_STD = [0.229]


def _resolve_norm_stats(config: dict) -> tuple[list[float], list[float]]:
    """
    Resolve normalization mean/std from config.

    If the values are "auto", load from data/processed/norm_stats.json.
    Falls back to ImageNet grayscale values if the file doesn't exist.
    """
    aug_cfg = config.get("augmentation", config)
    mean = aug_cfg.get("normalize_mean", _IMAGENET_MEAN)
    std = aug_cfg.get("normalize_std", _IMAGENET_STD)

    if mean == "auto" or std == "auto":
        processed_dir = config.get("data", {}).get("processed_dir", "data/processed")
        stats_path = os.path.join(processed_dir, "norm_stats.json")
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            mean = stats["mean"]
            std = stats["std"]
            print(f"Using dataset-specific normalization: mean={mean}, std={std}")
        else:
            mean = _IMAGENET_MEAN
            std = _IMAGENET_STD
            print(
                f"norm_stats.json not found at {stats_path}, "
                f"falling back to ImageNet values: mean={mean}, std={std}. "
                f"Run 'python -m src.data.compute_norm_stats' to compute dataset stats."
            )

    return mean, std


class SimCLRAugmentation:
    """Produce two independently augmented views of the same image for contrastive learning."""

    def __init__(self, config: dict):
        aug_cfg = config.get("augmentation", config)
        image_size = config.get("data", {}).get("image_size", config.get("image_size", 224))
        s = aug_cfg["color_jitter_strength"]
        mean, std = _resolve_norm_stats(config)

        # Kernel size must be odd and > 0; clamp to valid range
        k_min = aug_cfg.get("gaussian_blur_kernel_min", 3)
        k_max = aug_cfg.get("gaussian_blur_kernel_max", 23)
        # Pick a random odd kernel size at call time via a custom transform
        self._k_min = k_min if k_min % 2 == 1 else k_min + 1
        self._k_max = k_max if k_max % 2 == 1 else k_max - 1

        self._base_transforms = T.Compose([
            T.RandomResizedCrop(
                size=image_size,
                scale=tuple(aug_cfg["random_resized_crop_scale"]),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=aug_cfg["horizontal_flip_prob"]),
            T.RandomRotation(degrees=aug_cfg.get("rotation_degrees", 10)),
            T.RandomApply(
                [T.ColorJitter(brightness=0.8 * s, contrast=0.8 * s, saturation=0, hue=0)],
                p=aug_cfg["color_jitter_prob"],
            ),
            T.RandomApply(
                [_RandomGaussianBlur(k_min=self._k_min, k_max=self._k_max)],
                p=aug_cfg["gaussian_blur_prob"],
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        """Return two independently augmented views of `image`."""
        return self._base_transforms(image), self._base_transforms(image)


class FinetuneAugmentation:
    """Standard augmentation for supervised fine-tuning."""

    def __init__(self, config: dict, train: bool = True):
        aug_cfg = config.get("augmentation", config)
        image_size = config.get("data", {}).get("image_size", config.get("image_size", 224))
        mean, std = _resolve_norm_stats(config)

        if train:
            cj_prob = aug_cfg.get("color_jitter_prob", 0.0)
            cj_s = aug_cfg.get("color_jitter_strength", 0.0)
            erase_prob = aug_cfg.get("random_erasing_prob", 0.0)

            transforms = [
                T.RandomResizedCrop(
                    size=image_size,
                    scale=tuple(aug_cfg["random_resized_crop_scale"]),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=aug_cfg["horizontal_flip_prob"]),
                T.RandomRotation(degrees=aug_cfg.get("rotation_degrees", 10)),
            ]
            # Brightness/contrast jitter (no saturation/hue for grayscale)
            if cj_prob > 0 and cj_s > 0:
                transforms.append(
                    T.RandomApply(
                        [T.ColorJitter(brightness=cj_s, contrast=cj_s, saturation=0, hue=0)],
                        p=cj_prob,
                    )
                )
            transforms += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
            # Random erasing operates on tensors, so it goes after ToTensor/Normalize
            if erase_prob > 0:
                transforms.append(T.RandomErasing(p=erase_prob))
            self.transform = T.Compose(transforms)
        else:
            self.transform = T.Compose([
                T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __call__(self, image):
        return self.transform(image)


class _RandomGaussianBlur:
    """Apply Gaussian blur with a randomly chosen odd kernel size."""

    def __init__(self, k_min: int = 3, k_max: int = 23):
        self.k_min = k_min
        self.k_max = k_max

    def __call__(self, img):
        k = random.randrange(self.k_min, self.k_max + 1, 2)  # odd values only
        return T.functional.gaussian_blur(img, kernel_size=k, sigma=(0.1, 2.0))
