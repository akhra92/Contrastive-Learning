"""
PyTorch Dataset classes for NIH Chest X-ray14.

Two distinct contracts:
  SimCLRDataset   — returns (view1, view2) augmented pairs; no labels.
                    Used for self-supervised pre-training.
  ChestXrayDataset — returns (image, label_vector) for supervised fine-tuning.
"""

import os
import logging
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

ALL_CLASSES = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(ALL_CLASSES)


def build_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert 'Finding Labels' column (pipe-separated) into a (N, 15) float32 matrix."""
    matrix = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    for i, findings in enumerate(df["Finding Labels"]):
        for label in str(findings).split("|"):
            label = label.strip()
            if label in ALL_CLASSES:
                matrix[i, ALL_CLASSES.index(label)] = 1.0
    return matrix


class SimCLRDataset(Dataset):
    """
    Unlabelled dataset for self-supervised pre-training.

    Accepts a list of absolute image paths and a SimCLRAugmentation callable.
    Labels are intentionally omitted. The caller is responsible for excluding
    the official test split from `image_paths` (see
    src.training.pretrain._collect_image_paths): even without labels, training
    on the test images themselves is transductive leakage.
    """

    def __init__(self, image_paths: list, augmentation):
        self.image_paths = image_paths
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("L")  # grayscale
        except (OSError, UnidentifiedImageError) as e:
            logger.warning("Skipping unreadable image %s: %s", path, e)
            return self.__getitem__((idx + 1) % len(self.image_paths))
        view1, view2 = self.augmentation(image)
        return view1, view2


class ChestXrayDataset(Dataset):
    """
    Multi-label supervised dataset for NIH Chest X-ray14.

    Args:
        df         : DataFrame with columns ['Image Index', 'Finding Labels'].
        image_dir  : Directory containing the .png images.
        transform  : FinetuneAugmentation or any callable transform.
    """

    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.labels = build_label_matrix(self.df)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Image Index"])
        try:
            image = Image.open(img_path).convert("L")
        except (OSError, UnidentifiedImageError) as e:
            logger.warning("Skipping unreadable image %s: %s", img_path, e)
            return self.__getitem__((idx + 1) % len(self.df))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
