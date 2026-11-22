"""Phase 1: the BloodMNIST data pipeline.

BloodMNIST: blood-cell microscopy images, 8 classes, stored at 64x64 RGB.
facebook/vit-mae-base was pretrained on 224x224 ImageNet images with 16x16
patches (14x14 = 196 patches), normalised with ImageNet mean/std.

Docs you'll want open:
  * torchvision transforms: https://pytorch.org/vision/stable/transforms.html
  * DataLoader:             https://pytorch.org/docs/stable/data.html
  * medmnist:               `uv run python -c "from medmnist import BloodMNIST; help(BloodMNIST)"`
"""

from __future__ import annotations

from typing import Callable

import torch
from medmnist import INFO, BloodMNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SOURCE_SIZE = 64  # BloodMNIST resolution we download (the assignment says 64x64)
CLASS_NAMES = [INFO["bloodmnist"]["label"][str(i)] for i in range(len(INFO["bloodmnist"]["label"]))]
NUM_CLASSES = len(CLASS_NAMES)  # 8


def build_transform(img_size: int = 224, augment: bool = False) -> Callable:
    """PIL image (64x64 RGB) -> float32 tensor of shape (3, img_size, img_size), ImageNet-normalised.

    augment=False: a deterministic resize to (img_size, img_size). Use it for probing and visualisation.
    augment=True:  light, label-free augmentation for MAE training: a random resized crop
                   (keep at least half the area) plus horizontal and vertical flips
                   (a blood cell has no "up").
    """
    raise NotImplementedError("Phase 1: build_transform")


def label_to_int(y) -> int:
    """medmnist returns labels as numpy arrays of shape (1,). Turn them into a plain int.
    (A module-level function, not a lambda, so DataLoader workers can pickle it.)"""
    raise NotImplementedError("Phase 1: label_to_int")


def build_dataset(split: str, img_size: int = 224, augment: bool = False) -> Dataset:
    """BloodMNIST split ('train' | 'val' | 'test') at SOURCE_SIZE, with build_transform
    as the transform and label_to_int as the target_transform. Download if needed."""
    raise NotImplementedError("Phase 1: build_dataset")


def build_loader(
    split: str,
    batch_size: int = 64,
    img_size: int = 224,
    for_training: bool = False,
    augment: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """for_training=True  -> shuffle every epoch, drop the last incomplete batch.
    for_training=False -> fixed order, keep every sample (feature extraction must see all N images,
                          in order, so features line up with labels).
    Pin memory when CUDA is available. Keep workers alive between epochs when num_workers > 0."""
    raise NotImplementedError("Phase 1: build_loader")
