from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from src.model import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_t = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_t, val_t


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    img_size: int,
    num_workers: int = 2,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    train_t, val_t = build_transforms(img_size)

    if val_dir.exists():
        train_ds = datasets.ImageFolder(root=str(train_dir), transform=train_t)
        val_ds = datasets.ImageFolder(root=str(val_dir), transform=val_t)
    else:
        full_ds = datasets.ImageFolder(root=str(train_dir), transform=None)
        n_total = len(full_ds)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        generator = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(full_ds, [n_train, n_val], generator=generator)
        # Attach transforms post-split
        train_subset.dataset.transform = train_t
        val_subset.dataset.transform = val_t
        train_ds, val_ds = train_subset, val_subset

    class_to_idx = train_ds.dataset.class_to_idx if hasattr(train_ds, "dataset") else train_ds.class_to_idx

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_to_idx 