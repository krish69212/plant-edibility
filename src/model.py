from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(arch: str = "resnet18", num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    return model


def save_checkpoint(
    model: nn.Module,
    class_to_idx: Dict[str, int],
    output_path: Path,
    arch: str,
    img_size: int,
    mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
    std: Tuple[float, float, float] = tuple(IMAGENET_STD),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "img_size": img_size,
        "mean": mean,
        "std": std,
    }
    torch.save(checkpoint, str(output_path))


def load_checkpoint(checkpoint_path: Path, map_location: Optional[torch.device] = torch.device("cpu")) -> Tuple[nn.Module, Dict[str, int], Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
    arch = checkpoint.get("arch", "resnet18")
    class_to_idx = checkpoint["class_to_idx"]

    model = build_model(arch=arch, num_classes=len(class_to_idx), pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])

    meta = {
        "img_size": checkpoint.get("img_size", 224),
        "mean": checkpoint.get("mean", IMAGENET_MEAN),
        "std": checkpoint.get("std", IMAGENET_STD),
        "arch": arch,
    }

    return model, class_to_idx, meta 