from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_RESNETS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}


def build_resnet(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name not in SUPPORTED_RESNETS:
        supported = ", ".join(sorted(SUPPORTED_RESNETS))
        raise ValueError(f"Unsupported model '{name}'. Supported models: {supported}")

    weights = "DEFAULT" if pretrained else None
    model = SUPPORTED_RESNETS[name](weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def normalize_state_dict(checkpoint: dict) -> OrderedDict:
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not state_dict:
        raise ValueError("Empty checkpoint state_dict")
    if next(iter(state_dict)).startswith("module."):
        state_dict = OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
    return state_dict


def load_resnet_checkpoint(
    checkpoint_path: str | Path,
    architecture: str,
    num_classes: int,
    device: torch.device | str = "cpu",
) -> nn.Module:
    model = build_resnet(architecture, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(normalize_state_dict(checkpoint), strict=True)
    model.to(device)
    model.eval()
    return model


class ResNetWithFeatures(nn.Module):
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def forward(self, x):
        feature_maps = self.features(x)
        x = self.avgpool(feature_maps)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits, feature_maps
