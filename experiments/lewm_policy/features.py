"""Frozen LeWM / ResNet visual features fused with SO-101 proprioception."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def _import_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import LeWM source file: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_lewm_backbone(
    lewm_repo: str | Path,
    checkpoint_path: str | Path,
) -> tuple[nn.Module, int]:
    """Load the checkpoint's trained image encoder and training projection head."""
    root = Path(lewm_repo).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not (root / "models" / "encoder.py").is_file():
        raise FileNotFoundError(f"LeWM encoder source not found under {root}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"LeWM checkpoint not found: {checkpoint_file}")

    tag = f"lewm_{abs(hash(str(root))):x}"
    encoder_module = _import_from_path(tag + "_encoder", root / "models" / "encoder.py")
    predictor_module = _import_from_path(
        tag + "_predictor", root / "models" / "predictor2.py"
    )
    encoder = encoder_module.Encoder(
        img_size=224, patch=16, in_ch=3, dim=192, depth=12, heads=3, out_dim=192
    )
    projection = predictor_module.ProjectionHead(192)

    # The checkpoint is a trusted local project artifact and contains the full
    # training state dictionary, not a TorchScript module.
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if "encoder" not in checkpoint or "projector" not in checkpoint:
        raise KeyError("Expected checkpoint keys 'encoder' and 'projector'")
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    projection.load_state_dict(checkpoint["projector"], strict=True)

    class LeWMVisualBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.projection = projection

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.projection(self.encoder(images))

        def train(self, mode: bool = True):
            super().train(mode)
            self.encoder.eval()
            self.projection.eval()
            return self

    backbone = LeWMVisualBackbone()
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone, 192


def make_resnet_backbone(*, imagenet_weights: bool = True) -> tuple[nn.Module, int]:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except Exception as exc:
        raise RuntimeError("The ResNet comparison requires a working torchvision install") from exc
    weights = ResNet18_Weights.DEFAULT if imagenet_weights else None
    network = resnet18(weights=weights)
    network.fc = nn.Identity()

    class ImageNetResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = network
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.network((images - self.mean) / self.std)

        def train(self, mode: bool = True):
            super().train(mode)
            self.network.eval()
            return self

    backbone = ImageNetResNet()
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone, 512


class VisionProprioFeatures(BaseFeaturesExtractor):
    """LeWM/ResNet image features plus a small trainable robot-state encoder."""

    def __init__(
        self,
        observation_space,
        *,
        visual_backbone: str = "lewm",
        lewm_repo: str | None = None,
        lewm_checkpoint: str | None = None,
        imagenet_weights: bool = True,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        if visual_backbone == "lewm":
            if lewm_repo is None or lewm_checkpoint is None:
                raise ValueError("lewm_repo and lewm_checkpoint are required for LeWM")
            self.visual, visual_dim = make_lewm_backbone(lewm_repo, lewm_checkpoint)
        elif visual_backbone == "resnet18":
            self.visual, visual_dim = make_resnet_backbone(
                imagenet_weights=imagenet_weights
            )
        else:
            raise ValueError("visual_backbone must be 'lewm' or 'resnet18'")

        self.state_net = nn.Sequential(
            nn.Linear(13, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()
        )
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + 64, features_dim), nn.LayerNorm(features_dim), nn.Tanh()
        )
        self.visual.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.visual.eval()
        return self

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations["image"].float().div(255.0)
        state = observations["proprio"].float()
        with torch.no_grad():
            visual_features = self.visual(image)
        state_features = self.state_net(state)
        return self.fusion(torch.cat((visual_features, state_features), dim=1))

