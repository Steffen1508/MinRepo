"""
Modeller. Alle tager input (B, 3, T, H, W) i [0, 1] og normaliserer selv.

  frame     ResNet-18 (ImageNet) på hver frame for sig, gennemsnit af logits.
            Baseline: kan kun se udseende, ikke bevægelse.
  r2plus1d  R(2+1)D-18 pretrained på Kinetics-400. Hovedmodellen.
  r3d       R3D-18 pretrained på Kinetics-400. Til sammenligning/ablation.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import (R2Plus1D_18_Weights, R3D_18_Weights,
                                      r2plus1d_18, r3d_18)

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
KINETICS = ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])


class _Norm(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.register_buffer("mean", torch.tensor(stats[0]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(stats[1]).view(1, 3, 1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class FrameBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.norm = _Norm(IMAGENET)
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, num_classes))

    def forward(self, x):
        x = self.norm(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        return self.backbone(x).view(B, T, -1).mean(1)

    def head(self):
        return self.backbone.fc

    def groups(self):
        """Lagene i rækkefølge fra input til output, til delvis frysning."""
        b = self.backbone
        return [nn.Sequential(b.conv1, b.bn1), b.layer1, b.layer2, b.layer3, b.layer4]


class VideoNet(nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.norm = _Norm(KINETICS)
        if arch == "r2plus1d":
            self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT if pretrained else None)
        elif arch == "r3d":
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(arch)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, num_classes))

    def forward(self, x):
        return self.backbone(self.norm(x))

    def head(self):
        return self.backbone.fc

    def groups(self):
        b = self.backbone
        return [b.stem, b.layer1, b.layer2, b.layer3, b.layer4]


def build_model(name: str, num_classes: int, pretrained: bool = True,
                freeze: str = "none") -> nn.Module:
    """freeze:
        none     alt trænes
        backbone kun klassifikationshovedet trænes (lineær probe)
        partial  layer4 + hoved trænes, resten frosset
    """
    if name == "frame":
        m = FrameBaseline(num_classes, pretrained)
    elif name in ("r2plus1d", "r3d"):
        m = VideoNet(name, num_classes, pretrained)
    else:
        raise ValueError(f"Ukendt model: {name}")

    if freeze != "none":
        groups = m.groups()
        frozen = groups if freeze == "backbone" else groups[:-1]
        for g in frozen:
            for p in g.parameters():
                p.requires_grad = False
            # Hold BatchNorm i eval-tilstand i frosne lag
            for mod in g.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
                    mod.train = lambda mode=True, _m=mod: _m  # ignorer .train()
    return m


def count_params(m: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train
