"""CPLineNet: dense evidence model for roadmap Phase 3."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import HRNetBackbone


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyHighResolutionBackbone(nn.Module):
    """Small local-smoke backbone that preserves a stride-4 feature grid."""

    out_channels = 128

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.stem = ConvBlock(3, width)
        self.down1 = ConvBlock(width, width * 2, stride=2)
        self.down2 = ConvBlock(width * 2, width * 4, stride=2)
        self.refine = nn.Sequential(
            ConvBlock(width * 4, width * 4),
            ConvBlock(width * 4, width * 4),
        )
        self.out_channels = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.down2(self.down1(self.stem(x))))


class CPLineNet(nn.Module):
    """Predict dense line, angle, junction, offset, and assignment fields."""

    def __init__(
        self,
        *,
        backbone: str = "tiny",
        pretrained: bool = False,
        hidden_channels: int = 128,
        output_stride: int = 4,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.output_stride = output_stride
        if backbone == "tiny":
            self.backbone = TinyHighResolutionBackbone(width=max(16, hidden_channels // 4))
        elif backbone.startswith("hrnet"):
            self.backbone = HRNetBackbone(
                variant=backbone,
                pretrained=pretrained,
                output_stride=output_stride,
            )
        else:
            raise ValueError(f"Unsupported CPLineNet backbone: {backbone}")

        in_channels = int(self.backbone.out_channels)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.line_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.angle_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.junction_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.assignment_head = nn.Conv2d(hidden_channels, 4, kernel_size=1)

    def forward(self, images: torch.Tensor, *, return_features: bool = False) -> dict[str, torch.Tensor]:
        target_size = images.shape[-2:]
        features = self.backbone(images)
        shared = self.shared(features)
        line_logits = self._upsample(self.line_head(shared), target_size)
        angle = F.normalize(self._upsample(self.angle_head(shared), target_size), dim=1, eps=1e-6)
        junction_logits = self._upsample(self.junction_head(shared), target_size)
        junction_offset = torch.clamp(self._upsample(self.offset_head(shared), target_size), -0.5, 0.5)
        assignment_logits = self._upsample(self.assignment_head(shared), target_size)
        outputs: dict[str, torch.Tensor] = {
            "line_logits": line_logits,
            "angle": angle,
            "junction_logits": junction_logits,
            "junction_offset": junction_offset,
            "assignment_logits": assignment_logits,
        }
        if return_features:
            outputs["features"] = features
        return outputs

    @staticmethod
    def _upsample(tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    def get_param_groups(self, base_lr: float, backbone_lr_mult: float = 1.0) -> list[dict[str, Any]]:
        return [
            {"params": self.backbone.parameters(), "lr": base_lr * backbone_lr_mult, "name": "backbone"},
            {
                "params": list(self.shared.parameters())
                + list(self.line_head.parameters())
                + list(self.angle_head.parameters())
                + list(self.junction_head.parameters())
                + list(self.offset_head.parameters())
                + list(self.assignment_head.parameters()),
                "lr": base_lr,
                "name": "heads",
            },
        ]
