"""
Pixel-level prediction head for crease pattern detection.

Outputs:
1. Segmentation: 5-class per-pixel classification (BG, M, V, B, U)
2. Orientation: 2-channel direction field (cos θ, sin θ)
3. Junction heatmap: Single-channel heatmap for vertex detection
4. Junction offset: 2-channel sub-pixel offset (dx, dy) for refined vertex positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class PixelHead(nn.Module):
    """
    Multi-task pixel-level prediction head.

    Takes high-resolution features from backbone and produces:
    - Segmentation logits: (B, 5, H, W) for 5 classes
    - Orientation field: (B, 2, H, W) for (cos θ, sin θ)
    - Junction heatmap: (B, 1, H, W) for vertex detection
    - Junction offset: (B, 2, H, W) for sub-pixel vertex refinement
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_seg_classes: int = 5,
        output_stride: int = 4,
    ):
        """
        Initialize the pixel head.

        Args:
            in_channels: Number of input channels from backbone
            hidden_channels: Hidden dimension for intermediate layers
            num_seg_classes: Number of segmentation classes (default 5: BG, M, V, B, U)
            output_stride: Backbone output stride (for upsampling)
        """
        super().__init__()

        self.output_stride = output_stride
        self.num_seg_classes = num_seg_classes

        # Shared feature refinement
        self.shared_conv = nn.Sequential(
            ConvBNReLU(in_channels, hidden_channels),
            ConvBNReLU(hidden_channels, hidden_channels),
        )

        # Segmentation branch
        self.seg_head = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, num_seg_classes, kernel_size=1),
        )

        # Orientation branch (predicts cos/sin of edge angle)
        self.orient_head = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, 2, kernel_size=1),
        )

        # Junction heatmap branch
        self.junction_head = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Junction offset branch (sub-pixel refinement, no activation)
        self.junction_offset_head = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, 2, kernel_size=1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        features: torch.Tensor,
        target_size: tuple = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through pixel head.

        Args:
            features: (B, C, H/s, W/s) backbone features
            target_size: Optional (H, W) for output size. If None, upsamples by output_stride.

        Returns:
            Dictionary with:
            - 'segmentation': (B, num_classes, H, W) logits
            - 'orientation': (B, 2, H, W) normalized direction vectors
            - 'junction': (B, 1, H, W) heatmap values in [0, 1]
            - 'junction_offset': (B, 2, H, W) sub-pixel offsets in [-0.5, 0.5]
        """
        # Shared feature processing
        x = self.shared_conv(features)

        # Get predictions at feature resolution
        seg_logits = self.seg_head(x)
        orientation = self.orient_head(x)
        junction = self.junction_head(x)
        junction_offset = self.junction_offset_head(x)

        # Normalize orientation to unit vectors
        orientation = F.normalize(orientation, dim=1, eps=1e-6)

        # Clamp junction offset to [-0.5, 0.5] range
        junction_offset = torch.clamp(junction_offset, -0.5, 0.5)

        # Compute target size for upsampling
        if target_size is None:
            _, _, h, w = features.shape
            target_size = (h * self.output_stride, w * self.output_stride)

        # Upsample to full resolution
        seg_logits = F.interpolate(
            seg_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        orientation = F.interpolate(
            orientation,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        junction = F.interpolate(
            junction,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        junction_offset = F.interpolate(
            junction_offset,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Re-normalize orientation after interpolation
        orientation = F.normalize(orientation, dim=1, eps=1e-6)

        return {
            "segmentation": seg_logits,
            "orientation": orientation,
            "junction": junction,
            "junction_offset": junction_offset,
        }


class DeepPixelHead(nn.Module):
    """
    Deeper pixel head with skip connections for better gradient flow.

    Alternative to PixelHead with residual blocks and more capacity.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_seg_classes: int = 5,
        output_stride: int = 4,
        num_blocks: int = 3,
    ):
        super().__init__()

        self.output_stride = output_stride
        self.num_seg_classes = num_seg_classes

        # Input projection
        self.input_proj = ConvBNReLU(in_channels, hidden_channels, kernel_size=1, padding=0)

        # Shared residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, hidden_channels)
            for _ in range(num_blocks)
        ])

        # Task-specific heads with separate refinement
        self.seg_refine = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2),
            ConvBNReLU(hidden_channels // 2, hidden_channels // 4),
        )
        self.seg_out = nn.Conv2d(hidden_channels // 4, num_seg_classes, kernel_size=1)

        self.orient_refine = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2),
            ConvBNReLU(hidden_channels // 2, hidden_channels // 4),
        )
        self.orient_out = nn.Conv2d(hidden_channels // 4, 2, kernel_size=1)

        self.junction_refine = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2),
            ConvBNReLU(hidden_channels // 2, hidden_channels // 4),
        )
        self.junction_out = nn.Sequential(
            nn.Conv2d(hidden_channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Junction offset branch (sub-pixel refinement)
        self.junction_offset_refine = nn.Sequential(
            ConvBNReLU(hidden_channels, hidden_channels // 2),
            ConvBNReLU(hidden_channels // 2, hidden_channels // 4),
        )
        self.junction_offset_out = nn.Conv2d(hidden_channels // 4, 2, kernel_size=1)

    def forward(
        self,
        features: torch.Tensor,
        target_size: tuple = None,
    ) -> Dict[str, torch.Tensor]:
        # Input projection
        x = self.input_proj(features)

        # Shared residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Task-specific refinement
        seg_feat = self.seg_refine(x)
        orient_feat = self.orient_refine(x)
        junction_feat = self.junction_refine(x)
        junction_offset_feat = self.junction_offset_refine(x)

        # Output predictions
        seg_logits = self.seg_out(seg_feat)
        orientation = self.orient_out(orient_feat)
        junction = self.junction_out(junction_feat)
        junction_offset = self.junction_offset_out(junction_offset_feat)

        # Normalize orientation
        orientation = F.normalize(orientation, dim=1, eps=1e-6)

        # Clamp junction offset to [-0.5, 0.5] range
        junction_offset = torch.clamp(junction_offset, -0.5, 0.5)

        # Compute target size
        if target_size is None:
            _, _, h, w = features.shape
            target_size = (h * self.output_stride, w * self.output_stride)

        # Upsample
        seg_logits = F.interpolate(seg_logits, size=target_size, mode="bilinear", align_corners=False)
        orientation = F.interpolate(orientation, size=target_size, mode="bilinear", align_corners=False)
        junction = F.interpolate(junction, size=target_size, mode="bilinear", align_corners=False)
        junction_offset = F.interpolate(junction_offset, size=target_size, mode="bilinear", align_corners=False)

        # Re-normalize after interpolation
        orientation = F.normalize(orientation, dim=1, eps=1e-6)

        return {
            "segmentation": seg_logits,
            "orientation": orientation,
            "junction": junction,
            "junction_offset": junction_offset,
        }


class ResidualBlock(nn.Module):
    """Simple residual block with two conv layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        # Skip connection (identity if channels match)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + identity)
        return out
