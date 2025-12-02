"""
HRNet backbone for high-resolution feature extraction.

HRNet maintains high-resolution representations throughout the network,
which is critical for precise crease detection and junction localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import timm


class HRNetBackbone(nn.Module):
    """
    HRNet backbone for crease pattern detection.

    Uses timm's HRNet implementation with multi-scale feature fusion.
    Maintains high-resolution features (stride 4) for precise localization.

    Output stride: 4 (256×256 features for 1024×1024 input)
    """

    # Feature dimensions for different HRNet variants (from timm's features_only output)
    # These are the actual output dimensions from timm, not the HRNet stage widths
    FEATURE_DIMS = {
        "hrnet_w18": [64, 128, 256, 512],
        "hrnet_w18_small": [64, 128, 256, 512],
        "hrnet_w18_small_v2": [64, 128, 256, 512],
        "hrnet_w32": [64, 128, 256, 512],
        "hrnet_w40": [64, 128, 256, 512],
        "hrnet_w44": [64, 128, 256, 512],
        "hrnet_w48": [64, 128, 256, 512],
        "hrnet_w64": [64, 128, 256, 512],
    }

    def __init__(
        self,
        variant: str = "hrnet_w32",
        pretrained: bool = True,
        output_stride: int = 4,
        freeze_bn: bool = False,
    ):
        """
        Initialize HRNet backbone.

        Args:
            variant: HRNet variant (e.g., 'hrnet_w32', 'hrnet_w48')
            pretrained: Whether to load ImageNet pretrained weights
            output_stride: Desired output stride (4 is recommended)
            freeze_bn: Whether to freeze batch normalization layers
        """
        super().__init__()

        self.variant = variant
        self.output_stride = output_stride

        # Get feature dimensions for this variant
        if variant not in self.FEATURE_DIMS:
            raise ValueError(
                f"Unknown HRNet variant: {variant}. "
                f"Available: {list(self.FEATURE_DIMS.keys())}"
            )
        self.feature_dims = self.FEATURE_DIMS[variant]

        # Load HRNet from timm
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # All 4 stages
        )

        # Total output channels when concatenating all scales
        self.out_channels = sum(self.feature_dims)

        if freeze_bn:
            self._freeze_bn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through HRNet.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            High-resolution features (B, C, H/4, W/4)
            where C = sum of all stage channels
        """
        # Get multi-scale features from all stages
        features = self.backbone(x)  # List of 4 feature maps

        # Upsample all to the highest resolution (stride 4) and concatenate
        target_size = features[0].shape[2:]  # Highest resolution

        upsampled = [features[0]]
        for feat in features[1:]:
            upsampled.append(
                F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        # Concatenate along channel dimension
        return torch.cat(upsampled, dim=1)

    def get_feature_info(self) -> List[dict]:
        """Get information about feature maps at each stage."""
        return [
            {"channels": c, "stride": 2 ** (i + 2)}
            for i, c in enumerate(self.feature_dims)
        ]

    def _freeze_bn(self) -> None:
        """Freeze batch normalization layers."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def get_trainable_params(
        self, lr_mult: float = 0.1
    ) -> List[dict]:
        """
        Get parameters with optional learning rate multiplier.

        Useful for fine-tuning with lower LR for pretrained backbone.

        Args:
            lr_mult: Learning rate multiplier for backbone params

        Returns:
            List of param groups for optimizer
        """
        return [{"params": self.parameters(), "lr_mult": lr_mult}]


def build_backbone(
    name: str = "hrnet_w32",
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Build a backbone network.

    Args:
        name: Backbone name (currently only HRNet variants supported)
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for the backbone

    Returns:
        Backbone module
    """
    if name.startswith("hrnet"):
        return HRNetBackbone(variant=name, pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown backbone: {name}")
