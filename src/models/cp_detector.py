"""
Main crease pattern detection model.

Combines backbone and pixel head for end-to-end crease pattern detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .backbone import HRNetBackbone
from .heads import PixelHead


class CreasePatternDetector(nn.Module):
    """
    End-to-end crease pattern detection model.

    Architecture:
    - Backbone (HRNet): Extracts high-resolution features
    - Pixel Head: Predicts segmentation, orientation, and junctions

    For Phase 2, a Graph Head can be added for GNN-based refinement.
    """

    def __init__(
        self,
        backbone: str = "hrnet_w32",
        pretrained: bool = True,
        hidden_channels: int = 256,
        num_seg_classes: int = 5,
        freeze_backbone: bool = False,
        output_stride: int = 4,
    ):
        """
        Initialize the crease pattern detector.

        Args:
            backbone: Backbone network name (e.g., 'hrnet_w32')
            pretrained: Whether to use pretrained backbone weights
            hidden_channels: Hidden dimension for pixel head
            num_seg_classes: Number of segmentation classes
            freeze_backbone: Whether to freeze backbone weights
            output_stride: Backbone output stride
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_seg_classes = num_seg_classes
        self.output_stride = output_stride

        # Build backbone
        self.backbone = HRNetBackbone(
            variant=backbone,
            pretrained=pretrained,
            output_stride=output_stride,
        )

        # Build pixel head
        self.pixel_head = PixelHead(
            in_channels=self.backbone.out_channels,
            hidden_channels=hidden_channels,
            num_seg_classes=num_seg_classes,
            output_stride=output_stride,
        )

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Input images (B, 3, H, W)
            return_features: Whether to return backbone features

        Returns:
            Dictionary with:
            - 'segmentation': (B, num_classes, H, W) logits
            - 'orientation': (B, 2, H, W) direction field
            - 'junction': (B, 1, H, W) heatmap
            - 'features': (B, C, H/s, W/s) backbone features (if return_features=True)
        """
        # Get image dimensions
        _, _, H, W = images.shape

        # Extract features
        features = self.backbone(images)

        # Get pixel predictions
        outputs = self.pixel_head(features, target_size=(H, W))

        if return_features:
            outputs["features"] = features

        return outputs

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(
        self,
        base_lr: float,
        backbone_lr_mult: float = 0.1,
    ) -> list:
        """
        Get parameter groups with different learning rates.

        Useful for fine-tuning with lower LR for pretrained backbone.

        Args:
            base_lr: Base learning rate for head
            backbone_lr_mult: Learning rate multiplier for backbone

        Returns:
            List of parameter group dicts for optimizer
        """
        return [
            {
                "params": self.backbone.parameters(),
                "lr": base_lr * backbone_lr_mult,
                "name": "backbone",
            },
            {
                "params": self.pixel_head.parameters(),
                "lr": base_lr,
                "name": "pixel_head",
            },
        ]

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and post-process predictions.

        Args:
            images: Input images (B, 3, H, W)
            threshold: Threshold for junction detection

        Returns:
            Dictionary with:
            - 'seg_pred': (B, H, W) predicted class labels
            - 'seg_probs': (B, num_classes, H, W) class probabilities
            - 'orientation': (B, 2, H, W) direction field
            - 'junction': (B, 1, H, W) thresholded junctions
            - 'junction_coords': List of (N, 2) junction coordinates per image
        """
        self.eval()

        outputs = self.forward(images)

        # Segmentation predictions
        seg_probs = F.softmax(outputs["segmentation"], dim=1)
        seg_pred = seg_probs.argmax(dim=1)

        # Junction detection
        junction = outputs["junction"]
        junction_binary = (junction > threshold).float()

        # Extract junction coordinates
        junction_coords = []
        for b in range(images.shape[0]):
            coords = torch.nonzero(junction_binary[b, 0] > 0, as_tuple=False)
            # Convert from (y, x) to (x, y)
            if len(coords) > 0:
                coords = coords[:, [1, 0]].float()
            junction_coords.append(coords)

        return {
            "seg_pred": seg_pred,
            "seg_probs": seg_probs,
            "orientation": outputs["orientation"],
            "junction": junction,
            "junction_binary": junction_binary,
            "junction_coords": junction_coords,
        }


def build_model(
    config: dict = None,
    **kwargs,
) -> CreasePatternDetector:
    """
    Build a crease pattern detector model.

    Args:
        config: Configuration dictionary
        **kwargs: Override config values

    Returns:
        CreasePatternDetector model
    """
    # Default configuration
    default_config = {
        "backbone": "hrnet_w32",
        "pretrained": True,
        "hidden_channels": 256,
        "num_seg_classes": 5,
        "freeze_backbone": False,
        "output_stride": 4,
    }

    # Merge with provided config
    if config is not None:
        default_config.update(config)
    default_config.update(kwargs)

    return CreasePatternDetector(**default_config)
