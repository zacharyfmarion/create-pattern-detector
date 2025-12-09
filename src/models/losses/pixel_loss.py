"""
Pixel-level loss functions for crease pattern detection.

Includes:
- Focal loss for class-imbalanced segmentation
- Cosine similarity loss for orientation field
- Weighted MSE loss for junction heatmap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class SegmentationLoss(nn.Module):
    """
    Focal loss for class-imbalanced crease segmentation.

    Focal loss down-weights well-classified examples and focuses on hard cases.
    Particularly useful for crease patterns where background dominates.
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Per-class weights. Default: [0.1, 0.3, 0.3, 0.15, 0.15] for
                   [BG, M, V, B, U] - higher weight for crease classes
            gamma: Focusing parameter (higher = more focus on hard examples)
            ignore_index: Label to ignore in loss computation
        """
        super().__init__()

        if alpha is None:
            # Default weights: low for background, higher for creases
            # Border (B) is rarest non-BG class (~3%) so gets higher weight
            # M (~4.5%) and V (~5%) are more common
            alpha = [0.1, 0.25, 0.25, 0.30, 0.10]

        self.register_buffer("alpha", torch.tensor(alpha))
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: (B, C, H, W) predicted logits
            targets: (B, H, W) ground truth class labels

        Returns:
            Scalar loss value
        """
        B, C, H, W = logits.shape

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Flatten for easier indexing
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = targets.reshape(-1)  # (B*H*W,)

        # Create valid mask
        valid_mask = targets_flat != self.ignore_index

        # Get valid predictions and targets
        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]

        if len(targets_valid) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute cross entropy (without reduction)
        ce_loss = F.cross_entropy(
            logits_valid,
            targets_valid,
            reduction="none",
        )

        # Get probability of correct class
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)
        probs_valid = probs_flat[valid_mask]
        p_t = probs_valid.gather(1, targets_valid.unsqueeze(1)).squeeze(1)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight: per-class weighting
        alpha = self.alpha.to(logits.device)
        alpha_weight = alpha[targets_valid]

        # Combined loss
        loss = alpha_weight * focal_weight * ce_loss

        return loss.mean()


class OrientationLoss(nn.Module):
    """
    Cosine similarity loss for orientation field.

    Computes loss only at crease pixels (non-background).
    """

    def __init__(self, eps: float = 1e-6):
        """
        Initialize orientation loss.

        Args:
            eps: Small value for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute orientation loss.

        Args:
            pred: (B, 2, H, W) predicted (cos θ, sin θ)
            target: (B, 2, H, W) ground truth (cos θ, sin θ)
            mask: (B, H, W) boolean mask of crease pixels

        Returns:
            Scalar loss value
        """
        # Normalize predictions
        pred = F.normalize(pred, dim=1, eps=self.eps)
        target = F.normalize(target, dim=1, eps=self.eps)

        # Compute cosine similarity: sum of element-wise product
        # cos_sim = pred · target (dot product along channel dim)
        cos_sim = (pred * target).sum(dim=1)  # (B, H, W)

        # IMPORTANT: Orientation is bidirectional - a line at angle θ is the
        # same as at angle θ+180°. So we take abs(cos_sim) to ignore direction.
        # This makes cos_sim range from 0 to 1 (1 = aligned, 0 = perpendicular)
        cos_sim = torch.abs(cos_sim)

        # Loss = 1 - |cos_sim| (ranges from 0 to 1)
        loss = 1 - cos_sim

        # Apply mask (only compute at crease pixels)
        mask_float = mask.float()
        masked_loss = loss * mask_float

        # Average over crease pixels
        num_crease = mask_float.sum()
        if num_crease > 0:
            return masked_loss.sum() / num_crease
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class JunctionLoss(nn.Module):
    """
    Weighted MSE loss for junction heatmap.

    Uses higher weight for positive (junction) pixels to handle imbalance.
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
        reduction: str = "mean",
    ):
        """
        Initialize junction loss.

        Args:
            pos_weight: Weight for positive (junction) pixels
            reduction: Loss reduction method
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute junction loss.

        Args:
            pred: (B, 1, H, W) predicted heatmap (after sigmoid)
            target: (B, H, W) ground truth heatmap

        Returns:
            Scalar loss value
        """
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Compute MSE
        mse = (pred - target) ** 2

        # Create weight map: higher weight for positive pixels
        weight = torch.ones_like(target)
        weight[target > 0.5] = self.pos_weight

        # Weighted loss
        weighted_loss = mse * weight

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class JunctionOffsetLoss(nn.Module):
    """
    L1 loss for sub-pixel junction offset prediction.

    Computes loss only at junction anchor pixels (sparse supervision).
    Predicts (dx, dy) offset from pixel center to true junction location.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute junction offset loss.

        Args:
            pred: (B, 2, H, W) predicted offsets (dx, dy)
            target: (B, 2, H, W) ground truth offsets
            mask: (B, H, W) boolean mask of junction pixels

        Returns:
            Scalar loss value
        """
        # Expand mask to match pred shape (B, 1, H, W) -> (B, 2, H, W)
        mask_expanded = mask.unsqueeze(1).expand_as(pred).float()

        # L1 loss at masked pixels
        l1_loss = torch.abs(pred - target)
        masked_loss = l1_loss * mask_expanded

        # Average over junction pixels only
        num_junctions = mask_expanded.sum()
        if num_junctions > 0:
            return masked_loss.sum() / num_junctions
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


class PixelLoss(nn.Module):
    """
    Combined pixel-level loss for crease pattern detection.

    Combines segmentation, orientation, junction heatmap, and junction offset losses.
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        orient_weight: float = 0.5,
        junction_weight: float = 1.0,
        junction_offset_weight: float = 0.5,
        seg_alpha: Optional[List[float]] = None,
        seg_gamma: float = 2.0,
        junction_pos_weight: float = 10.0,
    ):
        """
        Initialize combined pixel loss.

        Args:
            seg_weight: Weight for segmentation loss
            orient_weight: Weight for orientation loss
            junction_weight: Weight for junction heatmap loss
            junction_offset_weight: Weight for junction offset loss
            seg_alpha: Per-class weights for segmentation
            seg_gamma: Focal loss gamma for segmentation
            junction_pos_weight: Positive weight for junction heatmap loss
        """
        super().__init__()

        self.seg_weight = seg_weight
        self.orient_weight = orient_weight
        self.junction_weight = junction_weight
        self.junction_offset_weight = junction_offset_weight

        self.seg_loss = SegmentationLoss(alpha=seg_alpha, gamma=seg_gamma)
        self.orient_loss = OrientationLoss()
        self.junction_loss = JunctionLoss(pos_weight=junction_pos_weight)
        self.junction_offset_loss = JunctionOffsetLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined pixel loss.

        Args:
            outputs: Model outputs with 'segmentation', 'orientation', 'junction', 'junction_offset'
            targets: Ground truth with 'segmentation', 'orientation', 'junction_heatmap',
                     'junction_offset', 'junction_mask'

        Returns:
            Dictionary with:
            - 'total': Combined loss
            - 'seg': Segmentation loss
            - 'orient': Orientation loss
            - 'junction': Junction heatmap loss
            - 'junction_offset': Junction offset loss
        """
        # Segmentation loss
        seg_loss = self.seg_loss(outputs["segmentation"], targets["segmentation"])

        # Orientation loss (only at crease pixels)
        # Crease mask: any non-background class (classes 1-4)
        crease_mask = targets["segmentation"] > 0
        orient_loss = self.orient_loss(
            outputs["orientation"],
            targets["orientation"],
            crease_mask,
        )

        # Junction heatmap loss
        junction_loss = self.junction_loss(
            outputs["junction"],
            targets["junction_heatmap"],
        )

        # Junction offset loss (only at junction anchor pixels)
        junction_offset_loss = self.junction_offset_loss(
            outputs["junction_offset"],
            targets["junction_offset"],
            targets["junction_mask"],
        )

        # Combined loss
        total_loss = (
            self.seg_weight * seg_loss
            + self.orient_weight * orient_loss
            + self.junction_weight * junction_loss
            + self.junction_offset_weight * junction_offset_loss
        )

        return {
            "total": total_loss,
            "seg": seg_loss,
            "orient": orient_loss,
            "junction": junction_loss,
            "junction_offset": junction_offset_loss,
        }
