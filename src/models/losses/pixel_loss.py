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
            alpha = [0.1, 0.3, 0.3, 0.15, 0.15]

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


class JunctionBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss for junction heatmap.

    More commonly used than MSE for heatmap regression in modern
    keypoint detection (e.g., pose estimation, corner detection).
    Uses pos_weight to handle class imbalance.
    """

    def __init__(
        self,
        pos_weight: float = 10.0,
    ):
        """
        Initialize junction BCE loss.

        Args:
            pos_weight: Weight for positive (junction) pixels to handle imbalance
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute junction BCE loss.

        Args:
            pred: (B, 1, H, W) predicted heatmap (after sigmoid, values 0-1)
            target: (B, 1, H, W) or (B, H, W) ground truth heatmap

        Returns:
            Scalar loss value
        """
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        # Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        # With pos_weight: -[pos_weight*y*log(p) + (1-y)*log(1-p)]
        bce = -(
            self.pos_weight * target * torch.log(pred)
            + (1 - target) * torch.log(1 - pred)
        )

        return bce.mean()


class JunctionFocalLoss(nn.Module):
    """
    Binary focal loss for junction heatmap detection.

    Focal loss helps with extreme class imbalance by down-weighting
    easy negatives and focusing on hard examples. This is better than
    weighted MSE for junction detection where background dominates.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: float = 10.0,
    ):
        """
        Initialize junction focal loss.

        Args:
            alpha: Balance factor for positive class (typically 0.25)
            gamma: Focusing parameter - higher means more focus on hard examples
            pos_weight: Additional weight multiplier for positive pixels
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute junction focal loss.

        Args:
            pred: (B, 1, H, W) predicted heatmap (after sigmoid, values 0-1)
            target: (B, 1, H, W) or (B, H, W) ground truth heatmap

        Returns:
            Scalar loss value
        """
        # Ensure target has channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)

        # Binary cross entropy components
        bce_pos = -target * torch.log(pred)
        bce_neg = -(1 - target) * torch.log(1 - pred)

        # Focal weights: (1 - p_t)^gamma
        # For positives: p_t = pred, so weight = (1 - pred)^gamma
        # For negatives: p_t = 1 - pred, so weight = pred^gamma
        focal_weight_pos = (1 - pred) ** self.gamma
        focal_weight_neg = pred ** self.gamma

        # Alpha weighting (higher alpha for positives)
        # Also apply additional pos_weight for extreme imbalance
        alpha_pos = self.alpha * self.pos_weight
        alpha_neg = 1 - self.alpha

        # Combined focal loss
        loss = alpha_pos * focal_weight_pos * bce_pos + alpha_neg * focal_weight_neg * bce_neg

        return loss.mean()


class PixelLoss(nn.Module):
    """
    Combined pixel-level loss for crease pattern detection.

    Combines segmentation, orientation, and junction losses with configurable weights.
    """

    def __init__(
        self,
        seg_weight: float = 1.0,
        orient_weight: float = 0.5,
        junction_weight: float = 1.0,
        seg_alpha: Optional[List[float]] = None,
        seg_gamma: float = 2.0,
        junction_pos_weight: float = 10.0,
        junction_loss_type: str = "mse",
        junction_focal: bool = False,  # Deprecated, use junction_loss_type
        junction_focal_gamma: float = 2.0,
    ):
        """
        Initialize combined pixel loss.

        Args:
            seg_weight: Weight for segmentation loss
            orient_weight: Weight for orientation loss
            junction_weight: Weight for junction loss
            seg_alpha: Per-class weights for segmentation
            seg_gamma: Focal loss gamma for segmentation
            junction_pos_weight: Positive weight for junction loss
            junction_loss_type: One of "mse", "bce", or "focal"
            junction_focal: Deprecated - use junction_loss_type="focal" instead
            junction_focal_gamma: Gamma for junction focal loss (if enabled)
        """
        super().__init__()

        self.seg_weight = seg_weight
        self.orient_weight = orient_weight
        self.junction_weight = junction_weight

        self.seg_loss = SegmentationLoss(alpha=seg_alpha, gamma=seg_gamma)
        self.orient_loss = OrientationLoss()

        # Handle deprecated junction_focal parameter
        if junction_focal:
            junction_loss_type = "focal"

        # Choose junction loss type
        if junction_loss_type == "focal":
            self.junction_loss = JunctionFocalLoss(
                alpha=0.25,
                gamma=junction_focal_gamma,
                pos_weight=junction_pos_weight,
            )
        elif junction_loss_type == "bce":
            self.junction_loss = JunctionBCELoss(pos_weight=junction_pos_weight)
        else:  # mse (default)
            self.junction_loss = JunctionLoss(pos_weight=junction_pos_weight)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined pixel loss.

        Args:
            outputs: Model outputs with 'segmentation', 'orientation', 'junction'
            targets: Ground truth with 'segmentation', 'orientation', 'junction_heatmap'

        Returns:
            Dictionary with:
            - 'total': Combined loss
            - 'seg': Segmentation loss
            - 'orient': Orientation loss
            - 'junction': Junction loss
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

        # Junction loss
        junction_loss = self.junction_loss(
            outputs["junction"],
            targets["junction_heatmap"],
        )

        # Combined loss
        total_loss = (
            self.seg_weight * seg_loss
            + self.orient_weight * orient_loss
            + self.junction_weight * junction_loss
        )

        return {
            "total": total_loss,
            "seg": seg_loss,
            "orient": orient_loss,
            "junction": junction_loss,
        }
