"""
Loss functions for graph head training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GraphLoss(nn.Module):
    """
    Combined loss for graph head training.

    Components:
    1. Edge existence loss (binary cross-entropy)
    2. Edge assignment loss (cross-entropy for M/V/B/U)
    """

    def __init__(
        self,
        existence_weight: float = 1.0,
        assignment_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            existence_weight: Weight for edge existence loss
            assignment_weight: Weight for edge assignment loss
            class_weights: Optional class weights for assignment loss (4,)
            label_smoothing: Label smoothing for assignment loss
        """
        super().__init__()

        self.existence_weight = existence_weight
        self.assignment_weight = assignment_weight
        self.label_smoothing = label_smoothing

        # Register class weights as buffer
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predictions: Dictionary with:
                - 'edge_existence': (E,) logits
                - 'edge_assignment': (E, 4) logits
            targets: Dictionary with:
                - 'edge_existence': (E,) binary labels
                - 'edge_assignment': (E,) class labels (0-3)
            mask: Optional (E,) mask for valid edges

        Returns:
            Dictionary with individual losses and total loss
        """
        pred_existence = predictions['edge_existence']
        pred_assignment = predictions['edge_assignment']
        target_existence = targets['edge_existence'].float()
        target_assignment = targets['edge_assignment'].long()

        # Apply mask if provided
        if mask is not None:
            pred_existence = pred_existence[mask]
            pred_assignment = pred_assignment[mask]
            target_existence = target_existence[mask]
            target_assignment = target_assignment[mask]

        # Edge existence loss (BCE with logits)
        existence_loss = F.binary_cross_entropy_with_logits(
            pred_existence,
            target_existence,
        )

        # Edge assignment loss (only for existing edges)
        # Filter to edges that exist in ground truth
        exist_mask = target_existence > 0.5
        if exist_mask.sum() > 0:
            assignment_loss = F.cross_entropy(
                pred_assignment[exist_mask],
                target_assignment[exist_mask],
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
        else:
            assignment_loss = torch.tensor(0.0, device=pred_existence.device)

        # Total loss
        total_loss = (
            self.existence_weight * existence_loss +
            self.assignment_weight * assignment_loss
        )

        return {
            'loss': total_loss,
            'existence_loss': existence_loss,
            'assignment_loss': assignment_loss,
        }


class GraphLossWithNegativeSampling(nn.Module):
    """
    Graph loss with negative edge sampling.

    For training, we need both positive edges (exist in GT) and
    negative edges (don't exist). This loss handles the sampling.
    """

    def __init__(
        self,
        existence_weight: float = 1.0,
        assignment_weight: float = 1.0,
        negative_ratio: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            existence_weight: Weight for edge existence loss
            assignment_weight: Weight for edge assignment loss
            negative_ratio: Ratio of negative to positive edges to sample
            class_weights: Optional class weights for assignment loss
        """
        super().__init__()

        self.base_loss = GraphLoss(
            existence_weight=existence_weight,
            assignment_weight=assignment_weight,
            class_weights=class_weights,
        )
        self.negative_ratio = negative_ratio

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        num_nodes: int,
        edge_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss with negative sampling.

        Args:
            predictions: Model predictions
            targets: Ground truth labels for candidate edges
            num_nodes: Number of nodes in the graph
            edge_index: (2, E) candidate edge indices

        Returns:
            Loss dictionary
        """
        # For now, use base loss without additional negative sampling
        # (candidate edges already include negatives from graph extraction)
        return self.base_loss(predictions, targets)


def compute_graph_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute evaluation metrics for graph predictions.

    Args:
        predictions: Model predictions
        targets: Ground truth

    Returns:
        Dictionary of metrics
    """
    pred_existence = torch.sigmoid(predictions['edge_existence']) > 0.5
    target_existence = targets['edge_existence'] > 0.5

    pred_assignment = predictions['edge_assignment'].argmax(dim=-1)
    target_assignment = targets['edge_assignment']

    # Existence metrics
    tp = (pred_existence & target_existence).sum().float()
    fp = (pred_existence & ~target_existence).sum().float()
    fn = (~pred_existence & target_existence).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    existence_acc = (pred_existence == target_existence).float().mean()

    # Assignment accuracy (only on true positive edges)
    tp_mask = pred_existence & target_existence
    if tp_mask.sum() > 0:
        assignment_acc = (
            pred_assignment[tp_mask] == target_assignment[tp_mask]
        ).float().mean()
    else:
        assignment_acc = torch.tensor(0.0)

    return {
        'existence_precision': precision.item(),
        'existence_recall': recall.item(),
        'existence_f1': f1.item(),
        'existence_accuracy': existence_acc.item(),
        'assignment_accuracy': assignment_acc.item(),
    }
