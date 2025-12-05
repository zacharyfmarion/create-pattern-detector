"""
Loss functions for Graph Head training.

Includes:
- Edge existence loss (BCE)
- Edge assignment loss (CE for M/V/B/U)
- Vertex refinement loss (L1)
- Segmentation consistency loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GraphLoss(nn.Module):
    """
    Combined loss for Graph Head training.

    Components:
    1. Edge existence loss (BCE) - for keep/drop classification
    2. Edge assignment loss (CE) - for M/V/B/U classification
    3. Vertex refinement loss (L1) - for sub-pixel position refinement
    4. Consistency loss (optional) - align predictions with pixel seg probs
    """

    def __init__(
        self,
        existence_weight: float = 1.0,
        assignment_weight: float = 1.0,
        refinement_weight: float = 0.5,
        consistency_weight: float = 0.0,  # Off by default for Phase 2a
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        pos_weight: Optional[float] = None,  # For imbalanced edge existence
    ):
        """
        Args:
            existence_weight: Weight for edge existence loss
            assignment_weight: Weight for edge assignment loss
            refinement_weight: Weight for vertex refinement loss
            consistency_weight: Weight for seg consistency loss
            class_weights: Optional class weights for assignment loss (4,)
            label_smoothing: Label smoothing for assignment loss
            pos_weight: Positive class weight for existence BCE
        """
        super().__init__()

        self.existence_weight = existence_weight
        self.assignment_weight = assignment_weight
        self.refinement_weight = refinement_weight
        self.consistency_weight = consistency_weight
        self.label_smoothing = label_smoothing

        # Register class weights as buffer
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

        # Positive weight for existence (handles class imbalance)
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        else:
            self.pos_weight = None

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        edge_mask: Optional[torch.Tensor] = None,
        vertex_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            predictions: Dictionary with:
                - 'edge_existence': (E,) logits
                - 'edge_assignment': (E, 4) logits
                - 'vertex_offset': (N, 2) predicted offsets
            targets: Dictionary with:
                - 'edge_existence': (E,) binary labels (1=keep, 0=drop)
                - 'edge_assignment': (E,) class labels (0=M, 1=V, 2=B, 3=U)
                - 'vertex_offset': (N, 2) GT offsets (GT_pos - detected_pos)
                - 'vertex_matched': (N,) binary mask for matched vertices
            edge_mask: Optional (E,) mask for valid edges
            vertex_mask: Optional (N,) mask for valid vertices

        Returns:
            Dictionary with individual losses and total loss
        """
        device = predictions['edge_existence'].device
        losses = {}

        # === Edge Existence Loss (BCE) ===
        pred_existence = predictions['edge_existence']
        target_existence = targets['edge_existence'].float()

        if edge_mask is not None:
            pred_existence = pred_existence[edge_mask]
            target_existence = target_existence[edge_mask]

        if pred_existence.numel() > 0:
            existence_loss = F.binary_cross_entropy_with_logits(
                pred_existence,
                target_existence,
                pos_weight=self.pos_weight,
            )
        else:
            existence_loss = torch.tensor(0.0, device=device)

        losses['existence_loss'] = existence_loss

        # === Edge Assignment Loss (CE, only for positive edges) ===
        pred_assignment = predictions['edge_assignment']
        target_assignment = targets['edge_assignment'].long()

        if edge_mask is not None:
            pred_assignment = pred_assignment[edge_mask]
            target_assignment = target_assignment[edge_mask]
            target_exist_masked = target_existence
        else:
            target_exist_masked = targets['edge_existence'].float()

        # Only compute assignment loss for edges that exist in GT
        exist_mask = target_exist_masked > 0.5

        if exist_mask.sum() > 0:
            assignment_loss = F.cross_entropy(
                pred_assignment[exist_mask],
                target_assignment[exist_mask],
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
        else:
            assignment_loss = torch.tensor(0.0, device=device)

        losses['assignment_loss'] = assignment_loss

        # === Vertex Refinement Loss (L1, only for matched vertices) ===
        if 'vertex_offset' in predictions and 'vertex_offset' in targets:
            pred_offset = predictions['vertex_offset']
            target_offset = targets['vertex_offset']

            # Get mask for matched vertices
            if 'vertex_matched' in targets:
                matched_mask = targets['vertex_matched'] > 0.5
            elif vertex_mask is not None:
                matched_mask = vertex_mask > 0.5
            else:
                matched_mask = torch.ones(
                    pred_offset.shape[0], dtype=torch.bool, device=device
                )

            if matched_mask.sum() > 0:
                refinement_loss = F.l1_loss(
                    pred_offset[matched_mask],
                    target_offset[matched_mask],
                )
            else:
                refinement_loss = torch.tensor(0.0, device=device)
        else:
            refinement_loss = torch.tensor(0.0, device=device)

        losses['refinement_loss'] = refinement_loss

        # === Consistency Loss (align with pixel seg probs) ===
        # This is computed externally and passed in targets if needed
        if self.consistency_weight > 0 and 'consistency_loss' in targets:
            consistency_loss = targets['consistency_loss']
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        losses['consistency_loss'] = consistency_loss

        # === Total Loss ===
        total_loss = (
            self.existence_weight * existence_loss +
            self.assignment_weight * assignment_loss +
            self.refinement_weight * refinement_loss +
            self.consistency_weight * consistency_loss
        )

        losses['loss'] = total_loss

        return losses


class GraphLossWithBatching(nn.Module):
    """
    Graph loss that handles PyG-style batched graphs.

    Computes per-graph losses and averages across the batch.
    """

    def __init__(
        self,
        existence_weight: float = 1.0,
        assignment_weight: float = 1.0,
        refinement_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.base_loss = GraphLoss(
            existence_weight=existence_weight,
            assignment_weight=assignment_weight,
            refinement_weight=refinement_weight,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for batched graphs.

        Uses edge_ptr and node_ptr from predictions to split by graph.

        Args:
            predictions: Batched predictions with 'edge_ptr', 'node_ptr'
            targets: Batched targets with same structure

        Returns:
            Loss dictionary (averaged across batch)
        """
        # If not batched (no ptr tensors), use base loss directly
        if 'edge_ptr' not in predictions:
            return self.base_loss(predictions, targets)

        edge_ptr = predictions['edge_ptr']
        node_ptr = predictions['node_ptr']
        num_graphs = len(edge_ptr) - 1

        # Accumulate losses
        total_losses = {
            'loss': 0.0,
            'existence_loss': 0.0,
            'assignment_loss': 0.0,
            'refinement_loss': 0.0,
        }

        for i in range(num_graphs):
            # Get slice indices
            e_start, e_end = edge_ptr[i].item(), edge_ptr[i+1].item()
            n_start, n_end = node_ptr[i].item(), node_ptr[i+1].item()

            # Slice predictions
            pred_i = {
                'edge_existence': predictions['edge_existence'][e_start:e_end],
                'edge_assignment': predictions['edge_assignment'][e_start:e_end],
                'vertex_offset': predictions['vertex_offset'][n_start:n_end],
            }

            # Slice targets
            target_i = {
                'edge_existence': targets['edge_existence'][e_start:e_end],
                'edge_assignment': targets['edge_assignment'][e_start:e_end],
                'vertex_offset': targets['vertex_offset'][n_start:n_end],
            }
            if 'vertex_matched' in targets:
                target_i['vertex_matched'] = targets['vertex_matched'][n_start:n_end]

            # Compute loss for this graph
            losses_i = self.base_loss(pred_i, target_i)

            # Accumulate
            for key in total_losses:
                if key in losses_i:
                    total_losses[key] = total_losses[key] + losses_i[key]

        # Average across graphs
        for key in total_losses:
            if isinstance(total_losses[key], torch.Tensor):
                total_losses[key] = total_losses[key] / num_graphs
            else:
                total_losses[key] = torch.tensor(
                    total_losses[key] / num_graphs,
                    device=predictions['edge_existence'].device
                )

        return total_losses


def compute_graph_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for graph predictions.

    Args:
        predictions: Model predictions
        targets: Ground truth
        threshold: Threshold for edge existence

    Returns:
        Dictionary of metrics
    """
    device = predictions['edge_existence'].device

    # Edge existence predictions
    pred_existence = torch.sigmoid(predictions['edge_existence']) > threshold
    target_existence = targets['edge_existence'] > 0.5

    # Edge assignment predictions
    pred_assignment = predictions['edge_assignment'].argmax(dim=-1)
    target_assignment = targets['edge_assignment']

    # === Existence Metrics ===
    tp = (pred_existence & target_existence).sum().float()
    fp = (pred_existence & ~target_existence).sum().float()
    fn = (~pred_existence & target_existence).sum().float()
    tn = (~pred_existence & ~target_existence).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    # === Assignment Metrics (on GT positive edges only) ===
    gt_positive_mask = target_existence
    if gt_positive_mask.sum() > 0:
        assignment_acc = (
            pred_assignment[gt_positive_mask] == target_assignment[gt_positive_mask]
        ).float().mean()

        # Per-class accuracy
        class_names = ['M', 'V', 'B', 'U']
        per_class_acc = {}
        for cls_idx, cls_name in enumerate(class_names):
            cls_mask = gt_positive_mask & (target_assignment == cls_idx)
            if cls_mask.sum() > 0:
                per_class_acc[f'acc_{cls_name}'] = (
                    pred_assignment[cls_mask] == cls_idx
                ).float().mean().item()
            else:
                per_class_acc[f'acc_{cls_name}'] = 0.0
    else:
        assignment_acc = torch.tensor(0.0, device=device)
        per_class_acc = {f'acc_{c}': 0.0 for c in ['M', 'V', 'B', 'U']}

    # === Assignment Metrics (on TP edges: predicted + GT positive) ===
    tp_mask = pred_existence & target_existence
    if tp_mask.sum() > 0:
        tp_assignment_acc = (
            pred_assignment[tp_mask] == target_assignment[tp_mask]
        ).float().mean()
    else:
        tp_assignment_acc = torch.tensor(0.0, device=device)

    # === Vertex Refinement Metrics ===
    if 'vertex_offset' in predictions and 'vertex_offset' in targets:
        pred_offset = predictions['vertex_offset']
        target_offset = targets['vertex_offset']

        if 'vertex_matched' in targets:
            matched = targets['vertex_matched'] > 0.5
        else:
            matched = torch.ones(pred_offset.shape[0], dtype=torch.bool, device=device)

        if matched.sum() > 0:
            offset_error = torch.norm(
                pred_offset[matched] - target_offset[matched], dim=1
            )
            mean_offset_error = offset_error.mean()
            median_offset_error = offset_error.median()
        else:
            mean_offset_error = torch.tensor(0.0, device=device)
            median_offset_error = torch.tensor(0.0, device=device)
    else:
        mean_offset_error = torch.tensor(0.0, device=device)
        median_offset_error = torch.tensor(0.0, device=device)

    metrics = {
        # Existence metrics
        'existence_precision': precision.item(),
        'existence_recall': recall.item(),
        'existence_f1': f1.item(),
        'existence_accuracy': accuracy.item(),
        # Assignment metrics
        'assignment_accuracy': assignment_acc.item(),
        'tp_assignment_accuracy': tp_assignment_acc.item(),
        # Refinement metrics
        'mean_offset_error': mean_offset_error.item(),
        'median_offset_error': median_offset_error.item(),
    }
    metrics.update(per_class_acc)

    return metrics


def compute_batched_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute metrics for batched graphs, averaging across batch.
    """
    if 'edge_ptr' not in predictions:
        return compute_graph_metrics(predictions, targets, threshold)

    edge_ptr = predictions['edge_ptr']
    node_ptr = predictions['node_ptr']
    num_graphs = len(edge_ptr) - 1

    all_metrics = []
    for i in range(num_graphs):
        e_start, e_end = edge_ptr[i].item(), edge_ptr[i+1].item()
        n_start, n_end = node_ptr[i].item(), node_ptr[i+1].item()

        pred_i = {
            'edge_existence': predictions['edge_existence'][e_start:e_end],
            'edge_assignment': predictions['edge_assignment'][e_start:e_end],
            'vertex_offset': predictions['vertex_offset'][n_start:n_end],
        }
        target_i = {
            'edge_existence': targets['edge_existence'][e_start:e_end],
            'edge_assignment': targets['edge_assignment'][e_start:e_end],
            'vertex_offset': targets['vertex_offset'][n_start:n_end],
        }
        if 'vertex_matched' in targets:
            target_i['vertex_matched'] = targets['vertex_matched'][n_start:n_end]

        metrics_i = compute_graph_metrics(pred_i, target_i, threshold)
        all_metrics.append(metrics_i)

    # Average across graphs
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = sum(m[key] for m in all_metrics) / num_graphs

    return avg_metrics
