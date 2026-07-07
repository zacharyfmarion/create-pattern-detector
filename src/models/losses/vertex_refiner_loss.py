"""Multi-task loss for VertexRefinerV1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses.cpline_loss import _penalty_reduced_focal_loss


@dataclass(frozen=True)
class VertexRefinerLossConfig:
    heatmap_weight: float = 1.0
    boundary_heatmap_weight: float = 1.0
    offset_weight: float = 0.5
    kind_weight: float = 0.2
    boundary_side_weight: float = 0.2
    degree_weight: float = 0.2
    incident_ray_weight: float = 0.5
    close_pair_repulsion_weight: float = 0.25
    heatmap_focal_alpha: float = 2.0
    heatmap_focal_beta: float = 4.0
    ray_focal_gamma: float = 2.0
    close_pair_margin: float = 0.05


class VertexRefinerLoss(nn.Module):
    def __init__(self, config: VertexRefinerLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or VertexRefinerLossConfig()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        heatmap_loss = _penalty_reduced_focal_loss(
            outputs["vertex_heatmap"],
            targets["vertex_heatmap"],
            alpha=self.config.heatmap_focal_alpha,
            beta=self.config.heatmap_focal_beta,
        )
        boundary_heatmap_loss = (
            _penalty_reduced_focal_loss(
                outputs["boundary_contact_heatmap"],
                targets["boundary_contact_heatmap"],
                alpha=self.config.heatmap_focal_alpha,
                beta=self.config.heatmap_focal_beta,
            )
            if "boundary_contact_heatmap" in outputs
            else outputs["vertex_heatmap"].new_tensor(0.0)
        )
        offset_loss = _masked_smooth_l1(
            outputs["vertex_offset"],
            targets["vertex_offset"],
            targets["vertex_offset_mask"],
        )
        kind_loss = _masked_cross_entropy(
            outputs["vertex_kind"],
            targets["vertex_kind"],
            targets["vertex_kind_mask"],
        )
        boundary_side_loss = (
            _masked_cross_entropy(
                outputs["boundary_side"],
                targets["boundary_side"],
                targets["boundary_side_mask"],
            )
            if "boundary_side" in outputs
            else outputs["vertex_heatmap"].new_tensor(0.0)
        )
        degree_loss = _masked_cross_entropy(
            outputs["degree"],
            targets["degree"],
            targets["degree_mask"],
        )
        ray_loss = _masked_binary_focal(
            outputs["incident_rays"],
            targets["incident_rays"],
            targets["incident_ray_mask"],
            gamma=self.config.ray_focal_gamma,
        )
        repulsion_loss = _close_pair_repulsion_loss(
            outputs["vertex_heatmap"],
            targets["vertex_heatmap"],
            margin=self.config.close_pair_margin,
        )
        total = (
            self.config.heatmap_weight * heatmap_loss
            + self.config.boundary_heatmap_weight * boundary_heatmap_loss
            + self.config.offset_weight * offset_loss
            + self.config.kind_weight * kind_loss
            + self.config.boundary_side_weight * boundary_side_loss
            + self.config.degree_weight * degree_loss
            + self.config.incident_ray_weight * ray_loss
            + self.config.close_pair_repulsion_weight * repulsion_loss
        )
        return {
            "total": total,
            "heatmap": heatmap_loss,
            "boundary_heatmap": boundary_heatmap_loss,
            "offset": offset_loss,
            "kind": kind_loss,
            "boundary_side": boundary_side_loss,
            "degree": degree_loss,
            "incident_rays": ray_loss,
            "close_pair_repulsion": repulsion_loss,
        }


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask[:, None, :, :]
    mask = mask.to(device=pred.device, dtype=torch.bool)
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    target = target.to(device=pred.device, dtype=pred.dtype)
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return loss[mask.expand_as(pred)].mean()


def _masked_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if not torch.any(mask):
        return logits.new_tensor(0.0)
    loss = F.cross_entropy(logits, target.to(logits.device), reduction="none")
    return loss[mask].mean()


def _masked_binary_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask[:, None, :, :]
    mask = mask.to(device=logits.device, dtype=torch.bool).expand_as(logits)
    if not torch.any(mask):
        return logits.new_tensor(0.0)
    target = target.to(device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = torch.where(target >= 0.5, probs, 1.0 - probs)
    focal = torch.pow((1.0 - p_t).clamp(0.0, 1.0), gamma) * bce
    return focal[mask].mean()


def _close_pair_repulsion_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Discourage filling low target saddles between nearby heatmap peaks."""
    probs = torch.sigmoid(logits)
    target = target.to(device=logits.device, dtype=logits.dtype)
    dip_mask = (target > 0.02) & (target < 0.75)
    if not torch.any(dip_mask):
        return logits.new_tensor(0.0)
    excess = torch.relu(probs - target - float(margin))
    return torch.square(excess[dip_mask]).mean()
