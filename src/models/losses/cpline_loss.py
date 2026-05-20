"""Losses for roadmap-native CPLineNet fields."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CPLineLossConfig:
    line_weight: float = 1.0
    angle_weight: float = 0.25
    junction_weight: float = 1.0
    junction_offset_weight: float = 0.25
    assignment_weight: float = 0.2
    line_pos_weight: float = 20.0
    junction_pos_weight: float = 50.0
    assignment_ignore_index: int = -100
    line_hard_negative_weight: float = 0.25
    line_hard_negative_ratio: float = 0.05
    line_hard_negative_multiplier: float = 4.0
    line_hard_negative_min_pixels: int = 256
    non_crease_weight: float = 0.0
    non_crease_pos_weight: float = 20.0
    line_style_weight: float = 0.0
    line_style_ignore_index: int = -100
    use_observed_assignment_target: bool = False
    boundary_contact_weight: float = 0.0
    boundary_contact_pos_weight: float = 50.0
    vertex_type_weight: float = 0.0
    boundary_side_weight: float = 0.0
    boundary_side_ignore_index: int = -100
    boundary_offset_weight: float = 0.0
    boundary_coord_weight: float = 0.0


class CPLineLoss(nn.Module):
    def __init__(self, config: CPLineLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or CPLineLossConfig()
        self.register_buffer("line_pos_weight", torch.tensor([self.config.line_pos_weight]))
        self.register_buffer("junction_pos_weight", torch.tensor([self.config.junction_pos_weight]))
        self.register_buffer("non_crease_pos_weight", torch.tensor([self.config.non_crease_pos_weight]))
        self.register_buffer(
            "boundary_contact_pos_weight",
            torch.tensor([self.config.boundary_contact_pos_weight]),
        )

    def forward(
        self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        line_loss = F.binary_cross_entropy_with_logits(
            outputs["line_logits"],
            targets["line_prob"],
            pos_weight=self.line_pos_weight.to(outputs["line_logits"].device),
        )
        line_hard_negative_loss = _hard_negative_line_loss(
            outputs["line_logits"],
            targets["line_prob"],
            ratio=self.config.line_hard_negative_ratio,
            multiplier=self.config.line_hard_negative_multiplier,
            min_pixels=self.config.line_hard_negative_min_pixels,
        )
        line_mask = targets["line_prob"][:, 0] > 0.1
        angle_loss = _masked_angle_loss(outputs["angle"], targets["angle"], line_mask)
        junction_loss = F.binary_cross_entropy_with_logits(
            outputs["junction_logits"],
            targets["junction_heatmap"],
            pos_weight=self.junction_pos_weight.to(outputs["junction_logits"].device),
        )
        offset_loss = _masked_l1(
            outputs["junction_offset"],
            targets["junction_offset"],
            targets["junction_mask"],
        )
        assignment_target = targets["assignment"]
        if self.config.use_observed_assignment_target and "v2_observed_assignment" in targets:
            assignment_target = targets["v2_observed_assignment"]
        assignment_loss = F.cross_entropy(
            outputs["assignment_logits"],
            assignment_target,
            ignore_index=self.config.assignment_ignore_index,
        )
        non_crease_loss = _optional_non_crease_loss(outputs, targets, self.non_crease_pos_weight)
        line_style_loss = _optional_line_style_loss(
            outputs,
            targets,
            ignore_index=self.config.line_style_ignore_index,
        )
        boundary_contact_loss = _optional_boundary_contact_loss(
            outputs,
            targets,
            self.boundary_contact_pos_weight,
        )
        vertex_type_loss = _optional_vertex_type_loss(outputs, targets)
        boundary_side_loss = _optional_boundary_side_loss(
            outputs,
            targets,
            ignore_index=self.config.boundary_side_ignore_index,
        )
        boundary_offset_loss = _optional_boundary_offset_loss(outputs, targets)
        boundary_coord_loss = _optional_boundary_coord_loss(outputs, targets)
        total = (
            self.config.line_weight * line_loss
            + self.config.line_hard_negative_weight * line_hard_negative_loss
            + self.config.angle_weight * angle_loss
            + self.config.junction_weight * junction_loss
            + self.config.junction_offset_weight * offset_loss
            + self.config.assignment_weight * assignment_loss
            + self.config.non_crease_weight * non_crease_loss
            + self.config.line_style_weight * line_style_loss
            + self.config.boundary_contact_weight * boundary_contact_loss
            + self.config.vertex_type_weight * vertex_type_loss
            + self.config.boundary_side_weight * boundary_side_loss
            + self.config.boundary_offset_weight * boundary_offset_loss
            + self.config.boundary_coord_weight * boundary_coord_loss
        )
        return {
            "total": total,
            "line": line_loss,
            "line_hard_negative": line_hard_negative_loss,
            "angle": angle_loss,
            "junction": junction_loss,
            "junction_offset": offset_loss,
            "assignment": assignment_loss,
            "non_crease": non_crease_loss,
            "line_style": line_style_loss,
            "boundary_contact": boundary_contact_loss,
            "vertex_type": vertex_type_loss,
            "boundary_side": boundary_side_loss,
            "boundary_offset": boundary_offset_loss,
            "boundary_coord": boundary_coord_loss,
        }


def _masked_angle_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    pred = F.normalize(pred, dim=1, eps=1e-6)
    target = F.normalize(target, dim=1, eps=1e-6)
    loss = 1.0 - (pred * target).sum(dim=1).clamp(-1.0, 1.0)
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    return loss[mask].mean()


def _hard_negative_line_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ratio: float,
    multiplier: float,
    min_pixels: int,
) -> torch.Tensor:
    if ratio <= 0.0 or multiplier <= 0.0:
        return logits.new_tensor(0.0)
    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    batch_losses = []
    for item_loss, item_target in zip(per_pixel, target):
        negative_mask = item_target < 0.05
        if not torch.any(negative_mask):
            continue
        negative_losses = item_loss[negative_mask]
        negative_count = int(negative_losses.numel())
        positive_count = int(torch.count_nonzero(item_target >= 0.1).item())
        ratio_k = max(1, int(round(negative_count * ratio)))
        positive_k = max(1, int(round(max(positive_count, 1) * multiplier)))
        k = min(negative_count, max(min_pixels, min(ratio_k, positive_k)))
        batch_losses.append(torch.topk(negative_losses, k=k, largest=True).values.mean())
    if not batch_losses:
        return logits.new_tensor(0.0)
    return torch.stack(batch_losses).mean()


def _optional_non_crease_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    if "non_crease_logits" not in outputs or "v2_non_crease_mask" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(
        outputs["non_crease_logits"],
        targets["v2_non_crease_mask"],
        pos_weight=pos_weight.to(outputs["non_crease_logits"].device),
    )


def _optional_line_style_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    ignore_index: int,
) -> torch.Tensor:
    if "line_style_logits" not in outputs or "v2_line_style" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    return F.cross_entropy(
        outputs["line_style_logits"],
        targets["v2_line_style"],
        ignore_index=ignore_index,
    )


def _optional_boundary_contact_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    if "boundary_contact_logits" not in outputs or "v2_boundary_contact_heatmap" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(
        outputs["boundary_contact_logits"],
        targets["v2_boundary_contact_heatmap"],
        pos_weight=pos_weight.to(outputs["boundary_contact_logits"].device),
    )


def _optional_vertex_type_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    if "vertex_type_logits" not in outputs or "v2_vertex_type" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    return F.cross_entropy(outputs["vertex_type_logits"], targets["v2_vertex_type"])


def _optional_boundary_side_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    ignore_index: int,
) -> torch.Tensor:
    if "boundary_side_logits" not in outputs or "v2_boundary_side" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    if not torch.any(targets["v2_boundary_side"] != ignore_index):
        return outputs["line_logits"].new_tensor(0.0)
    return F.cross_entropy(
        outputs["boundary_side_logits"],
        targets["v2_boundary_side"],
        ignore_index=ignore_index,
    )


def _optional_boundary_offset_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    if "boundary_offset" not in outputs or "v2_boundary_offset" not in targets or "v2_boundary_mask" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    return _masked_l1(outputs["boundary_offset"], targets["v2_boundary_offset"], targets["v2_boundary_mask"])


def _optional_boundary_coord_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    if "boundary_coord" not in outputs or "v2_boundary_coord" not in targets or "v2_boundary_mask" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    return _masked_l1(outputs["boundary_coord"], targets["v2_boundary_coord"], targets["v2_boundary_mask"])


def _masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    expanded = mask.unsqueeze(1).expand_as(pred)
    return torch.abs(pred - target)[expanded].mean()
