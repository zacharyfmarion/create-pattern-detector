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
    boundary_contact_corner_negative_weight: float = 4.0
    boundary_contact_hard_negative_weight: float = 0.0
    boundary_contact_hard_negative_ratio: float = 0.02
    boundary_contact_hard_negative_multiplier: float = 8.0
    boundary_contact_hard_negative_min_pixels: int = 256
    vertex_type_weight: float = 0.0
    vertex_type_class_weights: tuple[float, float, float, float] = (0.05, 4.0, 8.0, 1.5)
    vertex_type_focal_gamma: float = 1.5
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
        self.register_buffer(
            "vertex_type_class_weights",
            torch.tensor(self.config.vertex_type_class_weights, dtype=torch.float32),
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
            positive_weight=self.config.boundary_contact_pos_weight,
            corner_negative_weight=self.config.boundary_contact_corner_negative_weight,
        )
        boundary_contact_hard_negative_loss = _optional_boundary_contact_hard_negative_loss(
            outputs,
            targets,
            ratio=self.config.boundary_contact_hard_negative_ratio,
            multiplier=self.config.boundary_contact_hard_negative_multiplier,
            min_pixels=self.config.boundary_contact_hard_negative_min_pixels,
        )
        vertex_type_loss = _optional_vertex_type_loss(
            outputs,
            targets,
            self.vertex_type_class_weights,
            focal_gamma=self.config.vertex_type_focal_gamma,
        )
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
            + self.config.boundary_contact_hard_negative_weight * boundary_contact_hard_negative_loss
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
            "boundary_contact_hard_negative": boundary_contact_hard_negative_loss,
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
    *,
    positive_weight: float,
    corner_negative_weight: float,
) -> torch.Tensor:
    if "boundary_contact_logits" not in outputs or "v2_boundary_contact_heatmap" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    logits = outputs["boundary_contact_logits"]
    target = targets["v2_boundary_contact_heatmap"]
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive_weight = float(positive_weight)
    corner_weight = float(corner_negative_weight)
    batch_losses = []
    vertex_type = targets.get("v2_vertex_type")
    for item_loss, item_target, item_vertex_type in zip(
        loss,
        target,
        [None] * len(target) if vertex_type is None else vertex_type,
    ):
        weighted_terms = []
        weights = []
        positive_mask = item_target >= 0.1
        if torch.any(positive_mask):
            weighted_terms.append(item_loss[positive_mask].mean() * positive_weight)
            weights.append(positive_weight)
        negative_mask = item_target < 0.05
        if torch.any(negative_mask):
            weighted_terms.append(item_loss[negative_mask].mean())
            weights.append(1.0)
        if item_vertex_type is not None and corner_weight > 0.0:
            corner_mask = (item_vertex_type.unsqueeze(0) == 1) & negative_mask
            if torch.any(corner_mask):
                weighted_terms.append(item_loss[corner_mask].mean() * corner_weight)
                weights.append(corner_weight)
        if weighted_terms:
            batch_losses.append(torch.stack(weighted_terms).sum() / max(1e-6, sum(weights)))
    if not batch_losses:
        return logits.new_tensor(0.0)
    return torch.stack(batch_losses).mean()


def _optional_boundary_contact_hard_negative_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    ratio: float,
    multiplier: float,
    min_pixels: int,
) -> torch.Tensor:
    if "boundary_contact_logits" not in outputs or "v2_boundary_contact_heatmap" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    if ratio <= 0.0 or multiplier <= 0.0:
        return outputs["line_logits"].new_tensor(0.0)
    per_pixel = F.binary_cross_entropy_with_logits(
        outputs["boundary_contact_logits"],
        targets["v2_boundary_contact_heatmap"],
        reduction="none",
    )
    batch_losses = []
    for item_loss, item_target in zip(per_pixel, targets["v2_boundary_contact_heatmap"]):
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
        return outputs["line_logits"].new_tensor(0.0)
    return torch.stack(batch_losses).mean()


def _optional_vertex_type_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    *,
    focal_gamma: float,
) -> torch.Tensor:
    if "vertex_type_logits" not in outputs or "v2_vertex_type" not in targets:
        return outputs["line_logits"].new_tensor(0.0)
    logits = outputs["vertex_type_logits"]
    target = targets["v2_vertex_type"]
    weight = class_weights.to(logits.device)
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    if focal_gamma <= 0.0:
        weighted_pixel = per_pixel
    else:
        probs = F.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        focal = torch.pow((1.0 - probs).clamp(0.0, 1.0), float(focal_gamma))
        weighted_pixel = per_pixel * focal

    weighted_terms = []
    weights = []
    class_count = min(int(logits.shape[1]), int(weight.numel()))
    for class_id in range(class_count):
        class_mask = target == class_id
        if not torch.any(class_mask):
            continue
        class_weight = weight[class_id]
        weighted_terms.append(weighted_pixel[class_mask].mean() * class_weight)
        weights.append(class_weight)
    if not weighted_terms:
        return logits.new_tensor(0.0)
    return torch.stack(weighted_terms).sum() / torch.stack(weights).sum().clamp_min(1e-6)


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
