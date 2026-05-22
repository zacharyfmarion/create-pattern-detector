"""Convert CPLineNet predictions into PlanarGraphBuilder evidence."""

from __future__ import annotations

import numpy as np
import torch

from src.vectorization.planar_graph_builder import VectorizerEvidence


@torch.no_grad()
def cpline_outputs_to_evidence(
    outputs: dict[str, torch.Tensor],
    *,
    batch_index: int = 0,
    line_threshold: float = 0.35,
) -> VectorizerEvidence:
    line_prob = torch.sigmoid(outputs["line_logits"][batch_index, 0]).detach().cpu().numpy()
    angle = outputs["angle"][batch_index].detach().cpu().permute(1, 2, 0).numpy()
    junction_heatmap = torch.sigmoid(outputs["junction_logits"][batch_index, 0]).detach().cpu().numpy()
    assignment_pred = outputs["assignment_logits"][batch_index].argmax(dim=0).detach().cpu().numpy()
    assignment_labels = np.zeros_like(assignment_pred, dtype=np.uint8)
    assignment_labels[line_prob >= line_threshold] = assignment_pred[line_prob >= line_threshold].astype(np.uint8) + 1
    return VectorizerEvidence(
        line_prob=line_prob.astype(np.float32),
        angle=angle.astype(np.float32),
        junction_heatmap=junction_heatmap.astype(np.float32),
        assignment_labels=assignment_labels,
        non_crease_prob=_optional_sigmoid_map(outputs, "non_crease_logits", batch_index),
        line_style_prob=_optional_softmax_map(outputs, "line_style_logits", batch_index),
        boundary_contact_heatmap=_optional_sigmoid_map(
            outputs,
            "boundary_contact_logits",
            batch_index,
        ),
        vertex_type_prob=_optional_softmax_map(outputs, "vertex_type_logits", batch_index),
        boundary_side_prob=_optional_softmax_map(outputs, "boundary_side_logits", batch_index),
        boundary_offset=_optional_channel_last(outputs, "boundary_offset", batch_index),
        boundary_coord=_optional_scalar_map(outputs, "boundary_coord", batch_index),
    )


def _optional_sigmoid_map(
    outputs: dict[str, torch.Tensor],
    key: str,
    batch_index: int,
) -> np.ndarray | None:
    if key not in outputs:
        return None
    tensor = outputs[key][batch_index]
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    elif tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    return torch.sigmoid(tensor).detach().cpu().numpy().astype(np.float32)


def _optional_softmax_map(
    outputs: dict[str, torch.Tensor],
    key: str,
    batch_index: int,
) -> np.ndarray | None:
    if key not in outputs:
        return None
    tensor = torch.softmax(outputs[key][batch_index], dim=0)
    return tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def _optional_channel_last(
    outputs: dict[str, torch.Tensor],
    key: str,
    batch_index: int,
) -> np.ndarray | None:
    if key not in outputs:
        return None
    return outputs[key][batch_index].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def _optional_scalar_map(
    outputs: dict[str, torch.Tensor],
    key: str,
    batch_index: int,
) -> np.ndarray | None:
    if key not in outputs:
        return None
    tensor = outputs[key][batch_index]
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    return tensor.detach().cpu().numpy().astype(np.float32)
