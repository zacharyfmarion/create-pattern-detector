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
    )
