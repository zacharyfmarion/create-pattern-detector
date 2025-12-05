"""
Ground truth label generation for Graph Head training.

Matches candidate graph vertices/edges to GT FOLD data and generates:
- Edge existence labels (1=keep, 0=drop)
- Edge assignment labels (0=M, 1=V, 2=B, 3=U)
- Vertex refinement offsets (GT_pos - detected_pos)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

from .fold_parser import FOLDParser, CreasePattern, transform_coords


@dataclass
class GraphLabels:
    """Ground truth labels for a single graph."""
    edge_existence: torch.Tensor  # (E,) binary
    edge_assignment: torch.Tensor  # (E,) class labels 0-3
    vertex_offset: torch.Tensor  # (N, 2) position offsets
    vertex_matched: torch.Tensor  # (N,) binary mask for matched vertices
    # Optional: matching info for debugging
    vertex_match_indices: Optional[torch.Tensor] = None  # (N,) GT index or -1
    edge_match_indices: Optional[torch.Tensor] = None  # (E,) GT index or -1


def generate_graph_labels(
    candidate_vertices: torch.Tensor,  # (N, 2) detected vertex positions
    candidate_edges: torch.Tensor,  # (2, E) edge indices
    gt_vertices: torch.Tensor,  # (M, 2) GT vertex positions
    gt_edges: torch.Tensor,  # (2, F) GT edge indices
    gt_assignments: torch.Tensor,  # (F,) GT edge assignments
    vertex_match_threshold: float = 8.0,
    edge_overlap_threshold: float = 0.3,
) -> GraphLabels:
    """
    Generate GT labels by matching candidate graph to GT graph.

    Matching strategy:
    1. Match candidate vertices to GT vertices (Hungarian, threshold)
    2. Match candidate edges to GT edges (endpoint matching + overlap)
    3. Generate labels based on matches

    Args:
        candidate_vertices: (N, 2) detected vertex positions in pixels
        candidate_edges: (2, E) edge index pairs
        gt_vertices: (M, 2) GT vertex positions in pixels
        gt_edges: (2, F) GT edge index pairs
        gt_assignments: (F,) GT edge assignment labels (0-3)
        vertex_match_threshold: Max distance (pixels) for vertex matching
        edge_overlap_threshold: Min line IoU for edge matching

    Returns:
        GraphLabels with all required training labels
    """
    device = candidate_vertices.device
    N = candidate_vertices.shape[0]
    E = candidate_edges.shape[1]
    M = gt_vertices.shape[0]
    F = gt_edges.shape[1]

    # === Step 1: Match vertices ===
    vertex_matches, vertex_offsets = match_vertices(
        candidate_vertices, gt_vertices, vertex_match_threshold
    )
    # vertex_matches: (N,) index into GT or -1 if unmatched
    # vertex_offsets: (N, 2) offset to matched GT vertex (0 if unmatched)

    vertex_matched = vertex_matches >= 0

    # === Step 2: Match edges ===
    edge_matches, edge_labels = match_edges(
        candidate_vertices,
        candidate_edges,
        gt_vertices,
        gt_edges,
        gt_assignments,
        vertex_matches,
        edge_overlap_threshold,
    )
    # edge_matches: (E,) index into GT edges or -1
    # edge_labels: (E,) assignment label or 0

    edge_existence = (edge_matches >= 0).float()

    return GraphLabels(
        edge_existence=edge_existence,
        edge_assignment=edge_labels,
        vertex_offset=vertex_offsets,
        vertex_matched=vertex_matched.float(),
        vertex_match_indices=vertex_matches,
        edge_match_indices=edge_matches,
    )


def match_vertices(
    candidate_vertices: torch.Tensor,  # (N, 2)
    gt_vertices: torch.Tensor,  # (M, 2)
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match candidate vertices to GT vertices using Hungarian algorithm.

    Args:
        candidate_vertices: (N, 2) detected positions
        gt_vertices: (M, 2) GT positions
        threshold: Max matching distance in pixels

    Returns:
        matches: (N,) GT index for each candidate, -1 if unmatched
        offsets: (N, 2) offset to GT position (0 if unmatched)
    """
    device = candidate_vertices.device
    N = candidate_vertices.shape[0]
    M = gt_vertices.shape[0]

    if N == 0:
        return (
            torch.full((0,), -1, dtype=torch.long, device=device),
            torch.zeros((0, 2), device=device),
        )

    if M == 0:
        return (
            torch.full((N,), -1, dtype=torch.long, device=device),
            torch.zeros((N, 2), device=device),
        )

    # Compute pairwise distances
    # (N, 1, 2) - (1, M, 2) -> (N, M)
    dists = torch.norm(
        candidate_vertices.unsqueeze(1) - gt_vertices.unsqueeze(0),
        dim=2
    )

    # Hungarian matching
    dists_np = dists.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(dists_np)

    # Build matches tensor
    matches = torch.full((N,), -1, dtype=torch.long, device=device)
    offsets = torch.zeros((N, 2), device=device)

    for r, c in zip(row_ind, col_ind):
        if dists_np[r, c] <= threshold:
            matches[r] = c
            offsets[r] = gt_vertices[c] - candidate_vertices[r]

    return matches, offsets


def match_edges(
    candidate_vertices: torch.Tensor,  # (N, 2)
    candidate_edges: torch.Tensor,  # (2, E)
    gt_vertices: torch.Tensor,  # (M, 2)
    gt_edges: torch.Tensor,  # (2, F)
    gt_assignments: torch.Tensor,  # (F,)
    vertex_matches: torch.Tensor,  # (N,) candidate -> GT vertex mapping
    overlap_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match candidate edges to GT edges.

    Matching criteria:
    1. Both endpoints must be matched to GT vertices
    2. The matched GT vertices must form a GT edge
    3. (Optional) Line overlap must exceed threshold

    Args:
        candidate_vertices: (N, 2) candidate positions
        candidate_edges: (2, E) candidate edge indices
        gt_vertices: (M, 2) GT positions
        gt_edges: (2, F) GT edge indices
        gt_assignments: (F,) GT assignment labels
        vertex_matches: (N,) mapping from candidate to GT vertex index
        overlap_threshold: Min line IoU (unused for now, using endpoint matching)

    Returns:
        edge_matches: (E,) GT edge index or -1
        edge_labels: (E,) assignment label (0 for unmatched)
    """
    device = candidate_vertices.device
    E = candidate_edges.shape[1]
    F = gt_edges.shape[1]

    edge_matches = torch.full((E,), -1, dtype=torch.long, device=device)
    edge_labels = torch.zeros((E,), dtype=torch.long, device=device)

    if E == 0 or F == 0:
        return edge_matches, edge_labels

    # Build GT edge lookup: (v1, v2) -> edge_idx
    # Use frozenset semantics (undirected)
    gt_edge_lookup = {}
    for f in range(F):
        v1, v2 = gt_edges[0, f].item(), gt_edges[1, f].item()
        key = (min(v1, v2), max(v1, v2))
        gt_edge_lookup[key] = f

    # Match each candidate edge
    for e in range(E):
        src, dst = candidate_edges[0, e].item(), candidate_edges[1, e].item()

        # Get matched GT vertices
        gt_src = vertex_matches[src].item()
        gt_dst = vertex_matches[dst].item()

        if gt_src < 0 or gt_dst < 0:
            # One or both endpoints unmatched
            continue

        # Check if this forms a GT edge
        key = (min(gt_src, gt_dst), max(gt_src, gt_dst))
        if key in gt_edge_lookup:
            gt_edge_idx = gt_edge_lookup[key]
            edge_matches[e] = gt_edge_idx
            edge_labels[e] = gt_assignments[gt_edge_idx]

    return edge_matches, edge_labels


def match_edges_with_overlap(
    candidate_vertices: torch.Tensor,
    candidate_edges: torch.Tensor,
    gt_vertices: torch.Tensor,
    gt_edges: torch.Tensor,
    gt_assignments: torch.Tensor,
    overlap_threshold: float = 0.3,
    num_samples: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match candidate edges to GT edges using line overlap.

    This is a more robust matching that doesn't require perfect vertex matching.
    Uses sampling along edges and computing overlap.

    Args:
        candidate_vertices: (N, 2)
        candidate_edges: (2, E)
        gt_vertices: (M, 2)
        gt_edges: (2, F)
        gt_assignments: (F,)
        overlap_threshold: Min overlap ratio
        num_samples: Points to sample along each edge

    Returns:
        edge_matches: (E,) GT edge index or -1
        edge_labels: (E,)
    """
    device = candidate_vertices.device
    E = candidate_edges.shape[1]
    F = gt_edges.shape[1]

    edge_matches = torch.full((E,), -1, dtype=torch.long, device=device)
    edge_labels = torch.zeros((E,), dtype=torch.long, device=device)

    if E == 0 or F == 0:
        return edge_matches, edge_labels

    # Sample points along candidate edges
    t = torch.linspace(0, 1, num_samples, device=device)

    # Get candidate edge endpoints
    cand_src = candidate_vertices[candidate_edges[0]]  # (E, 2)
    cand_dst = candidate_vertices[candidate_edges[1]]  # (E, 2)

    # Sample points: (E, num_samples, 2)
    cand_samples = (
        cand_src.unsqueeze(1) * (1 - t.view(1, -1, 1)) +
        cand_dst.unsqueeze(1) * t.view(1, -1, 1)
    )

    # Get GT edge endpoints
    gt_src = gt_vertices[gt_edges[0]]  # (F, 2)
    gt_dst = gt_vertices[gt_edges[1]]  # (F, 2)

    # For each candidate edge, find best matching GT edge
    for e in range(E):
        samples = cand_samples[e]  # (num_samples, 2)
        best_overlap = 0.0
        best_gt = -1

        for f in range(F):
            # Compute distance from each sample to GT edge line segment
            p1 = gt_src[f]
            p2 = gt_dst[f]

            # Vector from p1 to p2
            edge_vec = p2 - p1
            edge_len = torch.norm(edge_vec)

            if edge_len < 1e-6:
                continue

            edge_unit = edge_vec / edge_len

            # Project samples onto edge line
            # t = dot(sample - p1, edge_unit) / edge_len
            rel_samples = samples - p1  # (num_samples, 2)
            proj_t = (rel_samples * edge_unit).sum(dim=1) / edge_len

            # Clamp to [0, 1] for distance to segment
            proj_t_clamped = proj_t.clamp(0, 1)

            # Closest point on segment
            closest = p1 + proj_t_clamped.unsqueeze(1) * edge_vec

            # Distance from sample to closest point
            dists = torch.norm(samples - closest, dim=1)

            # Count samples within threshold (e.g., 5 pixels)
            dist_threshold = 5.0
            overlap = (dists < dist_threshold).float().mean().item()

            if overlap > best_overlap:
                best_overlap = overlap
                best_gt = f

        if best_overlap >= overlap_threshold:
            edge_matches[e] = best_gt
            edge_labels[e] = gt_assignments[best_gt]

    return edge_matches, edge_labels


def generate_batch_labels(
    vertices_list: List[torch.Tensor],
    edges_list: List[torch.Tensor],
    gt_vertices_list: List[torch.Tensor],
    gt_edges_list: List[torch.Tensor],
    gt_assignments_list: List[torch.Tensor],
    vertex_match_threshold: float = 8.0,
) -> Dict[str, torch.Tensor]:
    """
    Generate labels for a batch of graphs (PyG-style concatenated).

    Args:
        vertices_list: List of (N_i, 2) candidate vertices per graph
        edges_list: List of (2, E_i) candidate edges per graph
        gt_vertices_list: List of (M_i, 2) GT vertices per graph
        gt_edges_list: List of (2, F_i) GT edges per graph
        gt_assignments_list: List of (F_i,) GT assignments per graph
        vertex_match_threshold: Matching threshold in pixels

    Returns:
        Batched labels dictionary with concatenated tensors
    """
    all_edge_existence = []
    all_edge_assignment = []
    all_vertex_offset = []
    all_vertex_matched = []

    for i in range(len(vertices_list)):
        labels = generate_graph_labels(
            candidate_vertices=vertices_list[i],
            candidate_edges=edges_list[i],
            gt_vertices=gt_vertices_list[i],
            gt_edges=gt_edges_list[i],
            gt_assignments=gt_assignments_list[i],
            vertex_match_threshold=vertex_match_threshold,
        )

        all_edge_existence.append(labels.edge_existence)
        all_edge_assignment.append(labels.edge_assignment)
        all_vertex_offset.append(labels.vertex_offset)
        all_vertex_matched.append(labels.vertex_matched)

    return {
        'edge_existence': torch.cat(all_edge_existence, dim=0),
        'edge_assignment': torch.cat(all_edge_assignment, dim=0),
        'vertex_offset': torch.cat(all_vertex_offset, dim=0),
        'vertex_matched': torch.cat(all_vertex_matched, dim=0),
    }


def crease_pattern_to_tensors(
    cp: CreasePattern,
    image_size: int = 512,
    padding: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a CreasePattern to tensors in image pixel coordinates.

    Uses the existing transform_coords from fold_parser.py.

    Args:
        cp: CreasePattern object
        image_size: Output image size
        padding: Image padding

    Returns:
        vertices: (M, 2) in image pixel coords
        edges: (2, F) edge indices
        assignments: (F,) integer labels
    """
    # Use existing transform from fold_parser
    pixel_vertices, _ = transform_coords(
        cp.vertices,
        image_size=image_size,
        padding=padding,
    )

    vertices = torch.from_numpy(pixel_vertices).float()

    # Edge indices (transpose to (2, F) format)
    edges = torch.from_numpy(cp.edges).long().T

    # Assignment labels (already integers in CreasePattern)
    assignments = torch.from_numpy(cp.assignments).long()

    return vertices, edges, assignments


def load_gt_from_fold(
    fold_path: str,
    image_size: int = 512,
    padding: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load GT graph from a FOLD file.

    Args:
        fold_path: Path to FOLD file
        image_size: Image size for coordinate transform
        padding: Padding for coordinate transform

    Returns:
        vertices: (M, 2) GT vertex positions in pixels
        edges: (2, F) GT edge indices
        assignments: (F,) GT assignments (0=M, 1=V, 2=B, 3=U)
    """
    parser = FOLDParser()
    cp = parser.parse(fold_path)
    return crease_pattern_to_tensors(cp, image_size, padding)
