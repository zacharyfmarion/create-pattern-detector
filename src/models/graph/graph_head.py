"""
Graph Head for crease pattern edge prediction and assignment.

Takes candidate graphs from over-complete extraction and:
1. Extracts rich node/edge features from backbone + pixel head outputs
2. Refines features through a GNN with explicit edge updates
3. Predicts edge existence, assignment (M/V/B/U), and vertex refinement

Supports PyG-style batching for variable-size graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .features import NodeFeatureExtractor, EdgeFeatureExtractor
from .layers import GraphNetwork


class EdgeExistenceHead(nn.Module):
    """Binary classifier for edge existence."""

    def __init__(self, edge_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: (E, edge_dim)

        Returns:
            existence_logits: (E,)
        """
        return self.head(edge_features).squeeze(-1)


class EdgeAssignmentHead(nn.Module):
    """4-class classifier for edge assignment (M, V, B, U)."""

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: (E, edge_dim)

        Returns:
            assignment_logits: (E, num_classes)
        """
        return self.head(edge_features)


class VertexRefinementHead(nn.Module):
    """Predict 2D offset for vertex position refinement."""

    def __init__(self, node_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (N, node_dim)

        Returns:
            offset: (N, 2) predicted position offset
        """
        return self.head(node_features)


class GraphHead(nn.Module):
    """
    Graph Head for crease pattern processing.

    Architecture:
    1. Extract rich node/edge features using pixel head outputs
    2. Process through GNN with edge updates + attention-based node updates
    3. Predict edge existence, assignment, and vertex refinement
    """

    def __init__(
        self,
        backbone_channels: int = 480,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_gnn_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 4,
        dropout: float = 0.1,
        num_edge_samples: int = 16,
    ):
        """
        Args:
            backbone_channels: Number of channels from backbone feature map
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            num_gnn_layers: Number of GNN message-passing layers
            num_heads: Number of attention heads in node updates
            num_classes: Number of edge assignment classes (M, V, B, U)
            dropout: Dropout rate
            num_edge_samples: Number of samples along each edge for line evidence
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Feature extractors
        self.node_extractor = NodeFeatureExtractor(
            backbone_channels=backbone_channels,
            hidden_dim=node_dim,
            num_seg_classes=num_classes,
        )

        self.edge_extractor = EdgeFeatureExtractor(
            hidden_dim=edge_dim,
            num_seg_classes=num_classes,
            num_samples=num_edge_samples,
        )

        # GNN for message passing
        self.gnn = GraphNetwork(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_layers=num_gnn_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Output heads
        self.existence_head = EdgeExistenceHead(
            edge_dim=edge_dim,
            hidden_dim=edge_dim * 2,
            dropout=dropout,
        )

        self.assignment_head = EdgeAssignmentHead(
            edge_dim=edge_dim,
            hidden_dim=edge_dim * 2,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.refinement_head = VertexRefinementHead(
            node_dim=node_dim,
            hidden_dim=node_dim * 2,
            dropout=dropout,
        )

    def forward(
        self,
        vertices: torch.Tensor,
        edge_index: torch.Tensor,
        backbone_features: torch.Tensor,
        seg_probs: torch.Tensor,
        orientation: Optional[torch.Tensor] = None,
        image_size: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a single graph.

        Args:
            vertices: (N, 2) vertex coordinates in pixel space
            edge_index: (2, E) edge indices
            backbone_features: (1, C, H, W) backbone feature map
            seg_probs: (1, 4, H, W) segmentation probabilities from pixel head
            orientation: (1, 2, H, W) orientation field (cos, sin) or None
            image_size: Image size for normalization

        Returns:
            Dictionary with:
            - 'edge_existence': (E,) logits for edge existence
            - 'edge_assignment': (E, 4) logits for M/V/B/U assignment
            - 'vertex_offset': (N, 2) predicted vertex position offsets
            - 'node_features': (N, node_dim) final node features
            - 'edge_features': (E, edge_dim) final edge features
        """
        N = vertices.shape[0]
        E = edge_index.shape[1]
        device = vertices.device

        # Handle empty graphs
        if N == 0 or E == 0:
            return self._empty_output(N, E, device)

        # Extract initial node features
        node_features = self.node_extractor(
            vertices=vertices,
            backbone_features=backbone_features,
            seg_probs=seg_probs,
            edge_index=edge_index,
            image_size=image_size,
        )  # (N, node_dim)

        # Compute node structural features for edge extraction
        node_degree = self._compute_degree(edge_index, N, device)
        node_boundary_dist = self._compute_boundary_distance(vertices, image_size)

        # Extract initial edge features
        edge_features = self.edge_extractor(
            vertices=vertices,
            edge_index=edge_index,
            seg_probs=seg_probs,
            orientation=orientation,
            node_degree=node_degree,
            node_boundary_dist=node_boundary_dist,
            image_size=image_size,
        )  # (E, edge_dim)

        # Apply GNN layers
        node_features, edge_features = self.gnn(
            node_features, edge_features, edge_index
        )

        # Apply output heads
        edge_existence = self.existence_head(edge_features)  # (E,)
        edge_assignment = self.assignment_head(edge_features)  # (E, 4)
        vertex_offset = self.refinement_head(node_features)  # (N, 2)

        return {
            'edge_existence': edge_existence,
            'edge_assignment': edge_assignment,
            'vertex_offset': vertex_offset,
            'node_features': node_features,
            'edge_features': edge_features,
        }

    def forward_batch(
        self,
        vertices_list: List[torch.Tensor],
        edge_index_list: List[torch.Tensor],
        backbone_features: torch.Tensor,
        seg_probs: torch.Tensor,
        orientation: Optional[torch.Tensor] = None,
        image_size: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with PyG-style batching.

        Combines multiple graphs into one large graph with:
        - Concatenated vertices
        - Edge indices offset by cumulative node count
        - Batch tensor indicating which graph each node belongs to

        Args:
            vertices_list: List of (N_i, 2) vertex coordinates per graph
            edge_index_list: List of (2, E_i) edge indices per graph
            backbone_features: (B, C, H, W) backbone features (one per graph)
            seg_probs: (B, 4, H, W) segmentation probs (one per graph)
            orientation: (B, 2, H, W) orientation field or None
            image_size: Image size for normalization

        Returns:
            Dictionary with batched outputs and batch/ptr tensors for unbatching
        """
        B = len(vertices_list)
        device = vertices_list[0].device

        # Compute batch offsets
        node_counts = [v.shape[0] for v in vertices_list]
        edge_counts = [e.shape[1] for e in edge_index_list]

        node_ptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(node_counts), dim=0)),
            device=device
        )
        edge_ptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(edge_counts), dim=0)),
            device=device
        )

        # Create batch indices
        node_batch = torch.cat([
            torch.full((n,), i, device=device, dtype=torch.long)
            for i, n in enumerate(node_counts)
        ])

        # Concatenate vertices
        all_vertices = torch.cat(vertices_list, dim=0)  # (total_N, 2)

        # Offset and concatenate edge indices
        offset_edges = []
        for i, (edges, offset) in enumerate(zip(edge_index_list, node_ptr[:-1])):
            offset_edges.append(edges + offset)
        all_edge_index = torch.cat(offset_edges, dim=1)  # (2, total_E)

        # Process each graph's features separately then combine
        all_node_features = []
        all_edge_features = []

        for i in range(B):
            vertices = vertices_list[i]
            edge_index = edge_index_list[i]
            bb_feat = backbone_features[i:i+1]  # (1, C, H, W)
            seg = seg_probs[i:i+1]  # (1, 4, H, W)
            orient = orientation[i:i+1] if orientation is not None else None

            N_i = vertices.shape[0]

            if N_i == 0 or edge_index.shape[1] == 0:
                all_node_features.append(
                    torch.zeros(N_i, self.node_dim, device=device)
                )
                all_edge_features.append(
                    torch.zeros(edge_index.shape[1], self.edge_dim, device=device)
                )
                continue

            # Extract node features
            node_feat = self.node_extractor(
                vertices=vertices,
                backbone_features=bb_feat,
                seg_probs=seg,
                edge_index=edge_index,
                image_size=image_size,
            )
            all_node_features.append(node_feat)

            # Compute structural features
            node_degree = self._compute_degree(edge_index, N_i, device)
            node_boundary_dist = self._compute_boundary_distance(vertices, image_size)

            # Extract edge features
            edge_feat = self.edge_extractor(
                vertices=vertices,
                edge_index=edge_index,
                seg_probs=seg,
                orientation=orient,
                node_degree=node_degree,
                node_boundary_dist=node_boundary_dist,
                image_size=image_size,
            )
            all_edge_features.append(edge_feat)

        # Concatenate features
        node_features = torch.cat(all_node_features, dim=0)  # (total_N, node_dim)
        edge_features = torch.cat(all_edge_features, dim=0)  # (total_E, edge_dim)

        # Apply GNN on the combined graph
        node_features, edge_features = self.gnn(
            node_features, edge_features, all_edge_index
        )

        # Apply output heads
        edge_existence = self.existence_head(edge_features)
        edge_assignment = self.assignment_head(edge_features)
        vertex_offset = self.refinement_head(node_features)

        return {
            'edge_existence': edge_existence,  # (total_E,)
            'edge_assignment': edge_assignment,  # (total_E, 4)
            'vertex_offset': vertex_offset,  # (total_N, 2)
            'node_features': node_features,
            'edge_features': edge_features,
            # Batching info for loss computation
            'node_batch': node_batch,  # (total_N,)
            'node_ptr': node_ptr,  # (B+1,)
            'edge_ptr': edge_ptr,  # (B+1,)
        }

    def _empty_output(
        self, N: int, E: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Return empty tensors for empty graphs."""
        return {
            'edge_existence': torch.zeros(E, device=device),
            'edge_assignment': torch.zeros(E, 4, device=device),
            'vertex_offset': torch.zeros(N, 2, device=device),
            'node_features': torch.zeros(N, self.node_dim, device=device),
            'edge_features': torch.zeros(E, self.edge_dim, device=device),
        }

    def _compute_degree(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        """Compute degree of each node."""
        degree = torch.zeros(num_nodes, device=device)
        src, dst = edge_index[0], edge_index[1]
        degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        return degree

    def _compute_boundary_distance(
        self, vertices: torch.Tensor, image_size: int
    ) -> torch.Tensor:
        """Compute normalized distance to nearest image boundary."""
        dist_left = vertices[:, 0]
        dist_right = image_size - vertices[:, 0]
        dist_top = vertices[:, 1]
        dist_bottom = image_size - vertices[:, 1]

        min_dist = torch.min(
            torch.min(dist_left, dist_right),
            torch.min(dist_top, dist_bottom)
        )
        return min_dist / (image_size / 2)


def unbatch_outputs(
    outputs: Dict[str, torch.Tensor],
    num_graphs: int,
) -> List[Dict[str, torch.Tensor]]:
    """
    Unbatch PyG-style batched outputs into per-graph dictionaries.

    Args:
        outputs: Batched output dictionary from forward_batch
        num_graphs: Number of graphs in the batch

    Returns:
        List of per-graph output dictionaries
    """
    node_ptr = outputs['node_ptr']
    edge_ptr = outputs['edge_ptr']

    result = []
    for i in range(num_graphs):
        node_start, node_end = node_ptr[i].item(), node_ptr[i+1].item()
        edge_start, edge_end = edge_ptr[i].item(), edge_ptr[i+1].item()

        result.append({
            'edge_existence': outputs['edge_existence'][edge_start:edge_end],
            'edge_assignment': outputs['edge_assignment'][edge_start:edge_end],
            'vertex_offset': outputs['vertex_offset'][node_start:node_end],
            'node_features': outputs['node_features'][node_start:node_end],
            'edge_features': outputs['edge_features'][edge_start:edge_end],
        })

    return result
