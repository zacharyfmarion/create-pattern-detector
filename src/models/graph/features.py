"""
Node and edge feature extraction for the Graph Head.

This module extracts rich features from:
- Backbone feature maps (bilinear sampled at vertex locations)
- Pixel head outputs (segmentation, orientation)
- Graph structure (degree, boundary distance, edge geometry)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class NodeFeatureExtractor(nn.Module):
    """
    Extract features for each node (vertex) in the graph.

    Features include:
    - Normalized coordinates (2-dim)
    - Backbone features at vertex location (C-dim)
    - Local segmentation statistics (4-dim)
    - Structural hints: degree, boundary distance (2-dim)
    """

    def __init__(
        self,
        backbone_channels: int = 480,
        hidden_dim: int = 128,
        num_seg_classes: int = 4,  # M, V, B, U
        window_size: int = 9,
    ):
        super().__init__()
        self.backbone_channels = backbone_channels
        self.hidden_dim = hidden_dim
        self.num_seg_classes = num_seg_classes
        self.window_size = window_size

        # Input dim: 2 (coords) + C (backbone) + 4 (seg stats) + 2 (structural)
        input_dim = 2 + backbone_channels + num_seg_classes + 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        vertices: torch.Tensor,  # (N, 2) in pixel coords
        backbone_features: torch.Tensor,  # (1, C, H, W)
        seg_probs: torch.Tensor,  # (1, num_classes, H, W)
        edge_index: torch.Tensor,  # (2, E)
        image_size: int = 512,
    ) -> torch.Tensor:
        """
        Extract node features.

        Args:
            vertices: (N, 2) vertex coordinates in pixel space
            backbone_features: (1, C, H, W) backbone feature map
            seg_probs: (1, num_classes, H, W) segmentation probabilities
            edge_index: (2, E) edge indices for computing degree
            image_size: Size of the image for normalization

        Returns:
            node_features: (N, hidden_dim) node embeddings
        """
        N = vertices.shape[0]
        device = vertices.device

        # 1. Normalized coordinates
        coords_norm = vertices / image_size  # (N, 2) in [0, 1]

        # 2. Sample backbone features at vertex locations
        backbone_feats = self._sample_features(
            backbone_features, vertices, image_size
        )  # (N, C)

        # 3. Local segmentation statistics (mean in window)
        seg_stats = self._compute_seg_stats(
            seg_probs, vertices, image_size
        )  # (N, num_classes)

        # 4. Structural hints
        degree = self._compute_degree(edge_index, N)  # (N,)
        degree_norm = degree / (degree.max() + 1e-6)  # Normalize

        dist_to_boundary = self._compute_boundary_distance(
            vertices, image_size
        )  # (N,)

        structural = torch.stack([degree_norm, dist_to_boundary], dim=1)  # (N, 2)

        # Concatenate all features
        node_feats = torch.cat([
            coords_norm,
            backbone_feats,
            seg_stats,
            structural,
        ], dim=1)  # (N, input_dim)

        # Project through MLP
        return self.mlp(node_feats)  # (N, hidden_dim)

    def _sample_features(
        self,
        features: torch.Tensor,
        vertices: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """Bilinear sample features at vertex locations."""
        # Convert pixel coords to normalized grid coords [-1, 1]
        grid = (vertices / image_size) * 2 - 1  # (N, 2) in [-1, 1]
        grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

        # Handle backbone feature map resolution
        _, C, H, W = features.shape

        # Scale grid to feature map resolution
        scale_x = W / image_size
        scale_y = H / image_size
        grid_scaled = grid.clone()
        grid_scaled[..., 0] = grid[..., 0] * scale_x * (image_size / W)
        grid_scaled[..., 1] = grid[..., 1] * scale_y * (image_size / H)

        # Actually we just need to use the normalized coords directly
        # since grid_sample handles the scaling
        sampled = F.grid_sample(
            features, grid, mode='bilinear', align_corners=True
        )  # (1, C, N, 1)

        return sampled.squeeze(0).squeeze(-1).T  # (N, C)

    def _compute_seg_stats(
        self,
        seg_probs: torch.Tensor,
        vertices: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """Compute mean segmentation probabilities in window around each vertex."""
        N = vertices.shape[0]
        device = vertices.device
        _, num_classes, H, W = seg_probs.shape

        # For efficiency, use grid_sample with a single point
        # (computing window mean would require more complex pooling)
        # For now, just sample at the vertex location
        grid = (vertices / image_size) * 2 - 1
        grid = grid.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

        sampled = F.grid_sample(
            seg_probs, grid, mode='bilinear', align_corners=True
        )  # (1, num_classes, N, 1)

        return sampled.squeeze(0).squeeze(-1).T  # (N, num_classes)

    def _compute_degree(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute degree of each node."""
        device = edge_index.device
        degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        # Count edges for each node
        src, dst = edge_index[0], edge_index[1]
        ones = torch.ones(src.shape[0], dtype=torch.float32, device=device)
        degree.scatter_add_(0, src, ones)
        degree.scatter_add_(0, dst, ones.clone())

        return degree

    def _compute_boundary_distance(
        self,
        vertices: torch.Tensor,
        image_size: int,
    ) -> torch.Tensor:
        """Compute normalized distance to nearest image boundary."""
        # Distance to each edge
        dist_left = vertices[:, 0]
        dist_right = image_size - vertices[:, 0]
        dist_top = vertices[:, 1]
        dist_bottom = image_size - vertices[:, 1]

        # Minimum distance to any edge, normalized
        min_dist = torch.min(
            torch.min(dist_left, dist_right),
            torch.min(dist_top, dist_bottom)
        )

        return min_dist / (image_size / 2)  # Normalize to [0, 1]


class EdgeFeatureExtractor(nn.Module):
    """
    Extract features for each edge in the graph.

    Features include:
    - Geometry: direction, length, midpoint (5-dim)
    - Line evidence: mean seg probs, crease prob, orientation alignment (6-dim)
    - Endpoint context: degrees, boundary distances (4-dim)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_seg_classes: int = 4,
        num_samples: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_seg_classes = num_seg_classes
        self.num_samples = num_samples

        # Input dim: 5 (geometry) + 4 (seg) + 1 (crease) + 1 (orient) + 4 (endpoint)
        input_dim = 5 + num_seg_classes + 1 + 1 + 4

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        vertices: torch.Tensor,  # (N, 2)
        edge_index: torch.Tensor,  # (2, E)
        seg_probs: torch.Tensor,  # (1, num_classes, H, W)
        orientation: Optional[torch.Tensor],  # (1, 2, H, W) or None
        node_degree: torch.Tensor,  # (N,)
        node_boundary_dist: torch.Tensor,  # (N,)
        image_size: int = 512,
    ) -> torch.Tensor:
        """
        Extract edge features.

        Args:
            vertices: (N, 2) vertex coordinates
            edge_index: (2, E) edge indices
            seg_probs: (1, num_classes, H, W) segmentation probabilities
            orientation: (1, 2, H, W) orientation field (cos, sin) or None
            node_degree: (N,) degree of each node
            node_boundary_dist: (N,) boundary distance of each node
            image_size: Image size for normalization

        Returns:
            edge_features: (E, hidden_dim) edge embeddings
        """
        E = edge_index.shape[1]
        device = vertices.device

        src, dst = edge_index[0], edge_index[1]
        v_src = vertices[src]  # (E, 2)
        v_dst = vertices[dst]  # (E, 2)

        # 1. Geometry features
        direction = v_dst - v_src  # (E, 2)
        length = torch.norm(direction, dim=1, keepdim=True)  # (E, 1)
        length_norm = length / (image_size * math.sqrt(2))  # Normalize by diagonal

        direction_unit = direction / (length + 1e-6)  # (E, 2)

        midpoint = (v_src + v_dst) / 2  # (E, 2)
        midpoint_norm = midpoint / image_size  # (E, 2) in [0, 1]

        geometry = torch.cat([
            direction_unit,  # 2
            length_norm,     # 1
            midpoint_norm,   # 2
        ], dim=1)  # (E, 5)

        # 2. Line evidence (sample along edge)
        line_evidence = self._sample_line_evidence(
            v_src, v_dst, seg_probs, orientation, image_size
        )  # (E, num_classes + 2)

        # 3. Endpoint context
        deg_src = node_degree[src]  # (E,)
        deg_dst = node_degree[dst]  # (E,)
        deg_max = node_degree.max() + 1e-6

        dist_src = node_boundary_dist[src]  # (E,)
        dist_dst = node_boundary_dist[dst]  # (E,)

        endpoint_ctx = torch.stack([
            deg_src / deg_max,
            deg_dst / deg_max,
            dist_src,
            dist_dst,
        ], dim=1)  # (E, 4)

        # Concatenate all features
        edge_feats = torch.cat([
            geometry,
            line_evidence,
            endpoint_ctx,
        ], dim=1)  # (E, input_dim)

        return self.mlp(edge_feats)  # (E, hidden_dim)

    def _sample_line_evidence(
        self,
        v_src: torch.Tensor,  # (E, 2)
        v_dst: torch.Tensor,  # (E, 2)
        seg_probs: torch.Tensor,  # (1, C, H, W)
        orientation: Optional[torch.Tensor],
        image_size: int,
    ) -> torch.Tensor:
        """Sample segmentation and orientation along edges."""
        E = v_src.shape[0]
        device = v_src.device
        num_classes = seg_probs.shape[1]

        # Generate sample points along each edge
        t = torch.linspace(0, 1, self.num_samples, device=device)  # (S,)
        t = t.view(1, -1, 1)  # (1, S, 1)

        v_src_exp = v_src.unsqueeze(1)  # (E, 1, 2)
        v_dst_exp = v_dst.unsqueeze(1)  # (E, 1, 2)

        samples = v_src_exp + t * (v_dst_exp - v_src_exp)  # (E, S, 2)

        # Convert to grid coords
        grid = (samples / image_size) * 2 - 1  # (E, S, 2)
        grid = grid.unsqueeze(0)  # (1, E, S, 2)

        # Expand seg_probs for batch sampling
        seg_probs_exp = seg_probs.expand(1, -1, -1, -1)

        # Sample segmentation
        seg_sampled = F.grid_sample(
            seg_probs_exp, grid, mode='bilinear', align_corners=True
        )  # (1, C, E, S)
        seg_sampled = seg_sampled.squeeze(0)  # (C, E, S)

        # Mean across samples
        seg_mean = seg_sampled.mean(dim=2).T  # (E, C)

        # Crease probability (M + V)
        # Assuming classes are M=0, V=1, B=2, U=3
        crease_prob = (seg_mean[:, 0] + seg_mean[:, 1]).unsqueeze(1)  # (E, 1)

        # Orientation alignment
        if orientation is not None:
            orient_sampled = F.grid_sample(
                orientation, grid, mode='bilinear', align_corners=True
            )  # (1, 2, E, S)
            orient_sampled = orient_sampled.squeeze(0)  # (2, E, S)

            # Edge direction
            edge_dir = v_dst - v_src  # (E, 2)
            edge_dir = edge_dir / (torch.norm(edge_dir, dim=1, keepdim=True) + 1e-6)
            edge_dir = edge_dir.T.unsqueeze(2)  # (2, E, 1)

            # Cosine similarity (orientation is bidirectional, so use abs)
            dot = (orient_sampled * edge_dir).sum(dim=0)  # (E, S)
            align = dot.abs().mean(dim=1, keepdim=True)  # (E, 1)
        else:
            align = torch.zeros(E, 1, device=device)

        return torch.cat([seg_mean, crease_prob, align], dim=1)  # (E, C+2)


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for vertex coordinates."""

    def __init__(self, d_model: int, max_size: int = 512):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size

        # Create 1D encodings for x and y
        pe = torch.zeros(max_size, d_model // 2)
        position = torch.arange(0, max_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float() *
            (-math.log(10000.0) / (d_model // 2))
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) pixel coordinates

        Returns:
            encoding: (N, d_model) positional encoding
        """
        # Clamp coordinates to valid range
        x = coords[:, 0].long().clamp(0, self.max_size - 1)
        y = coords[:, 1].long().clamp(0, self.max_size - 1)

        # Look up encodings
        enc_x = self.pe[x]  # (N, d_model // 2)
        enc_y = self.pe[y]  # (N, d_model // 2)

        return torch.cat([enc_x, enc_y], dim=1)  # (N, d_model)
