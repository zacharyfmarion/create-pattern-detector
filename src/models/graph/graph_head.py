"""
Graph head for crease pattern edge prediction and assignment.

Takes vertex positions and image features, uses a GNN to process
the graph structure, and predicts edge existence and M/V/B/U assignments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

from .layers import GraphConvBlock, EdgeUpdateLayer


class VertexFeatureExtractor(nn.Module):
    """
    Extract features at vertex positions from image feature maps.

    Uses bilinear interpolation to sample features at exact vertex coordinates,
    then processes with a small MLP.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
        use_position_encoding: bool = True,
        max_vertices: int = 500,
    ):
        """
        Args:
            in_channels: Number of input feature channels
            out_channels: Output feature dimension per vertex
            use_position_encoding: Whether to add positional encoding
            max_vertices: Maximum number of vertices (for position encoding)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_position_encoding = use_position_encoding

        # MLP to process sampled features
        self.feature_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        # Position encoding
        if use_position_encoding:
            self.pos_encoder = PositionalEncoding2D(out_channels // 2)
            self.pos_proj = nn.Linear(out_channels, out_channels)

    def forward(
        self,
        features: torch.Tensor,
        vertices: torch.Tensor,
        image_size: int = 512,
    ) -> torch.Tensor:
        """
        Extract features at vertex positions.

        Args:
            features: Image features (B, C, H, W)
            vertices: Vertex positions (B, N, 2) in pixel coordinates (x, y)
            image_size: Original image size for normalization

        Returns:
            Vertex features (B, N, out_channels)
        """
        B, C, H, W = features.shape
        _, N, _ = vertices.shape

        # Normalize vertices to [-1, 1] for grid_sample
        # vertices are (x, y) in [0, image_size]
        grid = vertices.clone()
        grid[..., 0] = 2 * grid[..., 0] / (image_size - 1) - 1  # x
        grid[..., 1] = 2 * grid[..., 1] / (image_size - 1) - 1  # y
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)

        # Sample features at vertex positions
        # grid_sample expects (B, C, H, W) and grid (B, H_out, W_out, 2)
        sampled = F.grid_sample(
            features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # (B, C, N, 1)

        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C)

        # Process through MLP
        vertex_features = self.feature_mlp(sampled)  # (B, N, out_channels)

        # Add positional encoding
        if self.use_position_encoding:
            pos_enc = self.pos_encoder(vertices / image_size)  # Normalize to [0, 1]
            vertex_features = self.pos_proj(vertex_features + pos_enc)

        return vertex_features


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for vertex positions.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model

        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 2) normalized positions in [0, 1]

        Returns:
            Positional encodings (B, N, d_model * 2)
        """
        B, N, _ = positions.shape

        # Separate x and y
        x = positions[..., 0:1] * 1000  # Scale for more variation
        y = positions[..., 1:2] * 1000

        # Compute sinusoidal encodings
        pe_x = torch.zeros(B, N, self.d_model, device=positions.device)
        pe_y = torch.zeros(B, N, self.d_model, device=positions.device)

        pe_x[..., 0::2] = torch.sin(x * self.div_term)
        pe_x[..., 1::2] = torch.cos(x * self.div_term)
        pe_y[..., 0::2] = torch.sin(y * self.div_term)
        pe_y[..., 1::2] = torch.cos(y * self.div_term)

        return torch.cat([pe_x, pe_y], dim=-1)


class EdgeClassifier(nn.Module):
    """
    Classify edges based on endpoint features.

    Predicts:
    1. Edge existence (binary)
    2. Edge assignment (M=0, V=1, B=2, U=3)
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 4,
    ):
        super().__init__()

        self.edge_features = edge_features
        self.num_classes = num_classes

        # Initial edge feature computation
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * node_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, edge_features),
        )

        # Edge existence classifier
        self.existence_head = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Edge assignment classifier
        self.assignment_head = nn.Sequential(
            nn.Linear(edge_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Classify edges.

        Args:
            node_features: Node features (N, node_dim)
            edge_index: Edge indices (2, E)

        Returns:
            Dictionary with:
            - 'edge_existence': (E,) logits for edge existence
            - 'edge_assignment': (E, num_classes) logits for assignment
        """
        src, dst = edge_index[0], edge_index[1]

        # Compute edge features from endpoint nodes
        edge_features = torch.cat([
            node_features[src],
            node_features[dst],
        ], dim=-1)

        edge_features = self.edge_encoder(edge_features)

        # Predict existence and assignment
        existence = self.existence_head(edge_features).squeeze(-1)
        assignment = self.assignment_head(edge_features)

        return {
            'edge_existence': existence,
            'edge_assignment': assignment,
            'edge_features': edge_features,
        }


class GraphHead(nn.Module):
    """
    Graph head for crease pattern processing.

    Architecture:
    1. Extract features at vertex positions
    2. Process graph with GNN layers
    3. Classify edges for existence and assignment
    """

    def __init__(
        self,
        in_channels: int = 480,  # HRNet output channels
        vertex_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 4,
    ):
        """
        Args:
            in_channels: Input feature channels from backbone
            vertex_dim: Initial vertex feature dimension
            hidden_dim: Hidden dimension for GNN layers
            num_gnn_layers: Number of GNN layers
            num_heads: Number of attention heads in GAT
            dropout: Dropout rate
            num_classes: Number of edge assignment classes (M, V, B, U)
        """
        super().__init__()

        self.vertex_dim = vertex_dim
        self.hidden_dim = hidden_dim

        # Vertex feature extractor
        self.vertex_extractor = VertexFeatureExtractor(
            in_channels=in_channels,
            out_channels=vertex_dim,
            use_position_encoding=True,
        )

        # Project to GNN hidden dimension
        self.input_proj = nn.Linear(vertex_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvBlock(
                in_features=hidden_dim,
                out_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_gnn_layers)
        ])

        # Edge classifier
        self.edge_classifier = EdgeClassifier(
            node_features=hidden_dim,
            edge_features=hidden_dim // 2,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

    def forward(
        self,
        features: torch.Tensor,
        vertices: torch.Tensor,
        edge_index: torch.Tensor,
        image_size: int = 512,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: Image features from backbone (B, C, H, W)
            vertices: Vertex positions (B, N, 2) in pixel coordinates
            edge_index: Candidate edge indices (2, E) - indices into vertices
            image_size: Original image size

        Returns:
            Dictionary with:
            - 'edge_existence': (E,) logits for edge existence
            - 'edge_assignment': (E, num_classes) logits for assignment
            - 'node_features': (N, hidden_dim) final node features
        """
        # For now, only support batch size 1 for simplicity
        # (graphs have different sizes, need batching strategy)
        if features.size(0) != 1:
            raise ValueError("GraphHead currently only supports batch size 1")

        # Extract vertex features
        vertex_features = self.vertex_extractor(
            features, vertices, image_size
        )  # (1, N, vertex_dim)

        # Remove batch dimension for GNN processing
        vertex_features = vertex_features.squeeze(0)  # (N, vertex_dim)

        # Project to hidden dimension
        x = self.input_proj(vertex_features)  # (N, hidden_dim)

        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)

        # Classify edges
        edge_outputs = self.edge_classifier(x, edge_index)

        return {
            'edge_existence': edge_outputs['edge_existence'],
            'edge_assignment': edge_outputs['edge_assignment'],
            'node_features': x,
        }

    def forward_batch(
        self,
        features_list: List[torch.Tensor],
        vertices_list: List[torch.Tensor],
        edge_index_list: List[torch.Tensor],
        image_size: int = 512,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Process a batch of graphs (each potentially different size).

        Args:
            features_list: List of image features (C, H, W)
            vertices_list: List of vertex positions (N_i, 2)
            edge_index_list: List of edge indices (2, E_i)
            image_size: Original image size

        Returns:
            List of output dictionaries, one per graph
        """
        outputs = []
        for features, vertices, edge_index in zip(
            features_list, vertices_list, edge_index_list
        ):
            # Add batch dimension
            features = features.unsqueeze(0)
            vertices = vertices.unsqueeze(0)

            out = self.forward(features, vertices, edge_index, image_size)
            outputs.append(out)

        return outputs
