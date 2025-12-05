"""
Graph Neural Network layers for the Graph Head.

Implements edge-aware message passing with:
- Explicit edge feature updates
- Attention-based node updates
- Residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EdgeUpdateLayer(nn.Module):
    """
    Update edge features based on endpoint node features.

    h_edge' = h_edge + MLP(concat(h_src, h_dst, h_edge))
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        hidden_dim = hidden_dim or edge_dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim),
        )

    def forward(
        self,
        node_features: torch.Tensor,  # (N, node_dim)
        edge_features: torch.Tensor,  # (E, edge_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:
        """
        Update edge features.

        Returns:
            updated_edge_features: (E, edge_dim)
        """
        src, dst = edge_index[0], edge_index[1]

        h_src = node_features[src]  # (E, node_dim)
        h_dst = node_features[dst]  # (E, node_dim)

        # Concatenate and transform
        combined = torch.cat([h_src, h_dst, edge_features], dim=1)
        update = self.mlp(combined)  # (E, edge_dim)

        # Residual connection
        return edge_features + update


class NodeUpdateLayer(nn.Module):
    """
    Update node features using attention over incident edges.

    For each node i:
        1. Compute attention scores over incident edges
        2. Aggregate edge features weighted by attention
        3. Update node feature with residual
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"

        # Attention computation
        self.attention = nn.Linear(node_dim + edge_dim, num_heads)

        # Edge value transformation
        self.edge_value = nn.Linear(edge_dim, node_dim)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,  # (N, node_dim)
        edge_features: torch.Tensor,  # (E, edge_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> torch.Tensor:
        """
        Update node features.

        Returns:
            updated_node_features: (N, node_dim)
        """
        N = node_features.shape[0]
        E = edge_index.shape[1]
        device = node_features.device

        src, dst = edge_index[0], edge_index[1]

        # For each edge, compute attention score based on dst node and edge
        # (dst is the receiving node in the message)
        h_dst = node_features[dst]  # (E, node_dim)

        # Attention input: concat node and edge features
        attn_input = torch.cat([h_dst, edge_features], dim=1)  # (E, node_dim + edge_dim)
        attn_scores = self.attention(attn_input)  # (E, num_heads)

        # Transform edge features to values
        edge_values = self.edge_value(edge_features)  # (E, node_dim)

        # Apply attention with sparse softmax per destination node
        # Group by destination node
        messages = self._aggregate_with_attention(
            edge_values, attn_scores, dst, N
        )  # (N, node_dim)

        # Apply output MLP with residual
        update = self.output(messages)

        return node_features + update

    def _aggregate_with_attention(
        self,
        values: torch.Tensor,  # (E, node_dim)
        scores: torch.Tensor,  # (E, num_heads)
        indices: torch.Tensor,  # (E,) destination indices
        num_nodes: int,
    ) -> torch.Tensor:
        """Aggregate values with attention, grouped by destination node."""
        device = values.device
        dtype = values.dtype  # Match dtype for AMP compatibility
        E = values.shape[0]

        if E == 0:
            return torch.zeros(num_nodes, values.shape[1], dtype=dtype, device=device)

        # Average scores across heads for simplicity
        # Cast to match dtype for AMP compatibility with scatter ops
        scores_mean = scores.mean(dim=1).to(dtype)  # (E,)

        # Compute max per destination for numerical stability
        max_scores = torch.full((num_nodes,), float('-inf'), dtype=dtype, device=device)
        max_scores.scatter_reduce_(
            0, indices, scores_mean, reduce='amax', include_self=False
        )
        # Handle nodes with no incoming edges
        max_scores = torch.where(
            max_scores == float('-inf'),
            torch.zeros_like(max_scores),
            max_scores
        )
        max_scores_expanded = max_scores[indices]  # (E,)

        # Softmax weights
        exp_scores = torch.exp(scores_mean - max_scores_expanded)  # (E,)

        # Sum of exp_scores per destination
        sum_exp = torch.zeros(num_nodes, dtype=dtype, device=device)
        sum_exp.scatter_add_(0, indices, exp_scores.to(dtype))
        sum_exp_expanded = sum_exp[indices] + 1e-6  # (E,)

        attn_weights = exp_scores / sum_exp_expanded  # (E,)
        attn_weights = self.dropout(attn_weights)

        # Weighted values - ensure dtype matches for scatter
        weighted_values = (values * attn_weights.unsqueeze(1)).to(dtype)  # (E, node_dim)

        # Aggregate by destination
        output = torch.zeros(num_nodes, values.shape[1], dtype=dtype, device=device)
        output.scatter_add_(
            0,
            indices.unsqueeze(1).expand(-1, values.shape[1]),
            weighted_values
        )

        return output


class GraphConvBlock(nn.Module):
    """
    Single graph convolution block with edge and node updates.

    Sequence:
    1. Edge update (based on endpoint nodes)
    2. Node update (attention over edges)
    3. Layer normalization
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.edge_update = EdgeUpdateLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.node_update = NodeUpdateLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.node_norm = nn.LayerNorm(node_dim)
        self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        node_features: torch.Tensor,  # (N, node_dim)
        edge_features: torch.Tensor,  # (E, edge_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply one graph convolution block.

        Returns:
            (updated_node_features, updated_edge_features)
        """
        # Edge update
        edge_features = self.edge_update(node_features, edge_features, edge_index)
        edge_features = self.edge_norm(edge_features)

        # Node update
        node_features = self.node_update(node_features, edge_features, edge_index)
        node_features = self.node_norm(node_features)

        return node_features, edge_features


class GraphNetwork(nn.Module):
    """
    Full graph neural network with multiple conv blocks.
    """

    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            GraphConvBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        node_features: torch.Tensor,  # (N, node_dim)
        edge_features: torch.Tensor,  # (E, edge_dim)
        edge_index: torch.Tensor,  # (2, E)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply all graph conv layers.

        Returns:
            (final_node_features, final_edge_features)
        """
        for layer in self.layers:
            node_features, edge_features = layer(
                node_features, edge_features, edge_index
            )

        return node_features, edge_features
