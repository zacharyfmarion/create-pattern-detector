"""
Graph neural network layers for crease pattern graph processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer.

    Computes attention-weighted message passing between connected nodes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension (per head if concat=True)
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat: If True, concatenate heads; if False, average them
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        # Linear transformation for each head
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)

        # Attention parameters for each head
        self.a_src = nn.Parameter(torch.zeros(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, out_features))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Node features (N, in_features)
            edge_index: Edge indices (2, E) where edge_index[0] is source, edge_index[1] is target
            return_attention: Whether to return attention weights

        Returns:
            out: Updated node features (N, out_features * num_heads) or (N, out_features)
            attention: Attention weights (E, num_heads) if return_attention=True
        """
        N = x.size(0)

        # Linear transformation: (N, in_features) -> (N, num_heads, out_features)
        h = self.W(x).view(N, self.num_heads, self.out_features)

        # Source and target node indices
        src, dst = edge_index[0], edge_index[1]

        # Compute attention scores
        # (E, num_heads, out_features) * (num_heads, out_features) -> (E, num_heads)
        e_src = (h[src] * self.a_src).sum(dim=-1)
        e_dst = (h[dst] * self.a_dst).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)  # (E, num_heads)

        # Softmax over incoming edges for each node
        attention = self._sparse_softmax(e, dst, N)  # (E, num_heads)
        attention = self.dropout(attention)

        # Aggregate messages
        # (E, num_heads, out_features)
        messages = h[src] * attention.unsqueeze(-1)

        # Sum messages for each target node
        out = torch.zeros(N, self.num_heads, self.out_features, device=x.device)
        dst_expanded = dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.out_features)
        out.scatter_add_(0, dst_expanded, messages)

        # Combine heads
        if self.concat:
            out = out.view(N, self.num_heads * self.out_features)
        else:
            out = out.mean(dim=1)

        if return_attention:
            return out, attention
        return out, None

    def _sparse_softmax(
        self,
        e: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute softmax over sparse edges grouped by target node.

        Args:
            e: Edge attention scores (E, num_heads)
            index: Target node indices (E,)
            num_nodes: Number of nodes

        Returns:
            Softmax attention weights (E, num_heads)
        """
        # Subtract max for numerical stability
        e_max = torch.zeros(num_nodes, e.size(1), device=e.device)
        e_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(e), e, reduce='amax', include_self=False)
        e = e - e_max[index]

        # Compute exp
        exp_e = torch.exp(e)

        # Sum exp for each target node
        sum_exp = torch.zeros(num_nodes, e.size(1), device=e.device)
        sum_exp.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_e), exp_e)

        # Normalize
        return exp_e / (sum_exp[index] + 1e-8)


class EdgeUpdateLayer(nn.Module):
    """
    Layer that updates edge features based on endpoint node features.
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2 * node_features + edge_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, edge_features),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update edge features.

        Args:
            node_features: Node features (N, node_dim)
            edge_features: Edge features (E, edge_dim)
            edge_index: Edge indices (2, E)

        Returns:
            Updated edge features (E, edge_dim)
        """
        src, dst = edge_index[0], edge_index[1]

        # Concatenate source node, target node, and current edge features
        combined = torch.cat([
            node_features[src],
            node_features[dst],
            edge_features,
        ], dim=-1)

        return self.mlp(combined)


class GraphConvBlock(nn.Module):
    """
    Graph convolution block with attention, normalization, and residual connection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gat = GraphAttentionLayer(
            in_features=in_features,
            out_features=out_features // num_heads,
            num_heads=num_heads,
            dropout=dropout,
            concat=True,
        )

        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Node features (N, in_features)
            edge_index: Edge indices (2, E)

        Returns:
            Updated node features (N, out_features)
        """
        residual = self.residual(x)

        out, _ = self.gat(x, edge_index)
        out = self.dropout(out)
        out = self.norm(out + residual)

        return out
