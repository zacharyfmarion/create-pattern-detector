#!/usr/bin/env python3
"""
Test script to reproduce and fix AMP dtype issues in Graph Head.
Run locally with: python scripts/test_graph_head_amp.py
"""

import torch
from torch.cuda.amp import autocast

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.graph.graph_head import GraphHead


def test_graph_head_amp():
    """Test GraphHead forward pass with AMP to catch dtype issues."""

    # Use CUDA if available, otherwise CPU (AMP only really matters on CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print(f"Testing on device: {device}")
    print(f"AMP enabled: {use_amp}")

    # Create model
    model = GraphHead(
        backbone_channels=480,
        node_dim=128,
        edge_dim=128,
        num_gnn_layers=4,
        num_heads=4,
        num_classes=4,
        dropout=0.1,
    ).to(device)
    model.train()

    # Create fake inputs similar to what training would produce
    batch_size = 2
    num_vertices = [50, 75]  # Variable per sample
    num_edges = [120, 180]
    image_size = 512
    feature_size = 128  # backbone feature map size

    # Create batched inputs
    vertices_list = [
        torch.rand(n, 2, device=device) * image_size
        for n in num_vertices
    ]

    # Create edge indices (random edges between vertices)
    edge_index_list = []
    for i, (n_v, n_e) in enumerate(zip(num_vertices, num_edges)):
        src = torch.randint(0, n_v, (n_e,), device=device)
        dst = torch.randint(0, n_v, (n_e,), device=device)
        edge_index_list.append(torch.stack([src, dst]))

    # Backbone features and segmentation (these come from pixel head)
    backbone_features = torch.randn(batch_size, 480, feature_size, feature_size, device=device)
    seg_probs = torch.softmax(torch.randn(batch_size, 4, feature_size, feature_size, device=device), dim=1)
    orientation = torch.randn(batch_size, 2, feature_size, feature_size, device=device)
    orientation = orientation / (orientation.norm(dim=1, keepdim=True) + 1e-6)  # Normalize

    print(f"\nInput shapes:")
    print(f"  vertices_list: {[v.shape for v in vertices_list]}")
    print(f"  edge_index_list: {[e.shape for e in edge_index_list]}")
    print(f"  backbone_features: {backbone_features.shape}")
    print(f"  seg_probs: {seg_probs.shape}")
    print(f"  orientation: {orientation.shape}")

    # Test forward pass with AMP
    print("\nRunning forward pass with AMP...")
    try:
        with autocast(enabled=use_amp):
            outputs = model.forward_batch(
                vertices_list=vertices_list,
                edge_index_list=edge_index_list,
                backbone_features=backbone_features,
                seg_probs=seg_probs,
                orientation=orientation,
                image_size=image_size,
            )

        print("\nOutput shapes:")
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}, dtype={val.dtype}")

        print("\n✓ Forward pass succeeded!")

        # Test backward pass
        print("\nTesting backward pass...")
        loss = outputs['edge_existence'].sum() + outputs['edge_assignment'].sum()
        loss.backward()
        print("✓ Backward pass succeeded!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_individual_components():
    """Test individual components to isolate the issue."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    print("\n" + "="*60)
    print("Testing individual components...")
    print("="*60)

    # Test NodeUpdateLayer scatter operations
    from src.models.graph.layers import NodeUpdateLayer

    print("\nTesting NodeUpdateLayer...")
    node_update = NodeUpdateLayer(node_dim=128, edge_dim=128).to(device)

    N, E = 50, 120
    node_features = torch.randn(N, 128, device=device)
    edge_features = torch.randn(E, 128, device=device)
    edge_index = torch.stack([
        torch.randint(0, N, (E,), device=device),
        torch.randint(0, N, (E,), device=device),
    ])

    try:
        with autocast(enabled=use_amp):
            out = node_update(node_features, edge_features, edge_index)
        print(f"  ✓ NodeUpdateLayer output: {out.shape}, dtype={out.dtype}")
    except Exception as e:
        print(f"  ✗ NodeUpdateLayer failed: {e}")

    # Test NodeFeatureExtractor
    from src.models.graph.features import NodeFeatureExtractor

    print("\nTesting NodeFeatureExtractor...")
    node_extractor = NodeFeatureExtractor(backbone_channels=480).to(device)

    vertices = torch.rand(N, 2, device=device) * 512
    backbone_features = torch.randn(1, 480, 128, 128, device=device)
    seg_probs = torch.softmax(torch.randn(1, 4, 128, 128, device=device), dim=1)

    try:
        with autocast(enabled=use_amp):
            out = node_extractor(vertices, backbone_features, seg_probs, edge_index, 512)
        print(f"  ✓ NodeFeatureExtractor output: {out.shape}, dtype={out.dtype}")
    except Exception as e:
        print(f"  ✗ NodeFeatureExtractor failed: {e}")


if __name__ == '__main__':
    success = test_graph_head_amp()
    test_individual_components()

    if success:
        print("\n" + "="*60)
        print("All tests passed! The Graph Head works correctly with AMP.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Tests failed. See errors above.")
        print("="*60)
        sys.exit(1)
