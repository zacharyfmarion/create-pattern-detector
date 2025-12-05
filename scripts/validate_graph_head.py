#!/usr/bin/env python3
"""
Incremental validation script for Graph Head.

Run this before full training to catch issues early.

Usage:
    python scripts/validate_graph_head.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.graph.layers import EdgeUpdateLayer, NodeUpdateLayer, GraphConvBlock, GraphNetwork
from src.models.graph.features import NodeFeatureExtractor, EdgeFeatureExtractor
from src.models.graph.graph_head import GraphHead
from src.models.losses.graph_loss import GraphLoss, compute_graph_metrics
from src.data.graph_labels import generate_graph_labels, load_gt_from_fold


def test_gnn_layers():
    """Test 1: GNN layer shapes and gradient flow."""
    print("\n" + "="*60)
    print("TEST 1: GNN Layers")
    print("="*60)

    node_dim, edge_dim = 128, 128
    N, E = 50, 100

    # Create dummy data
    node_features = torch.randn(N, node_dim, requires_grad=True)
    edge_features = torch.randn(E, edge_dim, requires_grad=True)
    edge_index = torch.randint(0, N, (2, E))

    # Test EdgeUpdateLayer
    print("\n[EdgeUpdateLayer]")
    edge_layer = EdgeUpdateLayer(node_dim, edge_dim)
    updated_edges = edge_layer(node_features, edge_features, edge_index)
    print(f"  Input shape: {edge_features.shape}")
    print(f"  Output shape: {updated_edges.shape}")
    assert updated_edges.shape == edge_features.shape, "Edge shape mismatch!"

    # Check gradient flow
    loss = updated_edges.sum()
    loss.backward(retain_graph=True)
    assert node_features.grad is not None, "No gradient to node features!"
    assert edge_features.grad is not None, "No gradient to edge features!"
    print("  ✓ Gradients flow correctly")

    # Reset grads
    node_features.grad = None
    edge_features.grad = None

    # Test NodeUpdateLayer
    print("\n[NodeUpdateLayer]")
    node_layer = NodeUpdateLayer(node_dim, edge_dim, num_heads=4)
    updated_nodes = node_layer(node_features, edge_features, edge_index)
    print(f"  Input shape: {node_features.shape}")
    print(f"  Output shape: {updated_nodes.shape}")
    assert updated_nodes.shape == node_features.shape, "Node shape mismatch!"

    loss = updated_nodes.sum()
    loss.backward(retain_graph=True)
    assert node_features.grad is not None, "No gradient to node features!"
    print("  ✓ Gradients flow correctly")

    # Test full GraphNetwork
    print("\n[GraphNetwork]")
    gnn = GraphNetwork(node_dim, edge_dim, num_layers=4)
    node_features = torch.randn(N, node_dim, requires_grad=True)
    edge_features = torch.randn(E, edge_dim, requires_grad=True)

    out_nodes, out_edges = gnn(node_features, edge_features, edge_index)
    print(f"  Node output shape: {out_nodes.shape}")
    print(f"  Edge output shape: {out_edges.shape}")

    loss = out_nodes.sum() + out_edges.sum()
    loss.backward()
    assert node_features.grad is not None, "No gradient to node features!"
    assert edge_features.grad is not None, "No gradient to edge features!"
    print("  ✓ Gradients flow through full GNN")

    print("\n✅ TEST 1 PASSED: GNN layers work correctly")
    return True


def test_feature_extractors():
    """Test 2: Feature extraction from backbone/pixel head outputs."""
    print("\n" + "="*60)
    print("TEST 2: Feature Extractors")
    print("="*60)

    N = 30  # vertices
    E = 60  # edges
    image_size = 512
    backbone_channels = 480

    # Create dummy inputs
    vertices = torch.rand(N, 2) * image_size
    edge_index = torch.randint(0, N, (2, E))
    backbone_features = torch.randn(1, backbone_channels, 32, 32)
    seg_probs = torch.softmax(torch.randn(1, 4, 128, 128), dim=1)
    orientation = torch.randn(1, 2, 128, 128)

    # Test NodeFeatureExtractor
    print("\n[NodeFeatureExtractor]")
    node_extractor = NodeFeatureExtractor(backbone_channels=backbone_channels, hidden_dim=128)
    node_features = node_extractor(
        vertices=vertices,
        backbone_features=backbone_features,
        seg_probs=seg_probs,
        edge_index=edge_index,
        image_size=image_size,
    )
    print(f"  Vertices: {vertices.shape}")
    print(f"  Node features: {node_features.shape}")
    assert node_features.shape == (N, 128), f"Expected (N, 128), got {node_features.shape}"
    assert not torch.isnan(node_features).any(), "NaN in node features!"
    print("  ✓ Node features extracted correctly")

    # Test EdgeFeatureExtractor
    print("\n[EdgeFeatureExtractor]")
    edge_extractor = EdgeFeatureExtractor(hidden_dim=128, num_samples=16)

    # Compute structural features
    node_degree = torch.zeros(N)
    node_degree.scatter_add_(0, edge_index[0], torch.ones(E))
    node_degree.scatter_add_(0, edge_index[1], torch.ones(E))
    node_boundary_dist = (image_size/2 - torch.abs(vertices - image_size/2).min(dim=1)[0]) / (image_size/2)

    edge_features = edge_extractor(
        vertices=vertices,
        edge_index=edge_index,
        seg_probs=seg_probs,
        orientation=orientation,
        node_degree=node_degree,
        node_boundary_dist=node_boundary_dist,
        image_size=image_size,
    )
    print(f"  Edge index: {edge_index.shape}")
    print(f"  Edge features: {edge_features.shape}")
    assert edge_features.shape == (E, 128), f"Expected (E, 128), got {edge_features.shape}"
    assert not torch.isnan(edge_features).any(), "NaN in edge features!"
    print("  ✓ Edge features extracted correctly")

    print("\n✅ TEST 2 PASSED: Feature extractors work correctly")
    return True


def test_graph_head_forward():
    """Test 3: Full GraphHead forward pass."""
    print("\n" + "="*60)
    print("TEST 3: GraphHead Forward Pass")
    print("="*60)

    N, E = 40, 80
    image_size = 512

    graph_head = GraphHead(
        backbone_channels=480,
        node_dim=128,
        edge_dim=128,
        num_gnn_layers=4,
        num_heads=4,
    )

    # Create inputs
    vertices = torch.rand(N, 2) * image_size
    edge_index = torch.randint(0, N, (2, E))
    backbone_features = torch.randn(1, 480, 32, 32)
    seg_probs = torch.softmax(torch.randn(1, 4, 128, 128), dim=1)

    print(f"\n  Vertices: {N}")
    print(f"  Edges: {E}")
    print(f"  Backbone features: {backbone_features.shape}")

    outputs = graph_head(
        vertices=vertices,
        edge_index=edge_index,
        backbone_features=backbone_features,
        seg_probs=seg_probs,
        image_size=image_size,
    )

    print(f"\n  Outputs:")
    print(f"    edge_existence: {outputs['edge_existence'].shape}")
    print(f"    edge_assignment: {outputs['edge_assignment'].shape}")
    print(f"    vertex_offset: {outputs['vertex_offset'].shape}")

    # Check shapes
    assert outputs['edge_existence'].shape == (E,), "Edge existence shape wrong"
    assert outputs['edge_assignment'].shape == (E, 4), "Edge assignment shape wrong"
    assert outputs['vertex_offset'].shape == (N, 2), "Vertex offset shape wrong"

    # Check no NaN
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            assert not torch.isnan(val).any(), f"NaN in {key}!"

    print("\n  ✓ All output shapes correct")
    print("  ✓ No NaN values")

    # Test gradient flow through full model
    print("\n  Testing gradient flow...")
    loss = outputs['edge_existence'].sum() + outputs['edge_assignment'].sum() + outputs['vertex_offset'].sum()
    loss.backward()

    # Check some parameters have gradients
    has_grad = False
    for name, param in graph_head.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients in model parameters!"
    print("  ✓ Gradients flow through full model")

    print("\n✅ TEST 3 PASSED: GraphHead forward pass works")
    return True


def test_loss_computation():
    """Test 4: Loss computation and gradient flow."""
    print("\n" + "="*60)
    print("TEST 4: Loss Computation")
    print("="*60)

    E, N = 50, 20

    # Create predictions (with grad)
    predictions = {
        'edge_existence': torch.randn(E, requires_grad=True),
        'edge_assignment': torch.randn(E, 4, requires_grad=True),
        'vertex_offset': torch.randn(N, 2, requires_grad=True),
    }

    # Create targets (various scenarios)
    scenarios = [
        ("All positive edges", torch.ones(E), torch.randint(0, 4, (E,))),
        ("All negative edges", torch.zeros(E), torch.zeros(E, dtype=torch.long)),
        ("Mixed 50/50", (torch.rand(E) > 0.5).float(), torch.randint(0, 4, (E,))),
        ("Sparse positive (10%)", (torch.rand(E) > 0.9).float(), torch.randint(0, 4, (E,))),
    ]

    loss_fn = GraphLoss()

    for name, edge_exist, edge_assign in scenarios:
        print(f"\n  [{name}]")

        targets = {
            'edge_existence': edge_exist,
            'edge_assignment': edge_assign,
            'vertex_offset': torch.randn(N, 2),
            'vertex_matched': torch.ones(N),
        }

        losses = loss_fn(predictions, targets)

        print(f"    Total: {losses['loss'].item():.4f}")
        print(f"    Existence: {losses['existence_loss'].item():.4f}")
        print(f"    Assignment: {losses['assignment_loss'].item():.4f}")
        print(f"    Refinement: {losses['refinement_loss'].item():.4f}")

        assert not torch.isnan(losses['loss']), f"NaN loss in {name}!"
        assert not torch.isinf(losses['loss']), f"Inf loss in {name}!"
        assert losses['loss'].item() >= 0, f"Negative loss in {name}!"

    print("\n  ✓ All loss scenarios computed without NaN/Inf")

    # Test gradient flow
    print("\n  Testing gradient flow through loss...")
    predictions = {
        'edge_existence': torch.randn(E, requires_grad=True),
        'edge_assignment': torch.randn(E, 4, requires_grad=True),
        'vertex_offset': torch.randn(N, 2, requires_grad=True),
    }
    targets = {
        'edge_existence': (torch.rand(E) > 0.5).float(),
        'edge_assignment': torch.randint(0, 4, (E,)),
        'vertex_offset': torch.randn(N, 2),
        'vertex_matched': torch.ones(N),
    }

    losses = loss_fn(predictions, targets)
    losses['loss'].backward()

    assert predictions['edge_existence'].grad is not None, "No grad for edge_existence!"
    assert predictions['edge_assignment'].grad is not None, "No grad for edge_assignment!"
    assert predictions['vertex_offset'].grad is not None, "No grad for vertex_offset!"
    print("  ✓ Gradients flow through loss")

    print("\n✅ TEST 4 PASSED: Loss computation works correctly")
    return True


def test_label_generation():
    """Test 5: GT label generation and matching."""
    print("\n" + "="*60)
    print("TEST 5: Label Generation")
    print("="*60)

    # Scenario 1: Perfect match
    print("\n  [Scenario 1: Perfect vertex/edge match]")
    candidate_vertices = torch.tensor([
        [100.0, 100.0],
        [200.0, 100.0],
        [150.0, 200.0],
    ])
    candidate_edges = torch.tensor([[0, 1], [1, 2]]).T

    gt_vertices = candidate_vertices.clone()
    gt_edges = candidate_edges.clone()
    gt_assignments = torch.tensor([0, 1])  # M, V

    labels = generate_graph_labels(
        candidate_vertices, candidate_edges,
        gt_vertices, gt_edges, gt_assignments,
        vertex_match_threshold=10.0,
    )

    print(f"    Edge existence: {labels.edge_existence.tolist()}")
    print(f"    Edge assignment: {labels.edge_assignment.tolist()}")
    print(f"    Vertex matched: {labels.vertex_matched.tolist()}")

    assert labels.edge_existence.sum() == 2, "Should match 2 edges"
    assert labels.vertex_matched.sum() == 3, "Should match 3 vertices"
    assert labels.edge_assignment[0] == 0, "First edge should be M"
    assert labels.edge_assignment[1] == 1, "Second edge should be V"
    print("    ✓ Perfect match works")

    # Scenario 2: Extra candidate edges
    print("\n  [Scenario 2: Over-complete candidate graph]")
    candidate_vertices = torch.tensor([
        [100.0, 100.0],
        [200.0, 100.0],
        [150.0, 200.0],
        [300.0, 150.0],  # Extra vertex
    ])
    candidate_edges = torch.tensor([
        [0, 1, 1, 0, 2],  # 5 edges, only 2 real
        [1, 2, 3, 2, 3],
    ])

    labels = generate_graph_labels(
        candidate_vertices, candidate_edges,
        gt_vertices, gt_edges, gt_assignments,
        vertex_match_threshold=10.0,
    )

    print(f"    Edge existence: {labels.edge_existence.tolist()}")
    print(f"    Vertex matched: {labels.vertex_matched.tolist()}")

    assert labels.edge_existence.sum() == 2, f"Should match 2 edges, got {labels.edge_existence.sum()}"
    assert labels.edge_existence[0] == 1 and labels.edge_existence[1] == 1, "First two edges should match"
    assert labels.edge_existence[2:].sum() == 0, "Extra edges should not match"
    print("    ✓ Over-complete graph correctly labeled")

    # Scenario 3: Slightly offset vertices
    print("\n  [Scenario 3: Offset vertices within threshold]")
    candidate_vertices = gt_vertices + torch.randn_like(gt_vertices) * 3  # ~3 pixel noise
    candidate_edges = gt_edges.clone()

    labels = generate_graph_labels(
        candidate_vertices, candidate_edges,
        gt_vertices, gt_edges, gt_assignments,
        vertex_match_threshold=10.0,
    )

    print(f"    Vertex matched: {labels.vertex_matched.tolist()}")
    print(f"    Vertex offsets (L2): {torch.norm(labels.vertex_offset, dim=1).tolist()}")

    assert labels.vertex_matched.sum() == 3, "All vertices should match within threshold"
    assert (torch.norm(labels.vertex_offset, dim=1) < 10).all(), "Offsets should be < threshold"
    print("    ✓ Noisy vertices matched correctly")

    print("\n✅ TEST 5 PASSED: Label generation works correctly")
    return True


def test_overfit_single_sample(checkpoint_path: str = None):
    """Test 6: Overfit on a single sample (the ultimate sanity check)."""
    print("\n" + "="*60)
    print("TEST 6: Overfit on Single Sample")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")

    # Create a simple synthetic sample
    N, E = 20, 40
    image_size = 512

    # Fixed random seed for reproducibility
    torch.manual_seed(42)

    # Create graph head
    graph_head = GraphHead(
        backbone_channels=480,
        node_dim=64,  # Smaller for faster test
        edge_dim=64,
        num_gnn_layers=2,
    ).to(device)

    # Create fixed inputs
    vertices = (torch.rand(N, 2) * (image_size - 100) + 50).to(device)
    edge_index = torch.randint(0, N, (2, E)).to(device)
    backbone_features = torch.randn(1, 480, 32, 32).to(device)
    seg_probs = torch.softmax(torch.randn(1, 4, 128, 128), dim=1).to(device)

    # Create fixed targets (memorizable pattern)
    edge_existence = (torch.arange(E) % 2 == 0).float().to(device)  # Alternate pattern
    edge_assignment = (torch.arange(E) % 4).to(device)  # Cycle through 0,1,2,3
    vertex_offset = torch.zeros(N, 2).to(device)  # No offset
    vertex_matched = torch.ones(N).to(device)

    targets = {
        'edge_existence': edge_existence,
        'edge_assignment': edge_assignment,
        'vertex_offset': vertex_offset,
        'vertex_matched': vertex_matched,
    }

    # Setup training
    optimizer = torch.optim.Adam(graph_head.parameters(), lr=1e-3)
    loss_fn = GraphLoss()

    print("\n  Training for 100 steps...")
    initial_loss = None

    for step in range(100):
        optimizer.zero_grad()

        outputs = graph_head(
            vertices=vertices,
            edge_index=edge_index,
            backbone_features=backbone_features,
            seg_probs=seg_probs,
            image_size=image_size,
        )

        losses = loss_fn(outputs, targets)
        loss = losses['loss']

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            metrics = compute_graph_metrics(outputs, targets)
            print(f"    Step {step:3d}: loss={loss.item():.4f}, exist_f1={metrics['existence_f1']:.3f}, assign_acc={metrics['assignment_accuracy']:.3f}")

    final_loss = loss.item()
    final_metrics = compute_graph_metrics(outputs, targets)

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
    print(f"  Final existence F1: {final_metrics['existence_f1']:.3f}")
    print(f"  Final assignment acc: {final_metrics['assignment_accuracy']:.3f}")

    # Check that we're overfitting
    if final_loss < initial_loss * 0.5:
        print("\n  ✓ Loss decreased significantly (good!)")
    else:
        print("\n  ⚠ Loss didn't decrease much - might be an issue")

    if final_metrics['existence_f1'] > 0.8:
        print("  ✓ Existence F1 > 0.8 (learning!)")
    else:
        print("  ⚠ Existence F1 low - check edge existence head")

    if final_metrics['assignment_accuracy'] > 0.5:
        print("  ✓ Assignment accuracy > 0.5 (learning!)")
    else:
        print("  ⚠ Assignment accuracy low - check assignment head")

    success = final_loss < initial_loss * 0.5 and final_metrics['existence_f1'] > 0.7

    if success:
        print("\n✅ TEST 6 PASSED: Model can overfit on single sample")
    else:
        print("\n⚠ TEST 6 WARNING: Overfitting not as expected - investigate")

    return success


def test_with_real_data(checkpoint_path: str, fold_path: str = None):
    """Test 7: Integration with real pixel head outputs."""
    print("\n" + "="*60)
    print("TEST 7: Integration with Real Pixel Head")
    print("="*60)

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("\n  ⚠ Skipping: No checkpoint provided")
        print("  Run with --checkpoint path/to/pixel_head.pt")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Load pixel head model
    from src.models import CreasePatternDetector
    from src.postprocessing.graph_extraction import CandidateGraphExtractor

    print("\n  Loading pixel head model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = CreasePatternDetector(
        backbone_name="hrnet_w32",
        num_seg_classes=5,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  ✓ Model loaded")

    # Create graph head
    graph_head = GraphHead(
        backbone_channels=480,
        node_dim=128,
        edge_dim=128,
        num_gnn_layers=4,
    ).to(device)
    print("  ✓ Graph head created")

    # Create graph extractor
    extractor = CandidateGraphExtractor(
        junction_threshold=0.3,
        junction_min_distance=5,
    )

    # Create dummy image
    print("\n  Running forward pass with dummy image...")
    dummy_image = torch.randn(1, 3, 512, 512).to(device)

    with torch.no_grad():
        pixel_outputs = model(dummy_image)

    print(f"    Segmentation: {pixel_outputs['segmentation'].shape}")
    print(f"    Junction: {pixel_outputs['junction'].shape}")
    print(f"    Features: {pixel_outputs['features'].shape}")

    # Extract candidate graph
    seg_probs = torch.softmax(pixel_outputs['segmentation'], dim=1)
    junction_heatmap = pixel_outputs['junction']

    candidate_graph = extractor.extract(
        seg_probs[0].cpu().numpy(),
        junction_heatmap[0, 0].cpu().numpy(),
    )

    if candidate_graph is None or len(candidate_graph.vertices) == 0:
        print("\n  ⚠ No vertices extracted from dummy image")
        print("  This is expected for random noise - try with real image")
        return None

    print(f"\n  Extracted graph:")
    print(f"    Vertices: {len(candidate_graph.vertices)}")
    print(f"    Edges: {len(candidate_graph.edges)}")

    # Run through graph head
    vertices = torch.from_numpy(candidate_graph.vertices).float().to(device)
    edge_index = torch.from_numpy(candidate_graph.edges.T).long().to(device)

    outputs = graph_head(
        vertices=vertices,
        edge_index=edge_index,
        backbone_features=pixel_outputs['features'],
        seg_probs=seg_probs,
        image_size=512,
    )

    print(f"\n  Graph head outputs:")
    print(f"    edge_existence: {outputs['edge_existence'].shape}")
    print(f"    edge_assignment: {outputs['edge_assignment'].shape}")
    print(f"    vertex_offset: {outputs['vertex_offset'].shape}")

    # Check no NaN
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            assert not torch.isnan(val).any(), f"NaN in {key}!"

    print("\n  ✓ No NaN values in outputs")
    print("\n✅ TEST 7 PASSED: Integration with pixel head works")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate Graph Head implementation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pixel head checkpoint for integration test")
    parser.add_argument("--fold", type=str, default=None,
                        help="Path to FOLD file for real data test")
    parser.add_argument("--skip-overfit", action="store_true",
                        help="Skip overfit test (faster)")
    args = parser.parse_args()

    print("="*60)
    print("GRAPH HEAD VALIDATION SUITE")
    print("="*60)

    results = {}

    # Run tests
    results['gnn_layers'] = test_gnn_layers()
    results['feature_extractors'] = test_feature_extractors()
    results['graph_head_forward'] = test_graph_head_forward()
    results['loss_computation'] = test_loss_computation()
    results['label_generation'] = test_label_generation()

    if not args.skip_overfit:
        results['overfit_single'] = test_overfit_single_sample()

    if args.checkpoint:
        results['real_integration'] = test_with_real_data(args.checkpoint, args.fold)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, result in results.items():
        if result is None:
            status = "⏭ SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
            all_passed = False
        print(f"  {name}: {status}")

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready for training!")
    else:
        print("⚠ SOME TESTS FAILED - Investigate before training")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
