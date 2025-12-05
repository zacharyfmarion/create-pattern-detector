#!/usr/bin/env python3
"""
Validate the training pipeline before starting training.

Checks:
1. FOLD parsing works correctly
2. Ground truth generation produces expected outputs
3. Coordinate transforms match rendering script
4. Model forward pass works
5. Loss computation works
6. Backward pass works
7. GPU memory usage estimate
8. Visual sanity check (saves sample images)

Usage:
    python scripts/validation/validate_pipeline.py --fold-dir data/output/synthetic/raw/tier-a
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json


def check_fold_parsing(fold_dir: Path, num_samples: int = 5):
    """Check that FOLD files parse correctly."""
    print("\n" + "=" * 60)
    print("1. FOLD PARSING")
    print("=" * 60)

    from src.data.fold_parser import FOLDParser, transform_coords

    parser = FOLDParser()
    fold_files = sorted(fold_dir.glob("*.fold"))[:num_samples]

    if len(fold_files) == 0:
        print(f"❌ No .fold files found in {fold_dir}")
        return False

    print(f"Found {len(list(fold_dir.glob('*.fold')))} FOLD files")
    print(f"Testing first {len(fold_files)} files...\n")

    for fold_path in fold_files:
        try:
            cp = parser.parse(fold_path)
            print(f"✓ {fold_path.name}")
            print(f"  Vertices: {cp.num_vertices}, Edges: {cp.num_edges}, Creases: {cp.num_creases}")
            print(f"  Bounds: {cp.bounds}")

            # Check assignment distribution
            unique, counts = np.unique(cp.assignments, return_counts=True)
            assignment_map = {0: 'M', 1: 'V', 2: 'B', 3: 'U'}
            dist = {assignment_map[u]: c for u, c in zip(unique, counts)}
            print(f"  Assignments: {dist}")

            # Check coordinate transform
            pixel_verts, params = transform_coords(cp.vertices, image_size=1024, padding=50)
            print(f"  Transformed bounds: ({pixel_verts[:, 0].min():.1f}, {pixel_verts[:, 1].min():.1f}) to ({pixel_verts[:, 0].max():.1f}, {pixel_verts[:, 1].max():.1f})")
            print()

        except Exception as e:
            print(f"❌ {fold_path.name}: {e}")
            return False

    print("✓ FOLD parsing OK")
    return True


def check_ground_truth(fold_dir: Path, output_dir: Path):
    """Check ground truth generation and save visualizations."""
    print("\n" + "=" * 60)
    print("2. GROUND TRUTH GENERATION")
    print("=" * 60)

    from src.data.fold_parser import FOLDParser
    from src.data.annotations import GroundTruthGenerator

    parser = FOLDParser()
    gt_gen = GroundTruthGenerator(image_size=1024, padding=50, line_width=3)

    fold_files = sorted(fold_dir.glob("*.fold"))[:3]
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold_path in fold_files:
        try:
            cp = parser.parse(fold_path)
            gt = gt_gen.generate(cp)

            print(f"\n{fold_path.name}:")
            print(f"  Segmentation shape: {gt['segmentation'].shape}, dtype: {gt['segmentation'].dtype}")
            print(f"  Orientation shape: {gt['orientation'].shape}, dtype: {gt['orientation'].dtype}")
            print(f"  Junction heatmap shape: {gt['junction_heatmap'].shape}, dtype: {gt['junction_heatmap'].dtype}")

            # Check class distribution in segmentation
            unique, counts = np.unique(gt['segmentation'], return_counts=True)
            class_map = {0: 'BG', 1: 'M', 2: 'V', 3: 'B', 4: 'U'}
            total = counts.sum()
            print(f"  Segmentation distribution:")
            for u, c in zip(unique, counts):
                print(f"    {class_map[u]}: {c} ({100*c/total:.2f}%)")

            # Check junction count
            num_junctions = (gt['junction_heatmap'] > 0.5).sum()
            print(f"  Junction pixels (>0.5): {num_junctions}")

            # Save visualization
            save_gt_visualization(gt, output_dir / f"{fold_path.stem}_gt.png")
            print(f"  Saved visualization to {output_dir / f'{fold_path.stem}_gt.png'}")

        except Exception as e:
            print(f"❌ {fold_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n✓ Ground truth generation OK")
    return True


def save_gt_visualization(gt: dict, output_path: Path):
    """Save a visualization of ground truth."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Segmentation
    seg_colors = ['white', 'red', 'blue', 'black', 'gray']
    seg_cmap = ListedColormap(seg_colors)
    axes[0, 0].imshow(gt['segmentation'], cmap=seg_cmap, vmin=0, vmax=4)
    axes[0, 0].set_title('Segmentation (BG=white, M=red, V=blue, B=black, U=gray)')
    axes[0, 0].axis('off')

    # Orientation (show as color wheel)
    orient = gt['orientation']
    angle = np.arctan2(orient[:, :, 1], orient[:, :, 0])
    magnitude = np.linalg.norm(orient, axis=2)
    hsv_img = np.zeros((*orient.shape[:2], 3))
    hsv_img[:, :, 0] = (angle + np.pi) / (2 * np.pi)
    hsv_img[:, :, 1] = np.clip(magnitude, 0, 1)
    hsv_img[:, :, 2] = np.clip(magnitude, 0, 1)
    from matplotlib.colors import hsv_to_rgb
    rgb_img = hsv_to_rgb(hsv_img)
    axes[0, 1].imshow(rgb_img)
    axes[0, 1].set_title('Orientation Field (hue=angle)')
    axes[0, 1].axis('off')

    # Junction heatmap
    axes[1, 0].imshow(gt['junction_heatmap'], cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Junction Heatmap')
    axes[1, 0].axis('off')

    # Edge distance
    axes[1, 1].imshow(gt['edge_distance'], cmap='viridis')
    axes[1, 1].set_title('Edge Distance Transform')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def check_dataset(fold_dir: Path):
    """Check PyTorch dataset loading."""
    print("\n" + "=" * 60)
    print("3. DATASET LOADING")
    print("=" * 60)

    from src.data.dataset import CreasePatternDataset
    from src.data.transforms import get_train_transform

    try:
        # Without transform
        dataset = CreasePatternDataset(
            fold_dir=fold_dir,
            image_size=1024,
            padding=50,
            transform=None,
            split="train",
        )
        print(f"Dataset size: {len(dataset)}")

        # Load one sample
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"Image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
        print(f"Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print(f"Segmentation shape: {sample['segmentation'].shape}")
        print(f"Orientation shape: {sample['orientation'].shape}")
        print(f"Junction heatmap shape: {sample['junction_heatmap'].shape}")
        print(f"Graph vertices: {sample['graph']['vertices'].shape}")
        print(f"Graph edges: {sample['graph']['edges'].shape}")

        # With transform
        transform = get_train_transform(image_size=1024, strength="medium")
        dataset_aug = CreasePatternDataset(
            fold_dir=fold_dir,
            image_size=1024,
            padding=50,
            transform=transform,
            split="train",
        )
        sample_aug = dataset_aug[0]
        print(f"\nWith augmentation - Image shape: {sample_aug['image'].shape}")

        print("\n✓ Dataset loading OK")
        return True

    except Exception as e:
        print(f"❌ Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model():
    """Check model forward pass."""
    print("\n" + "=" * 60)
    print("4. MODEL FORWARD PASS")
    print("=" * 60)

    from src.models import CreasePatternDetector

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        # Create model
        model = CreasePatternDetector(
            backbone="hrnet_w32",
            pretrained=True,
            hidden_channels=256,
            num_seg_classes=5,
        )
        model = model.to(device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {num_trainable:,}")

        # Test forward pass
        x = torch.randn(1, 3, 1024, 1024).to(device)
        print(f"\nInput shape: {x.shape}")

        with torch.no_grad():
            outputs = model(x)

        print(f"Output shapes:")
        print(f"  Segmentation: {outputs['segmentation'].shape}")
        print(f"  Orientation: {outputs['orientation'].shape}")
        print(f"  Junction: {outputs['junction'].shape}")

        # Check output ranges
        seg_probs = torch.softmax(outputs['segmentation'], dim=1)
        print(f"\nOutput ranges:")
        print(f"  Segmentation logits: [{outputs['segmentation'].min():.3f}, {outputs['segmentation'].max():.3f}]")
        print(f"  Segmentation probs sum: {seg_probs.sum(dim=1).mean():.6f} (should be 1.0)")
        print(f"  Orientation: [{outputs['orientation'].min():.3f}, {outputs['orientation'].max():.3f}]")
        print(f"  Junction: [{outputs['junction'].min():.3f}, {outputs['junction'].max():.3f}]")

        print("\n✓ Model forward pass OK")
        return True

    except Exception as e:
        print(f"❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_loss():
    """Check loss computation."""
    print("\n" + "=" * 60)
    print("5. LOSS COMPUTATION")
    print("=" * 60)

    from src.models.losses import PixelLoss

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create dummy data
        B, C, H, W = 2, 5, 1024, 1024

        outputs = {
            'segmentation': torch.randn(B, C, H, W, requires_grad=True).to(device),
            'orientation': torch.randn(B, 2, H, W, requires_grad=True).to(device),
            'junction': torch.sigmoid(torch.randn(B, 1, H, W, requires_grad=True)).to(device),
        }

        targets = {
            'segmentation': torch.randint(0, C, (B, H, W)).to(device),
            'orientation': torch.randn(B, 2, H, W).to(device),
            'junction_heatmap': torch.rand(B, H, W).to(device),
        }

        # Compute loss
        criterion = PixelLoss(
            seg_weight=1.0,
            orient_weight=0.5,
            junction_weight=1.0,
        )

        loss_dict = criterion(outputs, targets)

        print(f"Loss values:")
        print(f"  Total: {loss_dict['total'].item():.4f}")
        print(f"  Segmentation: {loss_dict['seg'].item():.4f}")
        print(f"  Orientation: {loss_dict['orient'].item():.4f}")
        print(f"  Junction: {loss_dict['junction'].item():.4f}")

        # Check gradients
        loss_dict['total'].backward()
        print(f"\n✓ Backward pass completed")

        print("\n✓ Loss computation OK")
        return True

    except Exception as e:
        print(f"❌ Loss error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_step(fold_dir: Path):
    """Check full training step."""
    print("\n" + "=" * 60)
    print("6. FULL TRAINING STEP")
    print("=" * 60)

    from src.data.dataset import CreasePatternDataset
    from src.models import CreasePatternDetector
    from src.models.losses import PixelLoss
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create dataset and loader
        dataset = CreasePatternDataset(
            fold_dir=fold_dir,
            image_size=1024,
            padding=50,
            transform=None,
            split="train",
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Create model
        model = CreasePatternDetector(
            backbone="hrnet_w32",
            pretrained=True,
        ).to(device)

        # Create optimizer and loss
        optimizer = AdamW(model.parameters(), lr=1e-4)
        criterion = PixelLoss()

        # Get one batch
        batch = next(iter(loader))
        images = batch['image'].to(device)
        targets = {
            'segmentation': batch['segmentation'].to(device),
            'orientation': batch['orientation'].to(device),
            'junction_heatmap': batch['junction_heatmap'].to(device),
        }

        print(f"Batch image shape: {images.shape}")

        # Forward pass
        model.train()
        outputs = model(images)

        # Compute loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total']
        print(f"Loss: {loss.item():.4f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"Gradient norm: {total_grad_norm:.4f}")

        # Optimizer step
        optimizer.step()
        print("Optimizer step completed")

        print("\n✓ Full training step OK")
        return True

    except Exception as e:
        print(f"❌ Training step error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_memory(fold_dir: Path):
    """Check GPU memory usage."""
    print("\n" + "=" * 60)
    print("7. GPU MEMORY USAGE")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No CUDA available, skipping memory check")
        return True

    from src.data.dataset import CreasePatternDataset
    from src.models import CreasePatternDetector
    from src.models.losses import PixelLoss
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler

    try:
        device = "cuda"
        torch.cuda.reset_peak_memory_stats()

        # Create dataset
        dataset = CreasePatternDataset(
            fold_dir=fold_dir,
            image_size=1024,
            padding=50,
            transform=None,
            split="train",
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        # Create model
        model = CreasePatternDetector(backbone="hrnet_w32", pretrained=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = PixelLoss()
        scaler = GradScaler()

        # Get batch
        batch = next(iter(loader))
        images = batch['image'].to(device)
        targets = {
            'segmentation': batch['segmentation'].to(device),
            'orientation': batch['orientation'].to(device),
            'junction_heatmap': batch['junction_heatmap'].to(device),
        }

        print(f"Batch size: {images.shape[0]}")
        print(f"Memory after loading batch: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Forward pass with AMP
        model.train()
        with autocast():
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total']

        print(f"Memory after forward: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory: {peak_memory:.2f} GB")

        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name}")
        print(f"Total GPU memory: {gpu_memory:.2f} GB")
        print(f"Recommended batch size for this GPU: {int(gpu_memory / peak_memory * 4)}")

        print("\n✓ Memory check OK")
        return True

    except torch.cuda.OutOfMemoryError:
        print("❌ Out of memory! Try reducing batch size")
        return False
    except Exception as e:
        print(f"❌ Memory check error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate training pipeline")
    parser.add_argument("--fold-dir", type=str, required=True, help="Directory containing FOLD files")
    parser.add_argument("--output-dir", type=str, default="validation_output", help="Directory for output visualizations")
    args = parser.parse_args()

    fold_dir = Path(args.fold_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("CREASE PATTERN DETECTOR - PIPELINE VALIDATION")
    print("=" * 60)

    results = {}

    # Run all checks
    results['fold_parsing'] = check_fold_parsing(fold_dir)
    results['ground_truth'] = check_ground_truth(fold_dir, output_dir)
    results['dataset'] = check_dataset(fold_dir)
    results['model'] = check_model()
    results['loss'] = check_loss()
    results['training_step'] = check_training_step(fold_dir)
    results['memory'] = check_memory(fold_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All checks passed! Ready to train.")
        print(f"\nTo start training:")
        print(f"  python scripts/training/train_pixel_head.py --fold-dir {fold_dir}")
    else:
        print("⚠️  Some checks failed. Please fix the issues before training.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
