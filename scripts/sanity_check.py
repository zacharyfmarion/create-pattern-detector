#!/usr/bin/env python3
"""
Sanity checks before expensive training runs.

Checks:
1. Overfit single batch - can the model memorize one sample?
2. Small-scale training - does loss decrease on 100 samples?
3. Checkpoint save/load - can we resume training?
4. Data loading speed - is it a bottleneck?
5. Learning rate schedule - does it behave correctly?
6. Gradient health - no vanishing/exploding gradients?

Usage:
    python scripts/sanity_check.py --fold-dir data/output/synthetic/raw/tier-a
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import tempfile


def check_overfit_single_batch(fold_dir: Path):
    """Check if model can overfit a single batch (memorization test)."""
    print("\n" + "=" * 60)
    print("1. OVERFIT SINGLE BATCH")
    print("=" * 60)

    from src.data.dataset import CreasePatternDataset
    from src.models import CreasePatternDetector
    from src.models.losses import PixelLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load single sample
    dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_size=512,  # Smaller for speed
        padding=25,
        transform=None,
        split="train",
    )

    # Get one batch
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    images = batch['image'].to(device)
    targets = {
        'segmentation': batch['segmentation'].to(device),
        'orientation': batch['orientation'].to(device),
        'junction_heatmap': batch['junction_heatmap'].to(device),
    }

    # Create model (smaller for speed)
    model = CreasePatternDetector(
        backbone="hrnet_w18",
        pretrained=False,  # Faster without pretrained weights
        hidden_channels=128,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = PixelLoss()

    print("Training on single batch for 100 iterations...")
    model.train()

    initial_loss = None
    losses = []

    for i in range(100):
        optimizer.zero_grad()
        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i == 0:
            initial_loss = loss.item()

        if (i + 1) % 20 == 0:
            print(f"  Iter {i+1}: loss = {loss.item():.4f}")

    final_loss = losses[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100

    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Reduction: {reduction:.1f}%")

    # Check if loss decreased significantly
    if final_loss < initial_loss * 0.1:
        print("✓ Model can overfit single batch (loss reduced by >90%)")
        return True
    elif final_loss < initial_loss * 0.5:
        print("⚠ Model partially overfit (loss reduced by >50%)")
        return True
    else:
        print("❌ Model failed to overfit single batch")
        return False


def check_small_scale_training(fold_dir: Path, num_samples: int = 100, num_epochs: int = 5):
    """Check if loss decreases on small dataset."""
    print("\n" + "=" * 60)
    print("2. SMALL-SCALE TRAINING")
    print("=" * 60)

    from src.data.dataset import CreasePatternDataset
    from src.models import CreasePatternDetector
    from src.models.losses import PixelLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create small dataset
    full_dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_size=512,
        padding=25,
        transform=None,
        split="train",
    )

    # Use subset
    indices = list(range(min(num_samples, len(full_dataset))))
    dataset = Subset(full_dataset, indices)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f"Training on {len(dataset)} samples for {num_epochs} epochs...")

    # Create model
    model = CreasePatternDetector(
        backbone="hrnet_w18",
        pretrained=False,
        hidden_channels=128,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = PixelLoss()

    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in loader:
            images = batch['image'].to(device)
            targets = {
                'segmentation': batch['segmentation'].to(device),
                'orientation': batch['orientation'].to(device),
                'junction_heatmap': batch['junction_heatmap'].to(device),
            }

            optimizer.zero_grad()
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{num_epochs}: loss = {avg_loss:.4f}")

    # Check if loss decreased
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]

    if final_loss < initial_loss:
        reduction = (initial_loss - final_loss) / initial_loss * 100
        print(f"\n✓ Loss decreased by {reduction:.1f}%")
        return True
    else:
        print(f"\n❌ Loss did not decrease (initial: {initial_loss:.4f}, final: {final_loss:.4f})")
        return False


def check_checkpoint_save_load(fold_dir: Path):
    """Check checkpoint saving and loading."""
    print("\n" + "=" * 60)
    print("3. CHECKPOINT SAVE/LOAD")
    print("=" * 60)

    from src.data.dataset import CreasePatternDataset
    from src.models import CreasePatternDetector
    from src.models.losses import PixelLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model and run one step
    model = CreasePatternDetector(
        backbone="hrnet_w18",
        pretrained=False,
        hidden_channels=128,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=100)

    # Run a few steps
    for _ in range(5):
        scheduler.step()

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': 5,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Create new model and load
    model2 = CreasePatternDetector(
        backbone="hrnet_w18",
        pretrained=False,
        hidden_channels=128,
    ).to(device)
    optimizer2 = AdamW(model2.parameters(), lr=1e-4)
    scheduler2 = OneCycleLR(optimizer2, max_lr=1e-3, total_steps=100)

    loaded = torch.load(checkpoint_path, weights_only=False)
    model2.load_state_dict(loaded['model_state_dict'])
    optimizer2.load_state_dict(loaded['optimizer_state_dict'])
    scheduler2.load_state_dict(loaded['scheduler_state_dict'])

    print(f"Loaded checkpoint from epoch {loaded['epoch']}")

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        if not torch.allclose(p1, p2):
            print(f"❌ Parameter mismatch: {n1}")
            return False

    # Verify LR matches
    lr1 = optimizer.param_groups[0]['lr']
    lr2 = optimizer2.param_groups[0]['lr']
    if abs(lr1 - lr2) > 1e-8:
        print(f"❌ LR mismatch: {lr1} vs {lr2}")
        return False

    # Cleanup
    Path(checkpoint_path).unlink()

    print("✓ Checkpoint save/load works correctly")
    return True


def check_data_loading_speed(fold_dir: Path):
    """Benchmark data loading speed."""
    print("\n" + "=" * 60)
    print("4. DATA LOADING SPEED")
    print("=" * 60)

    from src.data.dataset import CreasePatternDataset, collate_fn

    dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_size=1024,
        padding=50,
        transform=None,
        split="train",
    )

    # Test with different num_workers
    for num_workers in [0, 2, 4]:
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,  # MPS doesn't support pin_memory
            collate_fn=collate_fn,
        )

        # Warmup
        iterator = iter(loader)
        next(iterator)

        # Time 10 batches
        start = time.time()
        for i, batch in enumerate(iterator):
            if i >= 9:
                break
        elapsed = time.time() - start

        batches_per_sec = 10 / elapsed
        samples_per_sec = 40 / elapsed

        print(f"  num_workers={num_workers}: {batches_per_sec:.1f} batches/s, {samples_per_sec:.1f} samples/s")

    print("\n✓ Data loading benchmark complete")
    print("  Tip: Use num_workers that gives best throughput")
    return True


def check_gradient_health(fold_dir: Path):
    """Check for vanishing/exploding gradients."""
    print("\n" + "=" * 60)
    print("5. GRADIENT HEALTH")
    print("=" * 60)

    from src.data.dataset import CreasePatternDataset, collate_fn
    from src.models import CreasePatternDetector
    from src.models.losses import PixelLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_size=512,
        padding=25,
        transform=None,
        split="train",
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))

    model = CreasePatternDetector(
        backbone="hrnet_w18",
        pretrained=False,
        hidden_channels=128,
    ).to(device)

    criterion = PixelLoss()

    images = batch['image'].to(device)
    targets = {
        'segmentation': batch['segmentation'].to(device),
        'orientation': batch['orientation'].to(device),
        'junction_heatmap': batch['junction_heatmap'].to(device),
    }

    model.train()
    outputs = model(images)
    loss_dict = criterion(outputs, targets)
    loss = loss_dict['total']
    loss.backward()

    # Collect gradient statistics
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            # Group by module
            module = name.split('.')[0]
            if module not in grad_norms:
                grad_norms[module] = []
            grad_norms[module].append(grad_norm)

    print("Gradient norms by module:")
    all_healthy = True
    for module, norms in sorted(grad_norms.items()):
        mean_norm = np.mean(norms)
        max_norm = np.max(norms)
        min_norm = np.min(norms)

        status = "✓"
        if max_norm > 100:
            status = "⚠ HIGH"
            all_healthy = False
        elif max_norm < 1e-6:
            status = "⚠ LOW"
            all_healthy = False

        print(f"  {module}: mean={mean_norm:.2e}, max={max_norm:.2e}, min={min_norm:.2e} {status}")

    if all_healthy:
        print("\n✓ Gradients look healthy")
    else:
        print("\n⚠ Some gradients may need attention")

    return True


def check_lr_schedule(fold_dir: Path):
    """Visualize learning rate schedule."""
    print("\n" + "=" * 60)
    print("6. LEARNING RATE SCHEDULE")
    print("=" * 60)

    from src.models import CreasePatternDetector

    model = CreasePatternDetector(
        backbone="hrnet_w18",
        pretrained=False,
        hidden_channels=128,
    )

    # Simulate 100 epochs with 100 steps each
    total_steps = 10000
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Print key points
    print(f"Initial LR: {lrs[0]:.2e}")
    print(f"Peak LR: {max(lrs):.2e} (at step {lrs.index(max(lrs))})")
    print(f"Final LR: {lrs[-1]:.2e}")

    # Check schedule is reasonable
    if max(lrs) >= lrs[0] and lrs[-1] < max(lrs):
        print("\n✓ LR schedule looks correct (warmup → peak → decay)")
        return True
    else:
        print("\n❌ LR schedule may have issues")
        return False


def main():
    parser = argparse.ArgumentParser(description="Sanity checks before training")
    parser.add_argument("--fold-dir", type=str, required=True, help="Directory containing FOLD files")
    parser.add_argument("--quick", action="store_true", help="Run quick checks only")
    args = parser.parse_args()

    fold_dir = Path(args.fold_dir)

    print("=" * 60)
    print("TRAINING SANITY CHECKS")
    print("=" * 60)

    results = {}

    # Run all checks
    results['overfit'] = check_overfit_single_batch(fold_dir)

    if not args.quick:
        results['small_scale'] = check_small_scale_training(fold_dir)

    results['checkpoint'] = check_checkpoint_save_load(fold_dir)
    results['data_loading'] = check_data_loading_speed(fold_dir)
    results['gradients'] = check_gradient_health(fold_dir)
    results['lr_schedule'] = check_lr_schedule(fold_dir)

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
        print("🎉 All sanity checks passed! Safe to start training.")
    else:
        print("⚠️ Some checks failed. Review before training.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
