#!/usr/bin/env python3
"""
Training script for crease pattern detection.

Usage:
    python scripts/training/train_pixel_head.py --fold-dir data/output/synthetic/raw/tier-a --epochs 50
    python scripts/training/train_pixel_head.py --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import yaml

from src.data import CreasePatternDataset
from src.data.dataset import create_dataloaders
from src.models import CreasePatternDetector
from src.training import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train crease pattern detector")

    # Data
    parser.add_argument(
        "--fold-dir",
        type=str,
        required=True,
        help="Directory containing .fold files",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing pre-rendered images (optional)",
    )

    # Image size
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Image size (default: 1024). Use 512 for faster experiments.",
    )

    # Model
    parser.add_argument(
        "--backbone",
        type=str,
        default="hrnet_w32",
        help="Backbone network (default: hrnet_w32)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained backbone",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers",
    )

    # Loss weights
    parser.add_argument("--seg-weight", type=float, default=1.0)
    parser.add_argument("--orient-weight", type=float, default=0.5)
    parser.add_argument("--junction-weight", type=float, default=1.0)

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cp-detector",
        help="W&B project name",
    )

    # Config file (overrides command line args)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    return parser.parse_args()


def load_config(args) -> dict:
    """Build config from args and optional YAML file."""
    # Scale padding proportionally with image size
    padding = int(50 * args.image_size / 1024)
    line_width = max(1, int(2 * args.image_size / 1024))

    config = {
        # Data
        "fold_dir": args.fold_dir,
        "image_dir": args.image_dir,
        "image_size": args.image_size,
        "padding": padding,
        "line_width": line_width,
        # Model
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "hidden_channels": 256,
        "num_seg_classes": 5,
        # Training
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "backbone_lr_mult": 0.1,
        "scheduler": "onecycle",
        "max_grad_norm": 1.0,
        "amp": True,
        "num_workers": args.num_workers,
        # Loss
        "seg_weight": args.seg_weight,
        "orient_weight": args.orient_weight,
        "junction_weight": args.junction_weight,
        # Checkpointing
        "checkpoint_dir": args.checkpoint_dir,
        "save_every": args.save_every,
        # Logging
        "use_wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "log_every": 100,
    }

    # Override with YAML config if provided
    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        # Flatten nested config
        for key, value in yaml_config.items():
            if isinstance(value, dict):
                config.update(value)
            else:
                config[key] = value

    return config


def main():
    args = parse_args()
    config = load_config(args)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataloaders
    print(f"Loading data from {config['fold_dir']}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        fold_dir=config["fold_dir"],
        image_dir=config.get("image_dir"),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_size=config["image_size"],
        augment=True,
        padding=config["padding"],
        line_width=config["line_width"],
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    print(f"Creating model with backbone: {config['backbone']}...")
    model = CreasePatternDetector(
        backbone=config["backbone"],
        pretrained=config["pretrained"],
        hidden_channels=config["hidden_channels"],
        num_seg_classes=config["num_seg_classes"],
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\nStarting training for {config['epochs']} epochs...")
    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"Best validation IoU: {results['best_val_metric']:.4f} (epoch {results['best_epoch'] + 1})")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}")
    print(f"\nTo resume training from latest checkpoint:")
    print(f"  python scripts/training/train_pixel_head.py --fold-dir {config['fold_dir']} --resume {config['checkpoint_dir']}/latest.pt --epochs <total_epochs>")


if __name__ == "__main__":
    main()
