#!/usr/bin/env python3
"""
Train the graph head on ground truth graphs.

This script trains the graph neural network to predict edge existence
and M/V/B/U assignments using ground truth vertex positions.

Usage:
    python scripts/train_graph_head.py --fold-dir data/training/full-training/fold
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from src.data.graph_dataset import GraphDataset
from src.models.graph import GraphHead
from src.models.losses import GraphLoss, compute_graph_metrics
from src.models.backbone import HRNetBackbone


class GraphTrainer:
    """Trainer for graph head."""

    def __init__(
        self,
        model: nn.Module,
        backbone: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: nn.Module,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/graph_head",
    ):
        self.model = model
        self.backbone = backbone
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_f1 = 0.0
        self.epoch = 0

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        self.backbone.eval()  # Keep backbone frozen

        total_loss = 0.0
        total_existence_loss = 0.0
        total_assignment_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            images = batch['images'].to(self.device)
            vertices_list = batch['vertices_list']
            edge_index_list = batch['edge_index_list']
            edge_existence_list = batch['edge_existence_list']
            edge_assignment_list = batch['edge_assignment_list']

            # Extract backbone features (frozen)
            with torch.no_grad():
                features = self.backbone(images)

            # Process each graph in the batch
            batch_loss = 0.0
            batch_existence_loss = 0.0
            batch_assignment_loss = 0.0

            for i in range(len(vertices_list)):
                vertices = vertices_list[i].to(self.device)
                edge_index = edge_index_list[i].to(self.device)
                edge_existence = edge_existence_list[i].to(self.device)
                edge_assignment = edge_assignment_list[i].to(self.device)

                if len(vertices) < 2 or edge_index.size(1) == 0:
                    continue

                # Forward pass
                feat = features[i:i+1]  # (1, C, H, W)
                verts = vertices.unsqueeze(0)  # (1, N, 2)

                outputs = self.model(feat, verts, edge_index)

                # Compute loss
                targets = {
                    'edge_existence': edge_existence,
                    'edge_assignment': edge_assignment,
                }
                losses = self.loss_fn(outputs, targets)

                batch_loss += losses['loss']
                batch_existence_loss += losses['existence_loss'].item()
                batch_assignment_loss += losses['assignment_loss'].item()

            if batch_loss > 0:
                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += batch_loss.item()
                total_existence_loss += batch_existence_loss
                total_assignment_loss += batch_assignment_loss
                num_batches += 1

            pbar.set_postfix({
                'loss': f"{total_loss / max(num_batches, 1):.4f}",
                'exist': f"{total_existence_loss / max(num_batches, 1):.4f}",
                'assign': f"{total_assignment_loss / max(num_batches, 1):.4f}",
            })

        return {
            'loss': total_loss / max(num_batches, 1),
            'existence_loss': total_existence_loss / max(num_batches, 1),
            'assignment_loss': total_assignment_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.model.eval()
        self.backbone.eval()

        all_metrics = {
            'existence_precision': [],
            'existence_recall': [],
            'existence_f1': [],
            'existence_accuracy': [],
            'assignment_accuracy': [],
        }

        total_loss = 0.0
        num_samples = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['images'].to(self.device)
            vertices_list = batch['vertices_list']
            edge_index_list = batch['edge_index_list']
            edge_existence_list = batch['edge_existence_list']
            edge_assignment_list = batch['edge_assignment_list']

            # Extract backbone features
            features = self.backbone(images)

            for i in range(len(vertices_list)):
                vertices = vertices_list[i].to(self.device)
                edge_index = edge_index_list[i].to(self.device)
                edge_existence = edge_existence_list[i].to(self.device)
                edge_assignment = edge_assignment_list[i].to(self.device)

                if len(vertices) < 2 or edge_index.size(1) == 0:
                    continue

                # Forward pass
                feat = features[i:i+1]
                verts = vertices.unsqueeze(0)

                outputs = self.model(feat, verts, edge_index)

                # Compute loss
                targets = {
                    'edge_existence': edge_existence,
                    'edge_assignment': edge_assignment,
                }
                losses = self.loss_fn(outputs, targets)
                total_loss += losses['loss'].item()

                # Compute metrics
                metrics = compute_graph_metrics(outputs, targets)
                for key, value in metrics.items():
                    all_metrics[key].append(value)

                num_samples += 1

        # Average metrics
        avg_metrics = {
            key: np.mean(values) if values else 0.0
            for key, values in all_metrics.items()
        }
        avg_metrics['loss'] = total_loss / max(num_samples, 1)

        return avg_metrics

    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(num_epochs):
            self.epoch = epoch + 1

            # Train
            train_metrics = self.train_epoch()
            print(f"\nEpoch {self.epoch} Train: loss={train_metrics['loss']:.4f}")

            # Validate
            val_metrics = self.validate()
            print(f"Epoch {self.epoch} Val: loss={val_metrics['loss']:.4f}, "
                  f"F1={val_metrics['existence_f1']:.4f}, "
                  f"assign_acc={val_metrics['assignment_accuracy']:.4f}")

            # Update scheduler
            self.scheduler.step()

            # Save checkpoint
            if val_metrics['existence_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['existence_f1']
                self.save_checkpoint('best_graph_head.pt', val_metrics)
                print(f"  New best model! F1={self.best_val_f1:.4f}")

            # Save periodic checkpoint
            if self.epoch % 10 == 0:
                self.save_checkpoint(f'graph_head_epoch_{self.epoch}.pt', val_metrics)

        # Save final checkpoint
        self.save_checkpoint('final_graph_head.pt', val_metrics)
        print(f"\nTraining complete! Best F1: {self.best_val_f1:.4f}")

    def save_checkpoint(self, filename: str, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'metrics': metrics,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)


def main():
    parser = argparse.ArgumentParser(description="Train graph head")
    parser.add_argument(
        "--fold-dir",
        type=str,
        required=True,
        help="Directory containing .fold files",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
        help="Path to pixel head checkpoint (for backbone weights)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="GNN hidden dimension",
    )
    parser.add_argument(
        "--num-gnn-layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Ratio of negative to positive edges",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/graph_head",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # Create datasets
    print(f"\nLoading dataset from: {args.fold_dir}")
    train_dataset = GraphDataset(
        fold_dir=args.fold_dir,
        image_size=args.image_size,
        negative_ratio=args.negative_ratio,
        augment_vertices=True,
        split="train",
    )
    val_dataset = GraphDataset(
        fold_dir=args.fold_dir,
        image_size=args.image_size,
        negative_ratio=args.negative_ratio,
        augment_vertices=False,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=GraphDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=GraphDataset.collate_fn,
    )

    # Create backbone (for feature extraction)
    print("\nCreating backbone...")
    backbone = HRNetBackbone(
        variant="hrnet_w32",
        pretrained=True,
        output_stride=4,
    ).to(args.device)

    # Load backbone weights from pixel head checkpoint if provided
    if args.backbone_checkpoint:
        print(f"Loading backbone from: {args.backbone_checkpoint}")
        checkpoint = torch.load(args.backbone_checkpoint, map_location=args.device, weights_only=False)
        # Extract backbone weights from full model
        backbone_state = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('backbone.'):
                backbone_state[key.replace('backbone.', '')] = value
        backbone.load_state_dict(backbone_state)

    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    # Create graph head
    print("\nCreating graph head...")
    model = GraphHead(
        in_channels=backbone.out_channels,
        vertex_dim=128,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_heads=4,
        dropout=0.1,
        num_classes=4,
    ).to(args.device)

    print(f"Graph head parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Create loss function
    loss_fn = GraphLoss(
        existence_weight=1.0,
        assignment_weight=1.0,
    )

    # Create trainer
    trainer = GraphTrainer(
        model=model,
        backbone=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
