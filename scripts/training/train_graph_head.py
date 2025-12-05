#!/usr/bin/env python3
"""
Train the Graph Head (Phase 2).

This script trains the graph neural network to predict edge existence,
M/V/B/U assignments, and vertex refinement using candidate graphs
extracted from frozen pixel head outputs.

Usage:
    python scripts/training/train_graph_head.py \
        --pixel-checkpoint checkpoints/checkpoint_epoch_8.pt \
        --data-dir data/training/full-training \
        --epochs 30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.models import CreasePatternDetector
from src.models.graph import GraphHead
from src.models.losses import GraphLoss, compute_graph_metrics
from src.data.graph_labels import generate_graph_labels
from src.data.dataset import CreasePatternDataset
from src.postprocessing.graph_extraction import GraphExtractor, GraphExtractorConfig


def collate_fn(batch):
    """Custom collate that handles variable-size graph data."""
    images = torch.stack([item['image'] for item in batch])

    # Keep graph data as list (variable size graphs)
    graphs = [item.get('graph') for item in batch]

    return {
        'image': images,
        'graph': graphs,
    }


class GraphHeadTrainer:
    """
    Training loop for Graph Head (Phase 2).

    Features:
    - Frozen backbone + pixel head
    - Graph extraction from pixel head outputs
    - Edge existence + assignment + vertex refinement losses
    """

    def __init__(
        self,
        pixel_model: CreasePatternDetector,
        graph_head: GraphHead,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        self.pixel_model = pixel_model.to(device)
        self.graph_head = graph_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Freeze pixel model
        self.pixel_model.eval()
        for param in self.pixel_model.parameters():
            param.requires_grad = False

        # Training params
        self.epochs = config.get("epochs", 30)
        self.learning_rate = config.get("learning_rate", 5e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_amp = config.get("amp", True) and device == "cuda"

        # Graph extraction
        extractor_config = GraphExtractorConfig(
            junction_threshold=config.get("junction_threshold", 0.3),
            junction_min_distance=config.get("junction_min_distance", 5),
        )
        self.graph_extractor = GraphExtractor(extractor_config)

        # Loss function
        self.criterion = GraphLoss(
            existence_weight=config.get("existence_weight", 1.0),
            assignment_weight=config.get("assignment_weight", 1.0),
            refinement_weight=config.get("refinement_weight", 0.5),
        )

        # Optimizer
        self.optimizer = AdamW(
            self.graph_head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * self.epochs
        scheduler_type = config.get("scheduler", "onecycle")

        if scheduler_type == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos",
            )
            self.step_per_batch = True
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
            )
            self.step_per_batch = False

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/graph"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.get("save_every", 1)

        # Best model tracking
        self.best_val_metric = 0.0
        self.best_epoch = 0

        # Logging
        self.log_every = config.get("log_every", 50)
        self.use_wandb = config.get("use_wandb", False) and WANDB_AVAILABLE

        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "cp-detector-graph"),
                config=config,
            )

        # Matching params
        self.vertex_match_threshold = config.get("vertex_match_threshold", 8.0)
        self.image_size = config.get("image_size", 512)
        self.padding = config.get("padding", 50)

        self.current_epoch = 0

        # Thread pool for parallel graph extraction
        self._executor = None

    def _extract_graphs_parallel(self, seg_preds, junction_heatmaps):
        """Extract graphs from multiple samples in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        batch_size = len(seg_preds)

        def extract_single(i):
            try:
                return self.graph_extractor.extract(seg_preds[i], junction_heatmaps[i])
            except Exception:
                return None

        # Use thread pool (reuse across batches)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=min(8, batch_size))

        # Submit all extraction tasks
        futures = [self._executor.submit(extract_single, i) for i in range(batch_size)]

        # Collect results
        return [f.result() for f in futures]

    def train(self) -> dict:
        """Run full training loop."""
        print(f"\nTraining for {self.epochs} epochs...")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Device: {self.device}")
        print(f"AMP: {self.use_amp}")

        # Log initial visualization before training to catch errors early
        if self.use_wandb:
            print("\nLogging initial visualization to verify pipeline...")
            self._log_graph_visualizations(epoch=-1)
            print("Initial visualization logged successfully!")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self._log_epoch(epoch, train_metrics, val_metrics)

            # Save checkpoint
            is_best = val_metrics["existence_f1"] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics["existence_f1"]
                self.best_epoch = epoch

            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

        self.save_checkpoint(self.epochs - 1, is_best=False, final=True)

        return {
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch,
        }

    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.graph_head.train()

        metrics = {
            "loss": 0.0,
            "existence_loss": 0.0,
            "assignment_loss": 0.0,
            "refinement_loss": 0.0,
        }
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)

            # Get pixel head outputs (frozen)
            with torch.no_grad():
                pixel_outputs = self.pixel_model(images, return_features=True)

            # Extract graphs in parallel (CPU-bound)
            seg_preds = pixel_outputs["segmentation"].argmax(dim=1).cpu().numpy()
            junction_heatmaps = pixel_outputs["junction"][:, 0].cpu().numpy()

            candidate_graphs = self._extract_graphs_parallel(seg_preds, junction_heatmaps)

            # Collect valid samples for batched GPU processing
            vertices_list = []
            edge_index_list = []
            labels_list = []
            valid_indices = []

            gt_graph = batch.get("graph")

            for i in range(images.shape[0]):
                try:
                    candidate_graph = candidate_graphs[i]
                    if candidate_graph is None or len(candidate_graph.vertices) < 2:
                        continue

                    vertices = torch.from_numpy(candidate_graph.vertices).float().to(self.device)
                    edge_index = torch.from_numpy(candidate_graph.edges.T).long().to(self.device)

                    if edge_index.shape[1] == 0:
                        continue

                    # Get GT graph data (already in pixel coords from dataset)
                    if gt_graph is None or gt_graph[i] is None:
                        continue

                    gt_vertices = gt_graph[i]["vertices"].to(self.device)
                    gt_edges = gt_graph[i]["edges"].T.to(self.device)  # (N, 2) -> (2, E)
                    gt_assignments = gt_graph[i]["assignments"].to(self.device)

                    # Generate labels
                    labels = generate_graph_labels(
                        candidate_vertices=vertices,
                        candidate_edges=edge_index,
                        gt_vertices=gt_vertices,
                        gt_edges=gt_edges,
                        gt_assignments=gt_assignments,
                        vertex_match_threshold=self.vertex_match_threshold,
                    )

                    vertices_list.append(vertices)
                    edge_index_list.append(edge_index)
                    labels_list.append(labels)
                    valid_indices.append(i)

                except Exception as e:
                    if batch_idx == 0 and i == 0:
                        print(f"Warning: Sample prep failed: {e}")
                    continue

            if len(valid_indices) == 0:
                continue

            # Batched forward pass through graph head (single GPU call for all samples)
            try:
                seg_probs = torch.softmax(pixel_outputs["segmentation"][valid_indices], dim=1)
                backbone_feats = pixel_outputs["features"][valid_indices]

                with autocast(enabled=self.use_amp):
                    outputs = self.graph_head.forward_batch(
                        vertices_list=vertices_list,
                        edge_index_list=edge_index_list,
                        backbone_features=backbone_feats,
                        seg_probs=seg_probs,
                        image_size=self.image_size,
                    )

                    # Concatenate all labels for batched loss computation
                    all_edge_existence = torch.cat([l.edge_existence for l in labels_list])
                    all_edge_assignment = torch.cat([l.edge_assignment for l in labels_list])
                    all_vertex_offset = torch.cat([l.vertex_offset for l in labels_list])
                    all_vertex_matched = torch.cat([l.vertex_matched for l in labels_list])

                    targets = {
                        "edge_existence": all_edge_existence,
                        "edge_assignment": all_edge_assignment,
                        "vertex_offset": all_vertex_offset,
                        "vertex_matched": all_vertex_matched,
                    }
                    loss_dict = self.criterion(outputs, targets)

                batch_loss = loss_dict["loss"]
                batch_losses = {
                    "existence": loss_dict["existence_loss"].item(),
                    "assignment": loss_dict["assignment_loss"].item(),
                    "refinement": loss_dict["refinement_loss"].item(),
                }
                valid_samples = len(valid_indices)

            except Exception as e:
                if batch_idx == 0:
                    print(f"Warning: Batched forward failed: {e}")
                    import traceback
                    traceback.print_exc()
                continue

            # Backward
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.graph_head.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch_loss.backward()
                nn.utils.clip_grad_norm_(self.graph_head.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.step_per_batch:
                self.scheduler.step()

            # Update metrics (batch_losses already contains totals, not per-sample)
            metrics["loss"] += batch_loss.item()
            metrics["existence_loss"] += batch_losses["existence"]
            metrics["assignment_loss"] += batch_losses["assignment"]
            metrics["refinement_loss"] += batch_losses["refinement"]
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{batch_loss.item():.4f}",
                "samples": valid_samples,
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            if self.use_wandb and (batch_idx + 1) % self.log_every == 0:
                wandb.log({
                    "train/loss": batch_loss.item(),
                    "train/existence_loss": batch_losses["existence"],
                    "train/assignment_loss": batch_losses["assignment"],
                    "train/refinement_loss": batch_losses["refinement"],
                    "train/valid_samples": valid_samples,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                })

        if not self.step_per_batch:
            self.scheduler.step()

        if num_batches > 0:
            for key in metrics:
                metrics[key] /= num_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.graph_head.eval()

        all_metrics = []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)
            pixel_outputs = self.pixel_model(images, return_features=True)

            # Extract graphs in parallel
            seg_preds = pixel_outputs["segmentation"].argmax(dim=1).cpu().numpy()
            junction_heatmaps = pixel_outputs["junction"][:, 0].cpu().numpy()
            candidate_graphs = self._extract_graphs_parallel(seg_preds, junction_heatmaps)

            for i in range(images.shape[0]):
                try:
                    candidate_graph = candidate_graphs[i]
                    if candidate_graph is None or len(candidate_graph.vertices) < 2:
                        continue

                    vertices = torch.from_numpy(candidate_graph.vertices).float().to(self.device)
                    edge_index = torch.from_numpy(candidate_graph.edges.T).long().to(self.device)

                    if edge_index.shape[1] == 0:
                        continue

                    gt_graph = batch.get("graph")
                    if gt_graph is None or gt_graph[i] is None:
                        continue

                    gt_vertices = gt_graph[i]["vertices"].to(self.device)
                    gt_edges = gt_graph[i]["edges"].T.to(self.device)  # (N, 2) -> (2, E)
                    gt_assignments = gt_graph[i]["assignments"].to(self.device)

                    labels = generate_graph_labels(
                        candidate_vertices=vertices,
                        candidate_edges=edge_index,
                        gt_vertices=gt_vertices,
                        gt_edges=gt_edges,
                        gt_assignments=gt_assignments,
                        vertex_match_threshold=self.vertex_match_threshold,
                    )

                    seg_probs = torch.softmax(pixel_outputs["segmentation"][i:i+1], dim=1)

                    outputs = self.graph_head(
                        vertices=vertices,
                        edge_index=edge_index,
                        backbone_features=pixel_outputs["features"][i:i+1],
                        seg_probs=seg_probs,
                        image_size=self.image_size,
                    )

                    targets = {
                        "edge_existence": labels.edge_existence,
                        "edge_assignment": labels.edge_assignment,
                        "vertex_offset": labels.vertex_offset,
                        "vertex_matched": labels.vertex_matched,
                    }

                    sample_metrics = compute_graph_metrics(outputs, targets)
                    all_metrics.append(sample_metrics)

                except Exception:
                    continue

        if len(all_metrics) == 0:
            return {
                "existence_precision": 0.0,
                "existence_recall": 0.0,
                "existence_f1": 0.0,
                "assignment_accuracy": 0.0,
                "mean_offset_error": 0.0,
            }

        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        return avg_metrics

    def _log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """Log epoch metrics."""
        print(
            f"\nEpoch {epoch + 1}/{self.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Val Exist F1: {val_metrics['existence_f1']:.4f} - "
            f"Val Assign Acc: {val_metrics['assignment_accuracy']:.4f}"
        )

        if self.use_wandb:
            log_dict = {"epoch": epoch}
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = value
            for key, value in val_metrics.items():
                log_dict[f"val/{key}"] = value
            wandb.log(log_dict)

            # Log visualizations every few epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self._log_graph_visualizations(epoch)

    @torch.no_grad()
    def _log_graph_visualizations(self, epoch: int, num_samples: int = 3):
        """Log graph prediction visualizations to W&B."""
        import matplotlib.pyplot as plt

        self.graph_head.eval()
        wandb_images = []

        # Get a few validation samples
        val_iter = iter(self.val_loader)
        batch = next(val_iter)

        images = batch["image"].to(self.device)
        pixel_outputs = self.pixel_model(images, return_features=True)

        # Color map for assignments: M=red, V=blue, B=black, U=gray
        colors = {0: 'red', 1: 'blue', 2: 'black', 3: 'gray', 4: 'green'}

        for i in range(min(num_samples, images.shape[0])):
            try:
                seg_pred = pixel_outputs["segmentation"][i].argmax(dim=0).cpu().numpy()
                junction_heatmap = pixel_outputs["junction"][i, 0].cpu().numpy()

                candidate_graph = self.graph_extractor.extract(seg_pred, junction_heatmap)
                if candidate_graph is None or len(candidate_graph.vertices) < 2:
                    continue

                vertices = torch.from_numpy(candidate_graph.vertices).float().to(self.device)
                edge_index = torch.from_numpy(candidate_graph.edges.T).long().to(self.device)

                if edge_index.shape[1] == 0:
                    continue

                seg_probs = torch.softmax(pixel_outputs["segmentation"][i:i+1], dim=1)
                outputs = self.graph_head(
                    vertices=vertices,
                    edge_index=edge_index,
                    backbone_features=pixel_outputs["features"][i:i+1],
                    seg_probs=seg_probs,
                    image_size=self.image_size,
                )

                # Get predictions
                edge_probs = torch.sigmoid(outputs["edge_existence"]).cpu().numpy()
                edge_preds = edge_probs > 0.5
                assign_preds = outputs["edge_assignment"].argmax(dim=1).cpu().numpy()

                vertices_np = vertices.cpu().numpy()
                edges_np = edge_index.cpu().numpy()

                # Create figure with 3 subplots
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                axes[0].imshow(img)
                axes[0].set_title("Input Image")
                axes[0].axis("off")

                # Candidate graph (all edges)
                axes[1].imshow(img)
                for e in range(edges_np.shape[1]):
                    v1, v2 = edges_np[0, e], edges_np[1, e]
                    axes[1].plot(
                        [vertices_np[v1, 0], vertices_np[v2, 0]],
                        [vertices_np[v1, 1], vertices_np[v2, 1]],
                        'gray', alpha=0.3, linewidth=1
                    )
                axes[1].scatter(vertices_np[:, 0], vertices_np[:, 1], c='blue', s=10)
                axes[1].set_title(f"Candidate Graph ({edges_np.shape[1]} edges)")
                axes[1].axis("off")

                # Predicted graph (filtered + colored)
                axes[2].imshow(img)
                for e in range(edges_np.shape[1]):
                    if edge_preds[e]:
                        v1, v2 = edges_np[0, e], edges_np[1, e]
                        color = colors.get(assign_preds[e], 'gray')
                        axes[2].plot(
                            [vertices_np[v1, 0], vertices_np[v2, 0]],
                            [vertices_np[v1, 1], vertices_np[v2, 1]],
                            color, linewidth=2
                        )
                kept = edge_preds.sum()
                axes[2].set_title(f"Predicted Graph ({kept} edges)")
                axes[2].axis("off")

                plt.tight_layout()
                wandb_images.append(wandb.Image(fig, caption=f"Sample {i}"))
                plt.close(fig)

            except Exception as e:
                print(f"Visualization failed for sample {i}: {e}")
                continue

        if wandb_images:
            wandb.log({
                "val/graph_predictions": wandb_images,
                "epoch": epoch,
            })

    def save_checkpoint(self, epoch: int, is_best: bool = False, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "graph_head_state_dict": self.graph_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch,
            "config": self.config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save regular checkpoint
        if not final:
            path = self.checkpoint_dir / f"graph_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")

        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "graph_latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "graph_best.pt")
            print(f"New best model (epoch {epoch + 1}, F1: {self.best_val_metric:.4f})")

        # Save final
        if final:
            torch.save(checkpoint, self.checkpoint_dir / "graph_final.pt")

    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.graph_head.load_state_dict(checkpoint["graph_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_metric = checkpoint["best_val_metric"]
        self.best_epoch = checkpoint["best_epoch"]
        self.current_epoch = checkpoint["epoch"] + 1

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")


def main():
    parser = argparse.ArgumentParser(description="Train Graph Head (Phase 2)")
    parser.add_argument(
        "--pixel-checkpoint",
        type=str,
        required=True,
        help="Path to pixel head checkpoint",
    )
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
        help="Directory containing images (optional, renders on-the-fly if not provided)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image size (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs (default: 30)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--node-dim",
        type=int,
        default=128,
        help="Node feature dimension (default: 128)",
    )
    parser.add_argument(
        "--edge-dim",
        type=int,
        default=128,
        help="Edge feature dimension (default: 128)",
    )
    parser.add_argument(
        "--num-gnn-layers",
        type=int,
        default=4,
        help="Number of GNN layers (default: 4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/graph",
        help="Checkpoint directory (default: checkpoints/graph)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cp-detector-graph",
        help="W&B project name",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load pixel model
    print(f"\nLoading pixel model from: {args.pixel_checkpoint}")
    pixel_checkpoint = torch.load(args.pixel_checkpoint, map_location=device, weights_only=False)

    pixel_model = CreasePatternDetector(
        backbone="hrnet_w32",
        num_seg_classes=5,
    )
    pixel_model.load_state_dict(pixel_checkpoint["model_state_dict"])
    pixel_model.eval()
    print("Pixel model loaded")

    # Get backbone output channels
    with torch.no_grad():
        dummy = torch.zeros(1, 3, args.image_size, args.image_size)
        dummy_out = pixel_model(dummy, return_features=True)
        backbone_channels = dummy_out["features"].shape[1]
        print(f"Backbone channels: {backbone_channels}")

    # Create graph head
    print("\nCreating graph head...")
    graph_head = GraphHead(
        backbone_channels=backbone_channels,
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_classes=4,  # M=0, V=1, B=2, U=3 (matches FOLD assignments)
    )
    print(f"Graph head parameters: {sum(p.numel() for p in graph_head.parameters()):,}")

    # Create datasets
    print(f"\nLoading data from: {args.fold_dir}")
    if args.image_dir:
        print(f"Image directory: {args.image_dir}")

    # Train dataset
    train_dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_dir=args.image_dir,
        image_size=args.image_size,
        split="train",
    )

    # Validation dataset
    val_dataset = CreasePatternDataset(
        fold_dir=args.fold_dir,
        image_dir=args.image_dir,
        image_size=args.image_size,
        split="val",
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Config
    config = {
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "node_dim": args.node_dim,
        "edge_dim": args.edge_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "backbone_channels": backbone_channels,
        "checkpoint_dir": args.checkpoint_dir,
        "amp": not args.no_amp,
        "use_wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "scheduler": "onecycle",
        "existence_weight": 1.0,
        "assignment_weight": 1.0,
        "refinement_weight": 0.5,
        "vertex_match_threshold": 8.0,
        "junction_threshold": 0.3,
        "junction_min_distance": 5,
    }

    # Create trainer
    trainer = GraphHeadTrainer(
        pixel_model=pixel_model,
        graph_head=graph_head,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"Best F1: {results['best_val_metric']:.4f} at epoch {results['best_epoch'] + 1}")


if __name__ == "__main__":
    main()
