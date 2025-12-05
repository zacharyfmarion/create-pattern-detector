"""
Training loop for Graph Head (Phase 2).

Trains the Graph Head with frozen backbone + pixel head.
Uses over-complete candidate graphs extracted from pixel head outputs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models import CreasePatternDetector
from ..models.graph.graph_head import GraphHead
from ..models.losses.graph_loss import GraphLoss, GraphLossWithBatching, compute_graph_metrics
from ..data.graph_labels import generate_graph_labels, crease_pattern_to_tensors
from ..postprocessing.graph_extraction import CandidateGraphExtractor


class GraphTrainer:
    """
    Training loop for Graph Head (Phase 2).

    Features:
    - Frozen backbone + pixel head
    - Graph extraction from pixel head outputs
    - PyG-style batching for variable-size graphs
    - Edge existence + assignment + vertex refinement losses
    """

    def __init__(
        self,
        model: CreasePatternDetector,
        graph_head: GraphHead,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        """
        Initialize Graph trainer.

        Args:
            model: Trained CreasePatternDetector (frozen)
            graph_head: GraphHead to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.graph_head = graph_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Freeze backbone and pixel head
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Training hyperparameters
        self.epochs = config.get("epochs", 30)
        self.learning_rate = config.get("learning_rate", 5e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_amp = config.get("amp", True) and device == "cuda"

        # Graph extraction
        self.graph_extractor = CandidateGraphExtractor(
            junction_threshold=config.get("junction_threshold", 0.3),
            junction_min_distance=config.get("junction_min_distance", 5),
            seg_threshold=config.get("seg_threshold", 0.5),
        )

        # Loss function
        self.criterion = GraphLoss(
            existence_weight=config.get("existence_weight", 1.0),
            assignment_weight=config.get("assignment_weight", 1.0),
            refinement_weight=config.get("refinement_weight", 0.5),
        )

        # Optimizer (only graph head parameters)
        self.optimizer = AdamW(
            self.graph_head.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
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
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
            )

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints/graph"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.get("save_every", 5)

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

        # Matching parameters
        self.vertex_match_threshold = config.get("vertex_match_threshold", 8.0)
        self.image_size = config.get("image_size", 512)
        self.padding = config.get("padding", 50)

        self.current_epoch = 0

    def train(self) -> Dict[str, float]:
        """Run full training loop."""
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

    def train_epoch(self) -> Dict[str, float]:
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
            # Move images to device
            images = batch["image"].to(self.device)

            # Get pixel head outputs (frozen)
            with torch.no_grad():
                pixel_outputs = self.model(images)

            # Process each sample in batch
            batch_loss = torch.tensor(0.0, device=self.device)
            batch_losses = {"existence": 0.0, "assignment": 0.0, "refinement": 0.0}
            valid_samples = 0

            for i in range(images.shape[0]):
                try:
                    # Extract candidate graph
                    seg_probs = torch.softmax(pixel_outputs["segmentation"][i:i+1], dim=1)
                    junction_heatmap = pixel_outputs["junction"][i:i+1]

                    candidate_graph = self.graph_extractor.extract(
                        seg_probs[0].cpu().numpy(),
                        junction_heatmap[0, 0].cpu().numpy(),
                    )

                    if candidate_graph is None or len(candidate_graph.vertices) == 0:
                        continue

                    # Convert to tensors
                    vertices = torch.from_numpy(candidate_graph.vertices).float().to(self.device)
                    edge_index = torch.from_numpy(candidate_graph.edges.T).long().to(self.device)

                    if edge_index.shape[1] == 0:
                        continue

                    # Get GT from batch
                    gt_cp = batch.get("crease_pattern")
                    if gt_cp is None:
                        continue

                    gt_vertices, gt_edges, gt_assignments = crease_pattern_to_tensors(
                        gt_cp[i],
                        image_size=self.image_size,
                        padding=self.padding,
                    )
                    gt_vertices = gt_vertices.to(self.device)
                    gt_edges = gt_edges.to(self.device)
                    gt_assignments = gt_assignments.to(self.device)

                    # Generate labels
                    labels = generate_graph_labels(
                        candidate_vertices=vertices,
                        candidate_edges=edge_index,
                        gt_vertices=gt_vertices,
                        gt_edges=gt_edges,
                        gt_assignments=gt_assignments,
                        vertex_match_threshold=self.vertex_match_threshold,
                    )

                    # Forward through graph head
                    with autocast(enabled=self.use_amp):
                        outputs = self.graph_head(
                            vertices=vertices,
                            edge_index=edge_index,
                            backbone_features=pixel_outputs["features"][i:i+1],
                            seg_probs=seg_probs,
                            orientation=pixel_outputs.get("orientation", None),
                            image_size=self.image_size,
                        )

                        # Compute loss
                        targets = {
                            "edge_existence": labels.edge_existence,
                            "edge_assignment": labels.edge_assignment,
                            "vertex_offset": labels.vertex_offset,
                            "vertex_matched": labels.vertex_matched,
                        }
                        loss_dict = self.criterion(outputs, targets)

                    batch_loss = batch_loss + loss_dict["loss"]
                    batch_losses["existence"] += loss_dict["existence_loss"].item()
                    batch_losses["assignment"] += loss_dict["assignment_loss"].item()
                    batch_losses["refinement"] += loss_dict["refinement_loss"].item()
                    valid_samples += 1

                except Exception as e:
                    # Skip problematic samples
                    continue

            if valid_samples == 0:
                continue

            # Average loss across samples
            batch_loss = batch_loss / valid_samples

            # Backward pass
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

            # Step scheduler
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # Update metrics
            metrics["loss"] += batch_loss.item()
            metrics["existence_loss"] += batch_losses["existence"] / valid_samples
            metrics["assignment_loss"] += batch_losses["assignment"] / valid_samples
            metrics["refinement_loss"] += batch_losses["refinement"] / valid_samples
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{batch_loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            # Log to W&B
            if self.use_wandb and (batch_idx + 1) % self.log_every == 0:
                wandb.log({
                    "train/loss": batch_loss.item(),
                    "train/existence_loss": batch_losses["existence"] / valid_samples,
                    "train/assignment_loss": batch_losses["assignment"] / valid_samples,
                    "train/refinement_loss": batch_losses["refinement"] / valid_samples,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                })

        # Step scheduler (per-epoch)
        if not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()

        # Average metrics
        if num_batches > 0:
            for key in metrics:
                metrics[key] /= num_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.graph_head.eval()

        all_metrics = []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch["image"].to(self.device)

            # Get pixel head outputs
            pixel_outputs = self.model(images)

            for i in range(images.shape[0]):
                try:
                    # Extract candidate graph
                    seg_probs = torch.softmax(pixel_outputs["segmentation"][i:i+1], dim=1)
                    junction_heatmap = pixel_outputs["junction"][i:i+1]

                    candidate_graph = self.graph_extractor.extract(
                        seg_probs[0].cpu().numpy(),
                        junction_heatmap[0, 0].cpu().numpy(),
                    )

                    if candidate_graph is None or len(candidate_graph.vertices) == 0:
                        continue

                    vertices = torch.from_numpy(candidate_graph.vertices).float().to(self.device)
                    edge_index = torch.from_numpy(candidate_graph.edges.T).long().to(self.device)

                    if edge_index.shape[1] == 0:
                        continue

                    # Get GT
                    gt_cp = batch.get("crease_pattern")
                    if gt_cp is None:
                        continue

                    gt_vertices, gt_edges, gt_assignments = crease_pattern_to_tensors(
                        gt_cp[i],
                        image_size=self.image_size,
                        padding=self.padding,
                    )
                    gt_vertices = gt_vertices.to(self.device)
                    gt_edges = gt_edges.to(self.device)
                    gt_assignments = gt_assignments.to(self.device)

                    labels = generate_graph_labels(
                        candidate_vertices=vertices,
                        candidate_edges=edge_index,
                        gt_vertices=gt_vertices,
                        gt_edges=gt_edges,
                        gt_assignments=gt_assignments,
                        vertex_match_threshold=self.vertex_match_threshold,
                    )

                    # Forward
                    outputs = self.graph_head(
                        vertices=vertices,
                        edge_index=edge_index,
                        backbone_features=pixel_outputs["features"][i:i+1],
                        seg_probs=seg_probs,
                        orientation=pixel_outputs.get("orientation", None),
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

        # Average metrics
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

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
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

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        final: bool = False,
    ) -> None:
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
            path = self.checkpoint_dir / f"graph_checkpoint_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")

        # Save latest
        latest_path = self.checkpoint_dir / "graph_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "graph_best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model (epoch {epoch + 1}, F1: {self.best_val_metric:.4f})")

        # Save final
        if final:
            final_path = self.checkpoint_dir / "graph_final.pt"
            torch.save(checkpoint, final_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load graph head from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.graph_head.load_state_dict(checkpoint["graph_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_metric = checkpoint["best_val_metric"]
        self.best_epoch = checkpoint["best_epoch"]
        self.current_epoch = checkpoint["epoch"] + 1

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded graph checkpoint from epoch {checkpoint['epoch'] + 1}")
