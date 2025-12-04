"""
Training loop for crease pattern detection.

Handles training, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models import CreasePatternDetector
from ..models.losses import PixelLoss


class Trainer:
    """
    Training loop for crease pattern detector.

    Features:
    - Mixed precision training (AMP)
    - Learning rate scheduling (OneCycle or Cosine)
    - Gradient clipping
    - Checkpoint management
    - W&B logging (optional)
    """

    def __init__(
        self,
        model: CreasePatternDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        """
        Initialize trainer.

        Args:
            model: CreasePatternDetector model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Training hyperparameters
        self.epochs = config.get("epochs", 50)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.use_amp = config.get("amp", True) and device == "cuda"

        # Loss function
        self.criterion = PixelLoss(
            seg_weight=config.get("seg_weight", 1.0),
            orient_weight=config.get("orient_weight", 0.5),
            junction_weight=config.get("junction_weight", 1.0),
            junction_pos_weight=config.get("junction_pos_weight", 50.0),
            junction_loss_type=config.get("junction_loss_type", "bce"),  # BCE is standard for heatmaps
        )

        # Optimizer with different LR for backbone
        backbone_lr_mult = config.get("backbone_lr_mult", 0.1)
        self.optimizer = AdamW(
            model.get_param_groups(self.learning_rate, backbone_lr_mult),
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
        else:  # cosine
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
            )

        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config.get("save_every", 5)

        # Best model tracking
        self.best_val_metric = 0.0
        self.best_epoch = 0

        # Logging
        self.log_every = config.get("log_every", 100)
        self.use_wandb = config.get("use_wandb", False) and WANDB_AVAILABLE

        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "cp-detector"),
                config=config,
            )

        # Current epoch
        self.current_epoch = 0

    def train(self) -> Dict[str, float]:
        """
        Run full training loop.

        Returns:
            Dictionary with final metrics
        """
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log metrics
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Save checkpoint
            is_best = val_metrics["seg_iou"] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics["seg_iou"]
                self.best_epoch = epoch

            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

        # Save final checkpoint
        self.save_checkpoint(self.epochs - 1, is_best=False, final=True)

        return {
            "best_val_metric": self.best_val_metric,
            "best_epoch": self.best_epoch,
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        metrics = {
            "loss": 0.0,
            "seg_loss": 0.0,
            "orient_loss": 0.0,
            "junction_loss": 0.0,
        }
        num_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["image"].to(self.device)
            targets = {
                "segmentation": batch["segmentation"].to(self.device),
                "orientation": batch["orientation"].to(self.device),
                "junction_heatmap": batch["junction_heatmap"].to(self.device),
            }

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["total"]

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Step scheduler (if per-step)
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # Update metrics
            metrics["loss"] += loss.item()
            metrics["seg_loss"] += loss_dict["seg"].item()
            metrics["orient_loss"] += loss_dict["orient"].item()
            metrics["junction_loss"] += loss_dict["junction"].item()

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            # Log to W&B
            if self.use_wandb and (batch_idx + 1) % self.log_every == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/seg_loss": loss_dict["seg"].item(),
                    "train/orient_loss": loss_dict["orient"].item(),
                    "train/junction_loss": loss_dict["junction"].item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.current_epoch * num_batches + batch_idx,
                })

        # Step scheduler (if per-epoch)
        if not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        metrics = {
            "loss": 0.0,
            "seg_loss": 0.0,
            "orient_loss": 0.0,
            "junction_loss": 0.0,
            "seg_iou": 0.0,
            "junction_f1": 0.0,
        }
        num_batches = len(self.val_loader)

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            images = batch["image"].to(self.device)
            targets = {
                "segmentation": batch["segmentation"].to(self.device),
                "orientation": batch["orientation"].to(self.device),
                "junction_heatmap": batch["junction_heatmap"].to(self.device),
            }

            # Forward pass
            outputs = self.model(images)
            loss_dict = self.criterion(outputs, targets)

            # Update loss metrics
            metrics["loss"] += loss_dict["total"].item()
            metrics["seg_loss"] += loss_dict["seg"].item()
            metrics["orient_loss"] += loss_dict["orient"].item()
            metrics["junction_loss"] += loss_dict["junction"].item()

            # Compute IoU for segmentation
            seg_iou = self._compute_seg_iou(outputs["segmentation"], targets["segmentation"])
            metrics["seg_iou"] += seg_iou

            # Compute F1 for junction detection
            junction_f1 = self._compute_junction_f1(
                outputs["junction"],
                targets["junction_heatmap"],
            )
            metrics["junction_f1"] += junction_f1

            # Log visualizations every 50 batches
            if batch_idx % 50 == 0:
                self._log_predictions(images, targets, outputs, batch_idx=batch_idx)

        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches

        return metrics

    def _compute_seg_iou(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute mean IoU for crease classes (M, V, B, U)."""
        pred = logits.argmax(dim=1)
        ious = []

        # Compute IoU for each crease class (skip background)
        for c in range(1, 5):  # Classes 1-4: M, V, B, U
            pred_c = pred == c
            target_c = targets == c

            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()

            if union > 0:
                ious.append((intersection / union).item())

        return sum(ious) / len(ious) if ious else 0.0

    def _compute_junction_f1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> float:
        """Compute F1 score for junction detection."""
        pred_binary = (pred > threshold).float()
        target_binary = (target > 0.5).float()

        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return f1.item()

    def _log_predictions(
        self,
        images: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        num_samples: int = 4,
        batch_idx: int = 0,
    ) -> None:
        """Log prediction visualizations to W&B."""
        if not self.use_wandb:
            return

        import matplotlib.pyplot as plt
        import numpy as np

        batch_size = min(num_samples, images.shape[0])
        wandb_images = []

        for i in range(batch_size):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # Input image
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[0].imshow(img)
            axes[0].set_title("Input")
            axes[0].axis("off")

            # GT junction heatmap
            gt = targets["junction_heatmap"][i].cpu().numpy()
            axes[1].imshow(gt, cmap="hot", vmin=0, vmax=1)
            axes[1].set_title("GT Junctions")
            axes[1].axis("off")

            # Predicted junction heatmap
            pred = outputs["junction"][i, 0].cpu().numpy()
            axes[2].imshow(pred, cmap="hot", vmin=0, vmax=1)
            axes[2].set_title(f"Predicted (min={pred.min():.3f}, max={pred.max():.3f})")
            axes[2].axis("off")

            plt.tight_layout()
            wandb_images.append(wandb.Image(fig, caption=f"Batch {batch_idx} Sample {i}"))
            plt.close(fig)

        wandb.log({
            f"val/junction_predictions_batch_{batch_idx}": wandb_images,
            "epoch": self.current_epoch,
        })

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
            f"Val Loss: {val_metrics['loss']:.4f} - "
            f"Val IoU: {val_metrics['seg_iou']:.4f} - "
            f"Val Junction F1: {val_metrics['junction_f1']:.4f}"
        )

        if self.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_metrics["loss"],
                "train/epoch_seg_loss": train_metrics["seg_loss"],
                "train/epoch_orient_loss": train_metrics["orient_loss"],
                "train/epoch_junction_loss": train_metrics["junction_loss"],
                "val/loss": val_metrics["loss"],
                "val/seg_loss": val_metrics["seg_loss"],
                "val/orient_loss": val_metrics["orient_loss"],
                "val/junction_loss": val_metrics["junction_loss"],
                "val/seg_iou": val_metrics["seg_iou"],
                "val/junction_f1": val_metrics["junction_f1"],
            })

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        final: bool = False,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
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
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")

        # Always save latest checkpoint for easy resumption
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model (epoch {epoch + 1}, IoU: {self.best_val_metric:.4f})")

        # Save final model
        if final:
            final_path = self.checkpoint_dir / "final_model.pt"
            torch.save(checkpoint, final_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_metric = checkpoint["best_val_metric"]
        self.best_epoch = checkpoint["best_epoch"]
        self.current_epoch = checkpoint["epoch"] + 1

        # Recreate scheduler for remaining epochs instead of loading old state
        # OneCycleLR can't be extended, so we create a fresh one for remaining training
        remaining_epochs = self.epochs - self.current_epoch
        if remaining_epochs > 0:
            total_steps = len(self.train_loader) * remaining_epochs
            scheduler_type = self.config.get("scheduler", "onecycle")

            if scheduler_type == "onecycle":
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=self.learning_rate,
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy="cos",
                )
            else:  # cosine
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=remaining_epochs,
                )
            print(f"Created new scheduler for {remaining_epochs} remaining epochs")

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
