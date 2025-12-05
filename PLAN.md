# Plan: Add W&B Junction Prediction Visualization

## Goal
Visualize junction predictions during training in Weights & Biases to monitor whether loss function changes lead to better learning.

## Implementation Steps

### 1. Add visualization method to Trainer class
Add a `_log_predictions` method in `src/training/trainer.py` that:
- Takes a batch of images, ground truth, and predictions
- Creates side-by-side comparison images (input, GT heatmap, predicted heatmap)
- Logs them to W&B using `wandb.Image()`

### 2. Hook into validate() method
After computing validation metrics (around line 305), call the visualization method:
- Sample 4 images per validation run to avoid overhead
- Only log when W&B is enabled
- Log at the end of each epoch

### 3. Visualization format
Create a grid showing for each sample:
- Input image
- Ground truth junction heatmap
- Predicted junction heatmap (with min/max values annotated)

## Code Changes

**File: `src/training/trainer.py`**

```python
# Add to imports
import numpy as np

# Add new method after _compute_junction_f1 (~line 354):
def _log_predictions(
    self,
    images: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    num_samples: int = 4,
) -> None:
    """Log prediction visualizations to W&B."""
    if not self.use_wandb:
        return

    import matplotlib.pyplot as plt

    batch_size = min(num_samples, images.shape[0])

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
        wandb.log({f"val/junction_pred_{i}": wandb.Image(fig)})
        plt.close(fig)
```

**Modify validate() method** - after line 305, add:
```python
# Log visualizations for first batch only
if batch_idx == 0:
    self._log_predictions(images, targets, outputs)
```

This requires adding `enumerate()` to the validation loop.

## Benefits
- Visual feedback on whether junctions are being detected as sharp peaks vs blobby regions
- Easy comparison across epochs and loss function experiments
- Can spot edge artifacts early without waiting for full training
