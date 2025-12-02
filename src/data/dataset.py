"""
PyTorch Dataset for crease pattern detection.

Loads FOLD files, generates ground truth, and applies augmentations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .fold_parser import FOLDParser, CreasePattern
from .annotations import GroundTruthGenerator


class CreasePatternDataset(Dataset):
    """
    Dataset for crease pattern detection training.

    Loads pre-rendered images and FOLD files, generates ground truth
    annotations on the fly.
    """

    def __init__(
        self,
        fold_dir: str | Path,
        image_dir: Optional[str | Path] = None,
        image_size: int = 1024,
        padding: int = 50,
        line_width: int = 3,
        transform: Optional[Callable] = None,
        split: str = "train",
        split_seed: int = 42,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
    ):
        """
        Initialize the dataset.

        Args:
            fold_dir: Directory containing .fold files
            image_dir: Directory containing pre-rendered .png images (optional)
                      If None, images are rendered on the fly
            image_size: Size of images (assumed square)
            padding: Padding used in rendering
            line_width: Line width used in rendering
            transform: Optional augmentation transform
            split: One of 'train', 'val', 'test'
            split_seed: Random seed for splitting
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
        """
        self.fold_dir = Path(fold_dir)
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_size = image_size
        self.padding = padding
        self.line_width = line_width
        self.transform = transform
        self.split = split

        # Initialize parser and ground truth generator
        self.parser = FOLDParser()
        self.gt_generator = GroundTruthGenerator(
            image_size=image_size,
            padding=padding,
            line_width=line_width,
        )

        # Discover all FOLD files
        all_files = sorted(self.fold_dir.glob("*.fold"))

        if len(all_files) == 0:
            raise ValueError(f"No .fold files found in {fold_dir}")

        # Split into train/val/test
        rng = np.random.default_rng(split_seed)
        indices = rng.permutation(len(all_files))

        n_train = int(len(all_files) * train_ratio)
        n_val = int(len(all_files) * val_ratio)

        if split == "train":
            selected_indices = indices[:n_train]
        elif split == "val":
            selected_indices = indices[n_train : n_train + n_val]
        else:  # test
            selected_indices = indices[n_train + n_val :]

        self.files = [all_files[i] for i in selected_indices]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fold_path = self.files[idx]

        # Load image
        image = self._load_image(fold_path)

        # Parse FOLD and generate ground truth
        cp = self.parser.parse(fold_path)
        gt = self.gt_generator.generate(cp)

        # Apply augmentations if provided
        if self.transform:
            transformed = self.transform(
                image=image,
                segmentation=gt["segmentation"],
                orientation=gt["orientation"],
                junction_heatmap=gt["junction_heatmap"],
                vertices=gt["vertices"],
                edges=gt["edges"],
                assignments=gt["assignments"],
            )
            image = transformed["image"]
            gt["segmentation"] = transformed["segmentation"]
            gt["orientation"] = transformed["orientation"]
            gt["junction_heatmap"] = transformed["junction_heatmap"]
            gt["vertices"] = transformed["vertices"]
            gt["edges"] = transformed["edges"]
            gt["assignments"] = transformed["assignments"]

        # Convert to tensors
        sample = {
            # Image: (3, H, W) float32 normalized to [0, 1]
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            # Segmentation: (H, W) int64 class labels
            "segmentation": torch.from_numpy(gt["segmentation"]).long(),
            # Orientation: (2, H, W) float32
            "orientation": torch.from_numpy(gt["orientation"]).permute(2, 0, 1).float(),
            # Junction heatmap: (1, H, W) float32
            "junction_heatmap": torch.from_numpy(gt["junction_heatmap"]).unsqueeze(0).float(),
            # Edge distance: (1, H, W) float32
            "edge_distance": torch.from_numpy(gt["edge_distance"]).unsqueeze(0).float(),
        }

        # Include graph data for potential GNN usage
        sample["graph"] = {
            "vertices": torch.from_numpy(gt["vertices"]).float(),
            "edges": torch.from_numpy(gt["edges"]).long(),
            "assignments": torch.from_numpy(gt["assignments"]).long(),
        }

        # Metadata
        sample["meta"] = {
            "filename": fold_path.stem,
            "fold_path": str(fold_path),
        }

        return sample

    def _load_image(self, fold_path: Path) -> np.ndarray:
        """Load or render the image for a FOLD file."""
        if self.image_dir:
            # Try to load pre-rendered image
            image_path = self.image_dir / f"{fold_path.stem}.png"
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                return np.array(img)

        # Render on the fly
        return self._render_fold(fold_path)

    def _render_fold(self, fold_path: Path) -> np.ndarray:
        """Render a FOLD file to an RGB image."""
        from .fold_parser import transform_coords

        cp = self.parser.parse(fold_path)
        vertices, _ = transform_coords(
            cp.vertices,
            image_size=self.image_size,
            padding=self.padding,
        )

        # Create white background
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        # Color mapping for assignments
        colors = {
            0: (255, 0, 0),  # M - Red
            1: (0, 0, 255),  # V - Blue
            2: (0, 0, 0),  # B - Black
            3: (128, 128, 128),  # U - Gray
        }

        import cv2

        for edge_idx, (v1_idx, v2_idx) in enumerate(cp.edges):
            v1 = vertices[v1_idx].astype(np.int32)
            v2 = vertices[v2_idx].astype(np.int32)
            assignment = cp.assignments[edge_idx]
            color = colors[int(assignment)]

            cv2.line(
                img,
                (int(v1[0]), int(v1[1])),
                (int(v2[0]), int(v2[1])),
                color,
                self.line_width,
            )

        return img


def create_dataloaders(
    fold_dir: str | Path,
    image_dir: Optional[str | Path] = None,
    batch_size: int = 4,
    num_workers: int = 8,
    image_size: int = 1024,
    augment: bool = True,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        fold_dir: Directory containing .fold files
        image_dir: Directory containing pre-rendered images
        batch_size: Batch size
        num_workers: Number of dataloader workers
        image_size: Image size
        augment: Whether to apply augmentations to training data
        **dataset_kwargs: Additional arguments for CreasePatternDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .transforms import get_train_transform, get_val_transform

    train_transform = get_train_transform(image_size) if augment else get_val_transform(image_size)
    val_transform = get_val_transform(image_size)

    train_dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_dir=image_dir,
        image_size=image_size,
        transform=train_transform,
        split="train",
        **dataset_kwargs,
    )

    val_dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_dir=image_dir,
        image_size=image_size,
        transform=val_transform,
        split="val",
        **dataset_kwargs,
    )

    test_dataset = CreasePatternDataset(
        fold_dir=fold_dir,
        image_dir=image_dir,
        image_size=image_size,
        transform=val_transform,
        split="test",
        **dataset_kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
