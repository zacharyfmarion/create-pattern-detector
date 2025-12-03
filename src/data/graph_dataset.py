"""
Dataset for graph head training using ground truth graphs from FOLD files.

This dataset loads crease patterns with their GT vertex and edge structure,
and prepares them for training the graph neural network.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import Dataset
import random

from .fold_parser import FOLDParser, CreasePattern, transform_coords
from .annotations import GroundTruthGenerator


class GraphDataset(Dataset):
    """
    Dataset for training the graph head on ground truth graphs.

    Each sample contains:
    - Rendered crease pattern image (as segmentation visualization)
    - Ground truth vertices (transformed to pixel coords)
    - Ground truth edges with assignments
    - Candidate edges (GT edges + negative samples)
    """

    def __init__(
        self,
        fold_dir: str,
        image_size: int = 512,
        padding: int = 25,
        line_width: int = 2,
        negative_ratio: float = 1.0,
        augment_vertices: bool = True,
        vertex_noise_std: float = 2.0,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
    ):
        """
        Args:
            fold_dir: Directory containing .fold files
            image_size: Size of rendered images
            padding: Padding around crease pattern
            line_width: Line width for rendering
            negative_ratio: Ratio of negative to positive edges to sample
            augment_vertices: Whether to add noise to vertex positions
            vertex_noise_std: Standard deviation of vertex noise (pixels)
            split: 'train' or 'val'
            split_ratio: Ratio of files for training
            seed: Random seed for splitting
        """
        self.fold_dir = Path(fold_dir)
        self.image_size = image_size
        self.padding = padding
        self.line_width = line_width
        self.negative_ratio = negative_ratio
        self.augment_vertices = augment_vertices and split == "train"
        self.vertex_noise_std = vertex_noise_std
        self.split = split

        # Initialize parser and ground truth generator
        self.parser = FOLDParser()
        self.gt_generator = GroundTruthGenerator(
            image_size=image_size,
            padding=padding,
            line_width=line_width,
        )

        # Find all .fold files
        all_files = sorted(self.fold_dir.glob("*.fold"))
        if len(all_files) == 0:
            raise ValueError(f"No .fold files found in {fold_dir}")

        # Split into train/val
        random.seed(seed)
        indices = list(range(len(all_files)))
        random.shuffle(indices)

        split_idx = int(len(indices) * split_ratio)
        if split == "train":
            self.files = [all_files[i] for i in indices[:split_idx]]
        else:
            self.files = [all_files[i] for i in indices[split_idx:]]

        print(f"GraphDataset ({split}): {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fold_path = self.files[idx]

        # Parse FOLD file
        cp = self.parser.parse(fold_path)

        # Generate ground truth (including rendered image as segmentation)
        gt = self.gt_generator.generate(cp)

        # Create RGB image from segmentation for visualization/feature extraction
        # This simulates what the rendered image would look like
        image = self._segmentation_to_rgb(gt["segmentation"])

        # Get vertices in pixel coordinates
        vertices = gt["vertices"]
        edges = gt["edges"]
        assignments = gt["assignments"]

        # Optionally add noise to vertices (for robustness)
        if self.augment_vertices and len(vertices) > 0:
            noise = np.random.randn(*vertices.shape) * self.vertex_noise_std
            vertices = vertices + noise
            # Clamp to image bounds
            vertices = np.clip(vertices, 0, self.image_size - 1)

        # Create candidate edges (GT + negatives)
        edge_index, edge_labels, edge_assignments = self._create_candidate_edges(
            vertices, edges, assignments
        )

        # Convert image to tensor (C, H, W) normalized to [0, 1]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            'image': image_tensor,
            'vertices': torch.from_numpy(vertices).float(),
            'edge_index': torch.from_numpy(edge_index).long(),
            'edge_existence': torch.from_numpy(edge_labels).float(),
            'edge_assignment': torch.from_numpy(edge_assignments).long(),
            'num_gt_edges': len(edges),
            'meta': {
                'filename': fold_path.stem,
                'num_vertices': len(vertices),
            }
        }

    def _segmentation_to_rgb(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to RGB image."""
        h, w = segmentation.shape
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        # Color map: BG=white, M=red, V=blue, B=black, U=gray
        colors = {
            0: [255, 255, 255],  # BG
            1: [255, 0, 0],      # M
            2: [0, 0, 255],      # V
            3: [0, 0, 0],        # B
            4: [128, 128, 128],  # U
        }

        for class_id, color in colors.items():
            rgb[segmentation == class_id] = color

        return rgb

    def _create_candidate_edges(
        self,
        vertices: np.ndarray,
        gt_edges: np.ndarray,
        gt_assignments: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create candidate edge set with GT edges and negative samples.

        Returns:
            edge_index: (2, E_total) edge indices
            edge_labels: (E_total,) binary labels (1 for GT, 0 for negative)
            edge_assignments: (E_total,) assignments (valid only for GT edges)
        """
        n_vertices = len(vertices)
        n_gt_edges = len(gt_edges)

        if n_vertices < 2:
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        # Create set of GT edges for fast lookup
        gt_edge_set = set()
        for i, j in gt_edges:
            gt_edge_set.add((min(i, j), max(i, j)))

        # Sample negative edges
        n_negative = int(n_gt_edges * self.negative_ratio)
        negative_edges = []

        # Sample random vertex pairs that are not GT edges
        max_attempts = n_negative * 10
        attempts = 0
        while len(negative_edges) < n_negative and attempts < max_attempts:
            i = random.randint(0, n_vertices - 1)
            j = random.randint(0, n_vertices - 1)
            if i != j:
                edge_key = (min(i, j), max(i, j))
                if edge_key not in gt_edge_set:
                    negative_edges.append([i, j])
                    gt_edge_set.add(edge_key)  # Don't sample same pair twice
            attempts += 1

        negative_edges = np.array(negative_edges, dtype=np.int64) if negative_edges else np.empty((0, 2), dtype=np.int64)

        # Combine GT and negative edges
        if n_gt_edges > 0 and len(negative_edges) > 0:
            all_edges = np.vstack([gt_edges, negative_edges])
            edge_labels = np.concatenate([
                np.ones(n_gt_edges),
                np.zeros(len(negative_edges)),
            ])
            edge_assignments = np.concatenate([
                gt_assignments,
                np.zeros(len(negative_edges), dtype=np.int64),  # Dummy assignment for negatives
            ])
        elif n_gt_edges > 0:
            all_edges = gt_edges
            edge_labels = np.ones(n_gt_edges)
            edge_assignments = gt_assignments
        elif len(negative_edges) > 0:
            all_edges = negative_edges
            edge_labels = np.zeros(len(negative_edges))
            edge_assignments = np.zeros(len(negative_edges), dtype=np.int64)
        else:
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        # Convert to edge_index format (2, E)
        edge_index = all_edges.T

        return edge_index, edge_labels.astype(np.float32), edge_assignments

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """
        Custom collate function for variable-size graphs.

        Returns a list of individual samples rather than batching tensors,
        since each graph can have different numbers of vertices and edges.
        """
        return {
            'images': torch.stack([b['image'] for b in batch]),
            'vertices_list': [b['vertices'] for b in batch],
            'edge_index_list': [b['edge_index'] for b in batch],
            'edge_existence_list': [b['edge_existence'] for b in batch],
            'edge_assignment_list': [b['edge_assignment'] for b in batch],
            'meta': [b['meta'] for b in batch],
        }


class GraphDatasetWithFeatures(GraphDataset):
    """
    Graph dataset that also loads pre-computed image features.

    This is useful for training the graph head without running
    the backbone each time (faster training).
    """

    def __init__(
        self,
        fold_dir: str,
        features_dir: str,
        **kwargs,
    ):
        """
        Args:
            fold_dir: Directory containing .fold files
            features_dir: Directory containing pre-computed features (.pt files)
            **kwargs: Other arguments passed to GraphDataset
        """
        super().__init__(fold_dir, **kwargs)
        self.features_dir = Path(features_dir)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)

        # Load pre-computed features if available
        feature_path = self.features_dir / f"{sample['meta']['filename']}.pt"
        if feature_path.exists():
            features = torch.load(feature_path)
            sample['features'] = features
        else:
            # Will need to compute features on-the-fly
            sample['features'] = None

        return sample
