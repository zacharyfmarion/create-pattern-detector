"""
Data augmentations for crease pattern detection.

Uses albumentations for geometry-aware augmentations that properly
transform both images and annotations.
"""

from typing import Dict, Any, Optional, Callable
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class CreasePatternTransform:
    """
    Geometry-aware augmentations for crease patterns.

    Handles:
    - Image transformations
    - Segmentation mask transformations
    - Orientation field transformations (requires recomputation after rotation)
    - Junction heatmap transformations
    - Vertex coordinate transformations
    """

    def __init__(
        self,
        image_size: int = 1024,
        strength: str = "medium",
    ):
        """
        Initialize transforms.

        Args:
            image_size: Target image size
            strength: Augmentation strength ('light', 'medium', 'heavy')
        """
        self.image_size = image_size
        self.strength = strength
        self.transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        """Build the albumentations transform pipeline."""
        transforms = []

        # Geometric transforms (always applied for training)
        if self.strength != "none":
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])

        # Strength-dependent augmentations
        if self.strength == "heavy":
            transforms.extend([
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    p=0.5,
                ),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                    p=0.5,
                ),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            ])
        elif self.strength == "medium":
            transforms.extend([
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(-0.02, 0.02),
                    rotate=(-5, 5),
                    p=0.3,
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=0.3,
                ),
                A.GaussNoise(std_range=(5, 25), p=0.2),
            ])
        elif self.strength == "light":
            transforms.extend([
                A.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.02,
                    p=0.2,
                ),
            ])

        return A.Compose(
            transforms,
            additional_targets={
                "segmentation": "mask",
                "junction_heatmap": "mask",
            },
            keypoint_params=A.KeypointParams(
                format="xy",
                remove_invisible=False,
            ),
        )

    def __call__(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        orientation: np.ndarray,
        junction_heatmap: np.ndarray,
        vertices: np.ndarray,
        edges: np.ndarray,
        assignments: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Apply transforms to all data.

        Args:
            image: (H, W, 3) RGB image
            segmentation: (H, W) segmentation mask
            orientation: (H, W, 2) orientation field
            junction_heatmap: (H, W) junction heatmap
            vertices: (N, 2) vertex coordinates
            edges: (E, 2) edge indices
            assignments: (E,) edge assignments

        Returns:
            Dictionary with transformed data
        """
        # Convert vertices to keypoints format
        keypoints = [(float(v[0]), float(v[1])) for v in vertices]

        # Apply base transform
        transformed = self.transform(
            image=image,
            segmentation=segmentation,
            junction_heatmap=junction_heatmap,
            keypoints=keypoints,
        )

        # Extract transformed data
        result = {
            "image": transformed["image"],
            "segmentation": transformed["segmentation"],
            "junction_heatmap": transformed["junction_heatmap"],
            "edges": edges,
            "assignments": assignments,
        }

        # Update vertices from transformed keypoints
        if transformed["keypoints"]:
            result["vertices"] = np.array(transformed["keypoints"], dtype=np.float32)
        else:
            result["vertices"] = vertices

        # Recompute orientation field from transformed vertices
        result["orientation"] = self._recompute_orientation(
            result["vertices"],
            edges,
            self.image_size,
        )

        return result

    def _recompute_orientation(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        image_size: int,
    ) -> np.ndarray:
        """Recompute orientation field after geometric transform."""
        import cv2

        orientation = np.zeros((image_size, image_size, 2), dtype=np.float32)

        for v1_idx, v2_idx in edges:
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]

            # Compute edge direction
            direction = v2 - v1
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue

            cos_theta = direction[0] / length
            sin_theta = direction[1] / length

            # Create mask for the line
            mask = np.zeros((image_size, image_size), dtype=np.uint8)
            cv2.line(
                mask,
                (int(v1[0]), int(v1[1])),
                (int(v2[0]), int(v2[1])),
                1,
                3,  # line width
            )

            # Set orientation at masked pixels
            mask_bool = mask > 0
            orientation[mask_bool, 0] = cos_theta
            orientation[mask_bool, 1] = sin_theta

        return orientation


def get_train_transform(image_size: int = 1024, strength: str = "medium") -> Callable:
    """Get training transform."""
    return CreasePatternTransform(image_size=image_size, strength=strength)


def get_val_transform(image_size: int = 1024) -> Callable:
    """Get validation/test transform (no augmentation)."""
    return CreasePatternTransform(image_size=image_size, strength="none")
