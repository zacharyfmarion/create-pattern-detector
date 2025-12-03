"""
Data augmentations for crease pattern detection.

Uses albumentations for geometry-aware augmentations that properly
transform both images and annotations.
"""

from typing import Dict, Any, Optional, Callable
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class DarkMode(ImageOnlyTransform):
    """
    Convert to dark mode - dark background with visible colored lines.

    - Inverts near-white background pixels to dark
    - Brightens dark blue lines to light blue for visibility
    - Keeps red lines as-is (already visible on dark bg)
    - Converts black boundary lines to white/light gray
    """

    def __init__(self, bg_threshold=240, dark_bg_value=30, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.bg_threshold = bg_threshold  # Pixels above this are considered background
        self.dark_bg_value = dark_bg_value  # What to set dark background to

    def apply(self, img, **params):
        result = img.copy().astype(np.int16)  # Use int16 to avoid overflow

        # Find near-white pixels (background)
        is_background = np.all(img > self.bg_threshold, axis=2)

        # Find blue pixels (high blue, low red/green)
        is_blue = (img[:, :, 2] > 200) & (img[:, :, 0] < 50) & (img[:, :, 1] < 50)

        # Find black pixels (boundary lines)
        is_black = np.all(img < 30, axis=2)

        # Find gray pixels (unassigned lines)
        is_gray = (
            (img[:, :, 0] > 100) & (img[:, :, 0] < 160) &
            (img[:, :, 1] > 100) & (img[:, :, 1] < 160) &
            (img[:, :, 2] > 100) & (img[:, :, 2] < 160)
        )

        # Set background to dark
        result[is_background] = self.dark_bg_value

        # Brighten blue to light blue (e.g., [100, 150, 255])
        result[is_blue] = [100, 150, 255]

        # Convert black to light gray for visibility
        result[is_black] = [200, 200, 200]

        # Brighten gray to lighter gray
        result[is_gray] = [180, 180, 180]

        return result.astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("bg_threshold", "dark_bg_value")


class HueShiftCreases(ImageOnlyTransform):
    """
    Randomly shift the hue of red and blue crease colors.

    This makes the model robust to variations like:
    - Slightly orange-ish red
    - Slightly purple-ish blue
    - Different shades of the base colors
    """

    def __init__(self, hue_shift_limit=15, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.hue_shift_limit = hue_shift_limit

    def apply(self, img, hue_shift=0, **params):
        import cv2

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)

        # Shift hue channel (wraps around at 180)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        # Convert back to RGB
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def get_params(self):
        return {
            "hue_shift": np.random.randint(-self.hue_shift_limit, self.hue_shift_limit + 1)
        }

    def get_transform_init_args_names(self):
        return ("hue_shift_limit",)


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
                A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
                # Color robustness augmentations
                HueShiftCreases(hue_shift_limit=15, p=0.3),  # Shift red/blue hues
                DarkMode(p=0.2),  # Dark mode simulation (dark bg, original line colors)
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
