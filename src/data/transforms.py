"""
Data augmentations for crease pattern detection.

Uses albumentations for geometry-aware augmentations that properly
transform both images and annotations.
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


def _random_non_mv_color() -> List[int]:
    """Generate a random color that isn't red, blue, or white.

    Used by assignment removal augmentation to simulate CPs without M/V colors.
    """
    colors = [
        [0, 0, 0],       # Black (most common for unassigned)
        [0, 128, 0],     # Green
        [128, 0, 128],   # Purple
        [255, 165, 0],   # Orange
        [0, 128, 128],   # Teal
        [139, 69, 19],   # Brown
    ]
    return colors[np.random.randint(len(colors))]


class DarkMode(ImageOnlyTransform):
    """
    Convert to dark mode - dark background with visible colored lines.

    - Inverts near-white background pixels to dark
    - Brightens dark blue lines to light blue for visibility
    - Keeps red lines as-is (already visible on dark bg)
    - Converts black boundary lines to white/light gray
    """

    def __init__(self, bg_threshold=240, dark_bg_value=30, p=0.5):
        super().__init__(p=p)
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


class GrayBackground(ImageOnlyTransform):
    """
    Convert white background to a random gray shade.

    This helps the model generalize to CPs with non-white backgrounds
    (common in scanned/photographed images).

    - Converts near-white background pixels to gray
    - Adjusts black border lines to remain visible on gray
    """

    def __init__(self, bg_threshold=240, gray_range=(120, 200), p=0.5):
        super().__init__(p=p)
        self.bg_threshold = bg_threshold
        self.gray_range = gray_range

    def apply(self, img, gray_value=160, **params):
        result = img.copy()

        # Find near-white pixels (background)
        is_background = np.all(img > self.bg_threshold, axis=2)

        # Set background to gray
        result[is_background] = gray_value

        return result

    def get_params(self):
        return {
            "gray_value": np.random.randint(self.gray_range[0], self.gray_range[1] + 1)
        }

    def get_transform_init_args_names(self):
        return ("bg_threshold", "gray_range")


class LineThicknessVariation(ImageOnlyTransform):
    """
    Randomly thicken or thin lines using morphological operations.

    This helps the model generalize to CPs with different line weights
    from various rendering sources or resolutions.
    """

    def __init__(self, max_kernel_size=3, p=0.5):
        super().__init__(p=p)
        self.max_kernel_size = max_kernel_size

    def apply(self, img, dilate=True, kernel_size=2, **params):
        import cv2

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Identify background pixels (near-white)
        is_bg = np.all(img > 240, axis=2)

        if dilate:
            # Dilate makes lines thicker (expands dark regions into white)
            # We invert, dilate, then invert back
            inverted = 255 - img
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            result = 255 - dilated
        else:
            # Erode makes lines thinner (shrinks dark regions)
            inverted = 255 - img
            eroded = cv2.erode(inverted, kernel, iterations=1)
            result = 255 - eroded

        # Preserve original background to avoid artifacts
        result[is_bg] = img[is_bg]

        return result

    def get_params(self):
        return {
            "dilate": np.random.random() > 0.5,
            "kernel_size": np.random.randint(2, self.max_kernel_size + 1),
        }

    def get_transform_init_args_names(self):
        return ("max_kernel_size",)


class TextOverlay(ImageOnlyTransform):
    """
    Randomly add text overlays to the image.

    Simulates real-world CPs that often have labels, titles, annotations,
    or watermarks on them.
    """

    def __init__(
        self,
        num_texts_range: Tuple[int, int] = (1, 3),
        font_scale_range: Tuple[float, float] = (0.4, 1.5),
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.num_texts_range = num_texts_range
        self.font_scale_range = font_scale_range

        # Common text that might appear on crease patterns
        self.text_options = [
            # Origami terms
            "fold", "crease", "mountain", "valley", "M", "V",
            "CP", "crease pattern", "origami",
            # Author/source labels
            "design:", "by:", "ver.", "v1", "v2", "2024", "2023",
            # Random letters/numbers (like diagram labels)
            "A", "B", "C", "D", "1", "2", "3", "4",
            "Fig.", "Step", "Part",
            # Scale/measurement
            "1:1", "scale", "cm", "mm",
        ]

    def apply(self, img, texts=None, positions=None, colors=None, scales=None, **params):
        import cv2

        result = img.copy()
        h, w = img.shape[:2]

        if texts is None:
            return result

        for text, pos, color, scale in zip(texts, positions, colors, scales):
            # Add text with cv2
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = max(1, int(scale))

            # Get text size to ensure it fits
            (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

            # Clamp position to keep text visible
            x = min(max(0, pos[0]), w - text_w - 5)
            y = min(max(text_h + 5, pos[1]), h - 5)

            cv2.putText(result, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

        return result

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        h, w = img.shape[:2]

        num_texts = np.random.randint(self.num_texts_range[0], self.num_texts_range[1] + 1)

        texts = []
        positions = []
        colors = []
        scales = []

        for _ in range(num_texts):
            # Random text
            texts.append(np.random.choice(self.text_options))

            # Random position (bias toward edges/corners)
            if np.random.random() > 0.5:
                # Edge/corner placement
                x = np.random.choice([
                    np.random.randint(10, 100),  # Left edge
                    np.random.randint(w - 150, w - 10),  # Right edge
                ])
                y = np.random.choice([
                    np.random.randint(20, 80),  # Top edge
                    np.random.randint(h - 60, h - 10),  # Bottom edge
                ])
            else:
                # Random placement
                x = np.random.randint(10, w - 50)
                y = np.random.randint(30, h - 10)

            positions.append((x, y))

            # Random color (black, gray, or dark colors)
            color_choice = np.random.random()
            if color_choice < 0.6:
                colors.append((0, 0, 0))  # Black
            elif color_choice < 0.8:
                gray = np.random.randint(60, 140)
                colors.append((gray, gray, gray))  # Gray
            else:
                # Random dark color
                colors.append((
                    np.random.randint(0, 100),
                    np.random.randint(0, 100),
                    np.random.randint(0, 100),
                ))

            # Random scale
            scales.append(
                np.random.uniform(self.font_scale_range[0], self.font_scale_range[1])
            )

        return {
            "texts": texts,
            "positions": positions,
            "colors": colors,
            "scales": scales,
        }

    def get_transform_init_args_names(self):
        return ("num_texts_range", "font_scale_range")


class HueShiftCreases(ImageOnlyTransform):
    """
    Randomly shift the hue of red and blue crease colors.

    This makes the model robust to variations like:
    - Slightly orange-ish red
    - Slightly purple-ish blue
    - Different shades of the base colors
    """

    def __init__(self, hue_shift_limit=15, p=0.5):
        super().__init__(p=p)
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
                # Geometric transforms - use fit_output to prevent clipping
                A.Affine(
                    scale=(0.98, 1.02),  # Reduced to prevent clipping
                    translate_percent=(-0.01, 0.01),  # Reduced to prevent clipping
                    rotate=(-3, 3),  # Reduced rotation
                    fit_output=True,  # Scale result to fit original size
                    p=0.3,
                ),
                A.Perspective(
                    scale=(0.01, 0.03),  # Reduced perspective distortion
                    fit_output=True,  # Ensure full content stays visible
                    p=0.2,
                ),  # Perspective distortion (photos)

                # Color/lighting transforms
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=0.3,
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_limit=(1, 2),
                    shadow_dimension=5,
                    p=0.15,
                ),  # Uneven lighting (photos)

                # Noise/blur/compression transforms
                A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
                A.ImageCompression(quality_range=(70, 95), p=0.3),  # JPEG artifacts

                # Occlusion transforms
                A.CoarseDropout(
                    num_holes_range=(1, 3),
                    hole_height_range=(16, 48),
                    hole_width_range=(16, 48),
                    fill="random",
                    p=0.1,
                ),  # Simulates damaged/occluded areas

                # Custom color robustness augmentations
                HueShiftCreases(hue_shift_limit=15, p=0.3),  # Shift red/blue hues
                DarkMode(p=0.15),  # Dark mode simulation (dark bg, original line colors)
                GrayBackground(p=0.15),  # Gray background (helps border detection)
                LineThicknessVariation(max_kernel_size=3, p=0.2),  # Varying line weights
                TextOverlay(num_texts_range=(1, 3), p=0.15),  # Random text labels/annotations
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
        # Apply assignment removal FIRST (before other color transforms)
        # This ensures we detect original red/blue colors before DarkMode etc. modify them
        if self.strength in ("medium", "heavy"):
            image, segmentation = self._maybe_remove_assignments(
                image,
                segmentation,
                p=0.25,
            )

        # Convert vertices to keypoints format
        keypoints = [(float(v[0]), float(v[1])) for v in vertices]

        # Apply base transform (includes DarkMode, GrayBackground, etc.)
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

    def _maybe_remove_assignments(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        p: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Randomly convert M/V creases to unassigned.

        Simulates crease patterns that don't show mountain/valley fold assignments,
        which is common in many real-world CP images.

        Args:
            image: (H, W, 3) RGB image
            segmentation: (H, W) class labels (BG=0, M=1, V=2, B=3, U=4)
            p: probability of applying this augmentation

        Returns:
            Modified (image, segmentation) tuple
        """
        import cv2

        if np.random.random() > p:
            return image, segmentation

        # Choose mode: all_black (most common), all_random, or partial
        mode = np.random.choice(["all_black", "all_random", "partial"], p=[0.5, 0.2, 0.3])

        # Find M and V pixels from BOTH segmentation AND image colors
        # This catches anti-aliased edges that segmentation might miss
        is_mountain_seg = segmentation == 1  # CLASS_M
        is_valley_seg = segmentation == 2    # CLASS_V

        # Also detect by color (red = high R, low G/B; blue = high B, low R/G)
        is_red = (image[:, :, 0] > 200) & (image[:, :, 1] < 100) & (image[:, :, 2] < 100)
        is_blue = (image[:, :, 2] > 200) & (image[:, :, 0] < 100) & (image[:, :, 1] < 100)

        # Combine segmentation and color detection
        is_mountain = is_mountain_seg | is_red
        is_valley = is_valley_seg | is_blue

        # Dilate masks slightly to catch anti-aliased edges
        kernel = np.ones((3, 3), np.uint8)
        is_mountain = cv2.dilate(is_mountain.astype(np.uint8), kernel, iterations=1).astype(bool)
        is_valley = cv2.dilate(is_valley.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Don't overwrite background - only replace where there's already a crease
        is_any_crease = segmentation > 0
        is_mountain = is_mountain & is_any_crease
        is_valley = is_valley & is_any_crease

        result_img = image.copy()
        result_seg = segmentation.copy()

        if mode == "all_black":
            # Convert all M/V to black, mark as U
            result_img[is_mountain | is_valley] = [0, 0, 0]
            result_seg[is_mountain | is_valley] = 4  # CLASS_U

        elif mode == "all_random":
            # Convert all M/V to same random color
            color = _random_non_mv_color()
            result_img[is_mountain | is_valley] = color
            result_seg[is_mountain | is_valley] = 4

        elif mode == "partial":
            # Randomly choose to convert M only, V only, or both
            convert_m = np.random.random() > 0.5
            convert_v = np.random.random() > 0.5
            if not convert_m and not convert_v:
                convert_m = True  # At least one

            # 70% black, 30% random color
            color = [0, 0, 0] if np.random.random() > 0.3 else _random_non_mv_color()

            if convert_m:
                result_img[is_mountain] = color
                result_seg[is_mountain] = 4
            if convert_v:
                result_img[is_valley] = color
                result_seg[is_valley] = 4

        return result_img, result_seg


def get_train_transform(image_size: int = 1024, strength: str = "medium") -> Callable:
    """Get training transform."""
    return CreasePatternTransform(image_size=image_size, strength=strength)


def get_val_transform(image_size: int = 1024) -> Callable:
    """Get validation/test transform (no augmentation)."""
    return CreasePatternTransform(image_size=image_size, strength="none")
