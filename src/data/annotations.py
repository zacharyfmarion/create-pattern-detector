"""
Ground truth annotation generation for crease pattern detection.

Generates multi-channel ground truth maps from FOLD data:
- Segmentation: 5-class (BG, M, V, B, U)
- Orientation: per-pixel direction field (cos θ, sin θ)
- Junction heatmap: Gaussian at interior vertices
"""

from typing import Dict, Tuple
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

from .fold_parser import CreasePattern, transform_coords


class GroundTruthGenerator:
    """
    Generate multi-channel ground truth from FOLD/CreasePattern data.

    Produces:
    - segmentation: (H, W) int64 - class labels per pixel
    - orientation: (H, W, 2) float32 - (cos θ, sin θ) at crease pixels
    - junction_heatmap: (H, W) float32 - Gaussian at vertex locations
    - edge_distance: (H, W) float32 - distance to nearest crease
    """

    # Segmentation class indices
    CLASS_BG = 0  # Background
    CLASS_M = 1  # Mountain
    CLASS_V = 2  # Valley
    CLASS_B = 3  # Border
    CLASS_U = 4  # Unassigned

    def __init__(
        self,
        image_size: int = 1024,
        padding: int = 50,
        line_width: int = 2,
        junction_radius: float = 4.0,
        junction_sigma: float = 1.5,
        use_antialiasing: bool = True,
    ):
        """
        Initialize the ground truth generator.

        Args:
            image_size: Output image size in pixels
            padding: Padding around the pattern (must match rendering)
            line_width: Width of crease lines in pixels
            junction_radius: Radius for junction detection region
            junction_sigma: Sigma for Gaussian junction heatmap
            use_antialiasing: Whether to use anti-aliased line drawing
        """
        self.image_size = image_size
        self.padding = padding
        self.line_width = line_width
        self.junction_radius = junction_radius
        self.junction_sigma = junction_sigma
        self.use_antialiasing = use_antialiasing
        self.line_type = cv2.LINE_AA if use_antialiasing else cv2.LINE_8

    def generate(self, cp: CreasePattern) -> Dict[str, np.ndarray]:
        """
        Generate all ground truth maps from a CreasePattern.

        Args:
            cp: CreasePattern with vertices, edges, and assignments

        Returns:
            Dictionary with:
            - 'segmentation': (H, W) int64 - class labels
            - 'orientation': (H, W, 2) float32 - direction field
            - 'junction_heatmap': (H, W) float32 - vertex heatmap
            - 'edge_distance': (H, W) float32 - distance transform
            - 'vertices': (N, 2) float32 - transformed vertex coords
            - 'edges': (E, 2) int64 - edge indices
            - 'assignments': (E,) int8 - edge assignments
        """
        # Transform vertices to pixel coordinates
        pixel_vertices, _ = transform_coords(
            cp.vertices,
            image_size=self.image_size,
            padding=self.padding,
        )

        # Generate each ground truth channel
        segmentation = self._generate_segmentation(pixel_vertices, cp.edges, cp.assignments)
        orientation = self._generate_orientation(pixel_vertices, cp.edges, cp.assignments)
        junction_heatmap = self._generate_junctions(pixel_vertices, cp.edges, cp.assignments)
        edge_distance = self._generate_edge_distance(segmentation)
        junction_offset, junction_mask = self._generate_junction_offsets(
            pixel_vertices, cp.edges, cp.assignments
        )

        return {
            "segmentation": segmentation,
            "orientation": orientation,
            "junction_heatmap": junction_heatmap,
            "edge_distance": edge_distance,
            "junction_offset": junction_offset,
            "junction_mask": junction_mask,
            "vertices": pixel_vertices,
            "edges": cp.edges,
            "assignments": cp.assignments,
        }

    def generate_from_fold(self, fold_path: str) -> Dict[str, np.ndarray]:
        """Generate ground truth from a FOLD file path."""
        from .fold_parser import FOLDParser

        parser = FOLDParser()
        cp = parser.parse(fold_path)
        return self.generate(cp)

    def _generate_segmentation(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        assignments: np.ndarray,
    ) -> np.ndarray:
        """
        Generate 5-class segmentation map with optional anti-aliasing.

        Classes:
        0: Background
        1: Mountain (M)
        2: Valley (V)
        3: Border (B)
        4: Unassigned (U)

        For anti-aliasing, we draw each class to a separate float buffer,
        then take argmax to get the final class labels.
        """
        # Map assignment indices to segmentation classes
        # assignments: M=0, V=1, B=2, U=3
        # seg classes: BG=0, M=1, V=2, B=3, U=4
        assignment_to_class = {0: 1, 1: 2, 2: 3, 3: 4}

        if self.use_antialiasing:
            # Create per-class confidence buffers for soft segmentation
            # Class 0 (background) starts with full confidence
            class_buffers = [np.ones((self.image_size, self.image_size), dtype=np.float32)]
            for _ in range(4):  # Classes 1-4
                class_buffers.append(np.zeros((self.image_size, self.image_size), dtype=np.float32))

            for edge_idx, (v1_idx, v2_idx) in enumerate(edges):
                assignment = assignments[edge_idx]
                seg_class = assignment_to_class[int(assignment)]

                v1 = vertices[v1_idx]
                v2 = vertices[v2_idx]

                # Draw anti-aliased line to grayscale buffer
                line_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                cv2.line(
                    line_mask,
                    (int(round(v1[0])), int(round(v1[1]))),
                    (int(round(v2[0])), int(round(v2[1]))),
                    255,
                    self.line_width,
                    lineType=self.line_type,
                )

                # Convert to float and accumulate
                line_float = line_mask.astype(np.float32) / 255.0
                class_buffers[seg_class] = np.maximum(class_buffers[seg_class], line_float)
                # Reduce background confidence where we have lines
                class_buffers[0] = np.minimum(class_buffers[0], 1.0 - line_float)

            # Take argmax to get final class labels
            stacked = np.stack(class_buffers, axis=0)
            seg = np.argmax(stacked, axis=0).astype(np.int64)
        else:
            # Simple non-anti-aliased version
            seg = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

            for edge_idx, (v1_idx, v2_idx) in enumerate(edges):
                assignment = assignments[edge_idx]
                seg_class = assignment_to_class[int(assignment)]

                v1 = vertices[v1_idx].astype(np.int32)
                v2 = vertices[v2_idx].astype(np.int32)

                cv2.line(
                    seg,
                    (int(v1[0]), int(v1[1])),
                    (int(v2[0]), int(v2[1])),
                    seg_class,
                    self.line_width,
                )

            seg = seg.astype(np.int64)

        return seg

    def _generate_orientation(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        assignments: np.ndarray,
    ) -> np.ndarray:
        """
        Generate per-pixel orientation field (cos θ, sin θ).

        For each crease pixel, stores the direction of the crease line.
        Background pixels have (0, 0).
        """
        orientation = np.zeros((self.image_size, self.image_size, 2), dtype=np.float32)

        for edge_idx, (v1_idx, v2_idx) in enumerate(edges):
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]

            # Compute edge direction
            direction = v2 - v1
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue

            # Normalize to unit vector (cos θ, sin θ)
            cos_theta = direction[0] / length
            sin_theta = direction[1] / length

            # Draw orientation along the line
            self._draw_orientation_line(orientation, v1, v2, cos_theta, sin_theta)

        return orientation

    def _draw_orientation_line(
        self,
        orientation: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        cos_theta: float,
        sin_theta: float,
    ) -> None:
        """Draw orientation values along a line segment with optional anti-aliasing."""
        # Create a mask for the line
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        cv2.line(
            mask,
            (int(round(v1[0])), int(round(v1[1]))),
            (int(round(v2[0])), int(round(v2[1]))),
            255,
            self.line_width,
            lineType=self.line_type,
        )

        if self.use_antialiasing:
            # Use anti-aliased blending for smooth orientation transitions
            mask_float = mask.astype(np.float32) / 255.0
            # Blend orientation values based on mask intensity
            orientation[:, :, 0] = np.where(
                mask_float > orientation[:, :, 0] ** 2 + orientation[:, :, 1] ** 2,
                cos_theta,
                orientation[:, :, 0],
            )
            orientation[:, :, 1] = np.where(
                mask_float > orientation[:, :, 0] ** 2 + orientation[:, :, 1] ** 2,
                sin_theta,
                orientation[:, :, 1],
            )
        else:
            # Set orientation at masked pixels
            mask_bool = mask > 0
            orientation[mask_bool, 0] = cos_theta
            orientation[mask_bool, 1] = sin_theta

    def _generate_junctions(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        assignments: np.ndarray,
    ) -> np.ndarray:
        """
        Generate Gaussian heatmap for junction (vertex) locations.

        Includes:
        - Interior vertices with at least 2 incident crease edges (M or V)
        - Border vertices where crease edges meet the border
        - Border corners where 2+ border edges meet (paper corners)
        """
        heatmap = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        num_vertices = len(vertices)

        # Count incident creases per vertex (M or V only, not B or U)
        crease_degrees = np.zeros(num_vertices, dtype=np.int32)
        # Count incident border edges per vertex
        border_degrees = np.zeros(num_vertices, dtype=np.int32)

        for edge_idx, (v1_idx, v2_idx) in enumerate(edges):
            assignment = assignments[edge_idx]
            if assignment in (0, 1):  # M or V
                crease_degrees[v1_idx] += 1
                crease_degrees[v2_idx] += 1
            elif assignment == 2:  # Border
                border_degrees[v1_idx] += 1
                border_degrees[v2_idx] += 1

        # Check which vertices are on the border
        is_border_vertex = border_degrees > 0

        # Add Gaussian at vertices that are junctions
        for v_idx, coords in enumerate(vertices):
            if is_border_vertex[v_idx]:
                # Border vertex: include if:
                # - It has at least 1 crease edge (crease meets boundary), OR
                # - It has at least 2 border edges (paper corner)
                if crease_degrees[v_idx] >= 1 or border_degrees[v_idx] >= 2:
                    self._add_gaussian(heatmap, coords)
            else:
                # Interior vertex: include if it has at least 2 crease edges
                if crease_degrees[v_idx] >= 2:
                    self._add_gaussian(heatmap, coords)

        return heatmap

    def _add_gaussian(self, heatmap: np.ndarray, center: np.ndarray) -> None:
        """Add a 2D Gaussian at the center location."""
        x, y = center
        H, W = heatmap.shape

        # Compute bounds for the Gaussian patch
        size = int(self.junction_radius * 3)
        x_min = max(0, int(x) - size)
        x_max = min(W, int(x) + size + 1)
        y_min = max(0, int(y) - size)
        y_max = min(H, int(y) + size + 1)

        if x_max <= x_min or y_max <= y_min:
            return

        # Create coordinate grids
        xx = np.arange(x_min, x_max)
        yy = np.arange(y_min, y_max)
        xx, yy = np.meshgrid(xx, yy)

        # Compute Gaussian
        sigma = self.junction_sigma
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))

        # Use max to handle overlapping junctions
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(
            heatmap[y_min:y_max, x_min:x_max],
            gaussian.astype(np.float32),
        )

    def _generate_edge_distance(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Generate distance transform from crease edges.

        Returns distance to the nearest crease pixel for each background pixel.
        """
        # Create binary mask of crease pixels (any non-background)
        crease_mask = segmentation > 0

        # Distance transform (distance from background to nearest crease)
        distance = distance_transform_edt(~crease_mask)

        return distance.astype(np.float32)

    def _generate_junction_offsets(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        assignments: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sub-pixel offset ground truth for junctions.

        For each junction vertex, computes the offset from the anchor pixel
        (rounded integer coords) to the true sub-pixel junction location.

        Returns:
            junction_offset: (H, W, 2) float32 - (dx, dy) offset from pixel center
            junction_mask: (H, W) bool - where offset loss should be applied
        """
        offset = np.zeros((self.image_size, self.image_size, 2), dtype=np.float32)
        mask = np.zeros((self.image_size, self.image_size), dtype=bool)

        num_vertices = len(vertices)

        # Count incident creases per vertex (M or V only, not B or U)
        crease_degrees = np.zeros(num_vertices, dtype=np.int32)
        # Count incident border edges per vertex
        border_degrees = np.zeros(num_vertices, dtype=np.int32)

        for edge_idx, (v1_idx, v2_idx) in enumerate(edges):
            assignment = assignments[edge_idx]
            if assignment in (0, 1):  # M or V
                crease_degrees[v1_idx] += 1
                crease_degrees[v2_idx] += 1
            elif assignment == 2:  # Border
                border_degrees[v1_idx] += 1
                border_degrees[v2_idx] += 1

        # Check which vertices are on the border
        is_border_vertex = border_degrees > 0

        # Add offset for vertices that are junctions (same logic as _generate_junctions)
        for v_idx, (x, y) in enumerate(vertices):
            is_junction = False

            if is_border_vertex[v_idx]:
                # Border vertex: include if it has crease edges or 2+ border edges
                if crease_degrees[v_idx] >= 1 or border_degrees[v_idx] >= 2:
                    is_junction = True
            else:
                # Interior vertex: include if it has at least 2 crease edges
                if crease_degrees[v_idx] >= 2:
                    is_junction = True

            if not is_junction:
                continue

            # Anchor pixel (integer coords using round)
            j = int(round(x))  # column
            i = int(round(y))  # row

            if not (0 <= i < self.image_size and 0 <= j < self.image_size):
                continue

            # Sub-pixel offset from pixel center to true junction
            # Offset in range [-0.5, 0.5]
            dx = x - j  # true_x - pixel_x
            dy = y - i  # true_y - pixel_y

            offset[i, j, 0] = dx
            offset[i, j, 1] = dy
            mask[i, j] = True

        return offset, mask


def visualize_ground_truth(
    gt: Dict[str, np.ndarray],
    output_path: str = None,
) -> np.ndarray:
    """
    Create a visualization of ground truth annotations.

    Args:
        gt: Dictionary from GroundTruthGenerator.generate()
        output_path: Optional path to save visualization

    Returns:
        (H, W, 3) RGB visualization image
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Segmentation with custom colormap
    seg_colors = [
        [1, 1, 1],  # Background - white
        [1, 0, 0],  # Mountain - red
        [0, 0, 1],  # Valley - blue
        [0, 0, 0],  # Border - black
        [0.5, 0.5, 0.5],  # Unassigned - gray
    ]
    seg_cmap = ListedColormap(seg_colors)
    axes[0, 0].imshow(gt["segmentation"], cmap=seg_cmap, vmin=0, vmax=4)
    axes[0, 0].set_title("Segmentation")
    axes[0, 0].axis("off")

    # Orientation (as HSV color wheel)
    orient = gt["orientation"]
    angle = np.arctan2(orient[:, :, 1], orient[:, :, 0])
    magnitude = np.linalg.norm(orient, axis=2)
    hsv = np.zeros((*orient.shape[:2], 3), dtype=np.float32)
    hsv[:, :, 0] = (angle + np.pi) / (2 * np.pi)  # Hue from angle
    hsv[:, :, 1] = magnitude  # Saturation from magnitude
    hsv[:, :, 2] = magnitude  # Value from magnitude
    rgb = plt.cm.hsv(hsv[:, :, 0])[:, :, :3] * magnitude[:, :, np.newaxis]
    axes[0, 1].imshow(rgb)
    axes[0, 1].set_title("Orientation Field")
    axes[0, 1].axis("off")

    # Junction heatmap
    axes[1, 0].imshow(gt["junction_heatmap"], cmap="hot")
    axes[1, 0].set_title("Junction Heatmap")
    axes[1, 0].axis("off")

    # Edge distance
    axes[1, 1].imshow(gt["edge_distance"], cmap="viridis")
    axes[1, 1].set_title("Edge Distance")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    # Convert to numpy array
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return vis
