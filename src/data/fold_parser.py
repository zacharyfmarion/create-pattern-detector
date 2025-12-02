"""
FOLD format parser for crease patterns.

FOLD spec: https://github.com/edemaine/fold
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np


@dataclass
class CreasePattern:
    """
    Parsed crease pattern structure.

    Represents a crease pattern as a planar straight-line graph with:
    - Vertices at specific 2D coordinates
    - Edges connecting pairs of vertices
    - Assignment labels for each edge (M/V/B/U)
    """

    vertices: np.ndarray  # (N, 2) float32 - vertex coordinates
    edges: np.ndarray  # (E, 2) int64 - vertex index pairs
    assignments: np.ndarray  # (E,) int8 - edge assignments: M=0, V=1, B=2, U=3

    # Optional adjacency data (computed on demand)
    _vertex_degrees: Optional[np.ndarray] = field(default=None, repr=False)
    _vertex_edges: Optional[List[List[int]]] = field(default=None, repr=False)

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the pattern."""
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        """Number of edges in the pattern."""
        return len(self.edges)

    @property
    def num_creases(self) -> int:
        """Number of crease edges (M or V, excluding B and U)."""
        return int(np.sum((self.assignments == 0) | (self.assignments == 1)))

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y)."""
        if len(self.vertices) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        min_x, min_y = self.vertices.min(axis=0)
        max_x, max_y = self.vertices.max(axis=0)
        return (float(min_x), float(min_y), float(max_x), float(max_y))

    @property
    def vertex_degrees(self) -> np.ndarray:
        """Compute degree of each vertex (cached)."""
        if self._vertex_degrees is None:
            degrees = np.zeros(self.num_vertices, dtype=np.int32)
            for v1, v2 in self.edges:
                degrees[v1] += 1
                degrees[v2] += 1
            object.__setattr__(self, "_vertex_degrees", degrees)
        return self._vertex_degrees

    @property
    def interior_vertices(self) -> np.ndarray:
        """
        Get mask of interior vertices (not on border).

        A vertex is interior if it has no incident border edges.
        """
        is_interior = np.ones(self.num_vertices, dtype=bool)
        border_mask = self.assignments == 2  # B = 2

        for i, (v1, v2) in enumerate(self.edges):
            if border_mask[i]:
                is_interior[v1] = False
                is_interior[v2] = False

        return is_interior

    def get_incident_edges(self, vertex_idx: int) -> List[int]:
        """Get indices of edges incident to a vertex."""
        if self._vertex_edges is None:
            # Build vertex -> edges mapping
            v_edges: List[List[int]] = [[] for _ in range(self.num_vertices)]
            for e_idx, (v1, v2) in enumerate(self.edges):
                v_edges[v1].append(e_idx)
                v_edges[v2].append(e_idx)
            object.__setattr__(self, "_vertex_edges", v_edges)
        return self._vertex_edges[vertex_idx]

    def get_edge_direction(self, edge_idx: int) -> np.ndarray:
        """Get unit direction vector for an edge."""
        v1_idx, v2_idx = self.edges[edge_idx]
        v1, v2 = self.vertices[v1_idx], self.vertices[v2_idx]
        direction = v2 - v1
        length = np.linalg.norm(direction)
        if length < 1e-8:
            return np.array([1.0, 0.0], dtype=np.float32)
        return (direction / length).astype(np.float32)


class FOLDParser:
    """
    Parse FOLD JSON files into CreasePattern objects.

    Handles the FOLD file format specification, mapping edge assignments
    to our internal representation.
    """

    # Map FOLD assignment strings to integer labels
    ASSIGNMENT_MAP: Dict[str, int] = {
        "M": 0,  # Mountain
        "V": 1,  # Valley
        "B": 2,  # Border/Boundary
        "U": 3,  # Unassigned
        "F": 3,  # Flat (mapped to Unassigned)
        "C": 3,  # Cut (mapped to Unassigned)
    }

    # Reverse mapping for export
    ASSIGNMENT_LABELS: Dict[int, str] = {
        0: "M",
        1: "V",
        2: "B",
        3: "U",
    }

    def parse(self, fold_path: str | Path) -> CreasePattern:
        """
        Parse a FOLD file into a CreasePattern.

        Args:
            fold_path: Path to the .fold file

        Returns:
            CreasePattern with vertices, edges, and assignments
        """
        with open(fold_path, "r") as f:
            fold_data = json.load(f)

        return self.parse_dict(fold_data)

    def parse_dict(self, fold_data: Dict[str, Any]) -> CreasePattern:
        """
        Parse a FOLD dictionary into a CreasePattern.

        Args:
            fold_data: Parsed FOLD JSON data

        Returns:
            CreasePattern with vertices, edges, and assignments
        """
        # Handle nested structure (some files have fold data under 'fold' key)
        if "fold" in fold_data and "vertices_coords" in fold_data["fold"]:
            fold_data = fold_data["fold"]

        # Extract vertices
        vertices_raw = fold_data.get("vertices_coords", [])
        if len(vertices_raw) == 0:
            raise ValueError("FOLD file has no vertices_coords")

        vertices = np.array(vertices_raw, dtype=np.float32)

        # Extract edges
        edges_raw = fold_data.get("edges_vertices", [])
        if len(edges_raw) == 0:
            raise ValueError("FOLD file has no edges_vertices")

        edges = np.array(edges_raw, dtype=np.int64)

        # Extract assignments
        assignments_raw = fold_data.get("edges_assignment", [])
        if len(assignments_raw) == 0:
            # Default to unassigned if no assignments provided
            assignments_raw = ["U"] * len(edges)

        # Map string assignments to integers
        assignments = np.array(
            [self.ASSIGNMENT_MAP.get(a, 3) for a in assignments_raw],
            dtype=np.int8,
        )

        return CreasePattern(
            vertices=vertices,
            edges=edges,
            assignments=assignments,
        )

    def parse_batch(self, fold_paths: List[str | Path]) -> List[CreasePattern]:
        """Parse multiple FOLD files."""
        return [self.parse(p) for p in fold_paths]

    @staticmethod
    def to_fold_dict(cp: CreasePattern) -> Dict[str, Any]:
        """
        Convert a CreasePattern back to FOLD format dictionary.

        Args:
            cp: CreasePattern to convert

        Returns:
            Dictionary in FOLD format
        """
        return {
            "file_spec": 1.1,
            "file_creator": "cp-detector",
            "vertices_coords": cp.vertices.tolist(),
            "edges_vertices": cp.edges.tolist(),
            "edges_assignment": [
                FOLDParser.ASSIGNMENT_LABELS[int(a)] for a in cp.assignments
            ],
        }

    @staticmethod
    def save_fold(cp: CreasePattern, output_path: str | Path) -> None:
        """Save a CreasePattern to a FOLD file."""
        fold_dict = FOLDParser.to_fold_dict(cp)
        with open(output_path, "w") as f:
            json.dump(fold_dict, f, indent=2)


def transform_coords(
    vertices: np.ndarray,
    image_size: int = 1024,
    padding: int = 50,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Transform FOLD coords to padded image pixel coords.

    This matches the transform in the rendering script to ensure
    ground truth annotations align with rendered images.

    Args:
        vertices: (N, 2) array of vertex coordinates
        image_size: Target image size in pixels
        padding: Padding in pixels around the pattern

    Returns:
        Tuple of (transformed_vertices, transform_params)
    """
    vertices = np.array(vertices, dtype=np.float64)

    if len(vertices) == 0:
        return vertices.astype(np.float32), {}

    # Get bounds of the pattern
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)
    fold_width = max_x - min_x
    fold_height = max_y - min_y

    # Handle edge case of zero dimension
    if fold_width == 0:
        fold_width = 1
    if fold_height == 0:
        fold_height = 1

    # Available area after padding
    available = image_size - 2 * padding

    # Scale to fit, maintaining aspect ratio
    scale = available / max(fold_width, fold_height)

    # Center the pattern
    scaled_width = fold_width * scale
    scaled_height = fold_height * scale
    offset_x = padding + (available - scaled_width) / 2
    offset_y = padding + (available - scaled_height) / 2

    # Transform: translate to origin, scale, then offset to center with padding
    pixel_coords = np.zeros_like(vertices)
    pixel_coords[:, 0] = (vertices[:, 0] - min_x) * scale + offset_x
    pixel_coords[:, 1] = (vertices[:, 1] - min_y) * scale + offset_y

    transform_params = {
        "min_x": min_x,
        "min_y": min_y,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }

    return pixel_coords.astype(np.float32), transform_params
