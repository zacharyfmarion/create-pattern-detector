"""V2 square-crease-pattern label sidecars and visual QA overlays."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

ASSIGNMENT_NAMES = {0: "M", 1: "V", 2: "B", 3: "U"}
VERTEX_COLORS = {
    "corner": (40, 40, 40),
    "boundary_contact": (40, 160, 80),
    "interior_intersection": (255, 160, 0),
    "interior_endpoint": (180, 80, 240),
}
CARRIER_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
]


@dataclass(frozen=True)
class SquareFrame:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(self.x_max - self.x_min, 1.0)

    @property
    def height(self) -> float:
        return max(self.y_max - self.y_min, 1.0)


def build_v2_label_sidecar(
    *,
    sample_id: str,
    issue: str,
    image_size: int,
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    metadata: dict[str, Any] | None = None,
    oracle_mask: str | None = None,
    oracle_mask_kind: str | None = None,
) -> dict[str, Any]:
    """Build the Phase V2.1 graph/evidence sidecar from a rendered CP graph."""
    vertices = np.asarray(vertices, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64)
    assignments = np.asarray(assignments, dtype=np.int8)
    frame = infer_square_frame(vertices, edges, assignments, image_size)
    vertex_rows = describe_vertices(vertices, edges, frame)
    carriers, edge_to_carrier = describe_carriers(vertices, edges, assignments, frame)
    edge_rows = describe_edges(vertices, edges, assignments, edge_to_carrier, issue)
    spacing = spacing_stats(vertices, edges)
    assignment_constraints = describe_assignment_constraints(edges, assignments, issue)

    evidence: dict[str, Any] = {
        "line_style": line_style_for_issue(issue),
        "non_crease_regions": [],
        "artifact_mask": None,
        "target_line_mask": None,
        "ambiguous_mv": issue == "ambiguous_mv",
    }
    if oracle_mask is not None:
        if oracle_mask_kind == "non_crease_artifact":
            evidence["artifact_mask"] = oracle_mask
            evidence["non_crease_regions"].append(
                {
                    "kind": issue,
                    "mask": oracle_mask,
                    "training_target": "non_crease_line",
                }
            )
        else:
            evidence["target_line_mask"] = oracle_mask

    return {
        "schema": "cp-detector/v2-label-sidecar/v1",
        "id": sample_id,
        "issue": issue,
        "image_size": int(image_size),
        "square_frame": {
            "x_min": frame.x_min,
            "y_min": frame.y_min,
            "x_max": frame.x_max,
            "y_max": frame.y_max,
        },
        "vertices": vertex_rows,
        "edges": edge_rows,
        "carriers": carriers,
        "boundary": describe_boundary(vertex_rows, edges, assignments),
        "assignment_constraints": assignment_constraints,
        "render_evidence": evidence,
        "spacing": spacing,
        "metadata": metadata or {},
    }


def infer_square_frame(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    image_size: int,
) -> SquareFrame:
    border_indices = np.flatnonzero(assignments == 2)
    if len(border_indices) == 0:
        return SquareFrame(0.0, 0.0, float(image_size - 1), float(image_size - 1))
    border_vertex_indices = sorted({int(v) for idx in border_indices for v in edges[int(idx)]})
    border_vertices = vertices[border_vertex_indices]
    return SquareFrame(
        x_min=float(np.min(border_vertices[:, 0])),
        y_min=float(np.min(border_vertices[:, 1])),
        x_max=float(np.max(border_vertices[:, 0])),
        y_max=float(np.max(border_vertices[:, 1])),
    )


def describe_vertices(
    vertices: np.ndarray,
    edges: np.ndarray,
    frame: SquareFrame,
) -> list[dict[str, Any]]:
    degree = np.zeros(len(vertices), dtype=np.int32)
    for v1, v2 in edges:
        degree[int(v1)] += 1
        degree[int(v2)] += 1

    rows: list[dict[str, Any]] = []
    for idx, vertex in enumerate(vertices):
        boundary = boundary_position(vertex, frame)
        vertex_type = "interior_intersection"
        if boundary is not None:
            vertex_type = "corner" if boundary["is_corner"] else "boundary_contact"
        elif int(degree[idx]) <= 1:
            vertex_type = "interior_endpoint"
        rows.append(
            {
                "id": idx,
                "x": round(float(vertex[0]), 3),
                "y": round(float(vertex[1]), 3),
                "type": vertex_type,
                "degree": int(degree[idx]),
                "boundary_side": None if boundary is None else boundary["side"],
                "boundary_coordinate": None if boundary is None else boundary["coordinate"],
            }
        )
    return rows


def boundary_position(
    vertex: np.ndarray,
    frame: SquareFrame,
    tolerance_px: float = 2.5,
) -> dict[str, Any] | None:
    x, y = float(vertex[0]), float(vertex[1])
    corner_distances = {
        "top_left": np.hypot(x - frame.x_min, y - frame.y_min),
        "top_right": np.hypot(x - frame.x_max, y - frame.y_min),
        "bottom_right": np.hypot(x - frame.x_max, y - frame.y_max),
        "bottom_left": np.hypot(x - frame.x_min, y - frame.y_max),
    }
    nearest_corner, corner_distance = min(corner_distances.items(), key=lambda item: item[1])
    if float(corner_distance) <= tolerance_px:
        side = {
            "top_left": "top",
            "top_right": "right",
            "bottom_right": "bottom",
            "bottom_left": "left",
        }[nearest_corner]
        return {"side": side, "coordinate": 0.0 if side in {"top", "left"} else 1.0, "is_corner": True}

    candidates = [
        ("top", abs(y - frame.y_min), (x - frame.x_min) / frame.width),
        ("right", abs(x - frame.x_max), (y - frame.y_min) / frame.height),
        ("bottom", abs(y - frame.y_max), (x - frame.x_min) / frame.width),
        ("left", abs(x - frame.x_min), (y - frame.y_min) / frame.height),
    ]
    side, distance, coord = min(candidates, key=lambda item: item[1])
    if float(distance) <= tolerance_px:
        return {"side": side, "coordinate": round(float(np.clip(coord, 0.0, 1.0)), 6), "is_corner": False}
    return None


def describe_carriers(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    frame: SquareFrame,
    *,
    angle_tolerance_degrees: float = 2.0,
    rho_tolerance_px: float = 3.0,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    carriers: list[dict[str, Any]] = []
    edge_to_carrier: dict[int, int] = {}
    angle_tol = np.deg2rad(angle_tolerance_degrees)
    for edge_idx, (edge, assignment) in enumerate(zip(edges, assignments)):
        if int(assignment) == 2:
            continue
        p0 = vertices[int(edge[0])]
        p1 = vertices[int(edge[1])]
        length = float(np.linalg.norm(p1 - p0))
        if length <= 1e-6:
            continue
        params = line_parameters(p0, p1)
        matched_idx: int | None = None
        for carrier_idx, carrier in enumerate(carriers):
            if (
                angle_delta(params["theta"], float(carrier["theta"])) <= angle_tol
                and abs(params["rho"] - float(carrier["rho"])) <= rho_tolerance_px
            ):
                matched_idx = carrier_idx
                break
        if matched_idx is None:
            matched_idx = len(carriers)
            carriers.append(
                {
                    "id": matched_idx,
                    "theta": params["theta"],
                    "rho": params["rho"],
                    "t_min": params["t_min"],
                    "t_max": params["t_max"],
                    "edge_indices": [],
                    "vertex_indices": [],
                    "assignment_counts": {},
                    "boundary_intersections": [],
                    "carrier_intersections": [],
                }
            )
        carrier = carriers[matched_idx]
        carrier["t_min"] = min(float(carrier["t_min"]), params["t_min"])
        carrier["t_max"] = max(float(carrier["t_max"]), params["t_max"])
        carrier["edge_indices"].append(edge_idx)
        carrier["vertex_indices"].extend([int(edge[0]), int(edge[1])])
        assignment_name = ASSIGNMENT_NAMES.get(int(assignment), "U")
        counts = Counter(carrier["assignment_counts"])
        counts[assignment_name] += 1
        carrier["assignment_counts"] = dict(sorted(counts.items()))
        edge_to_carrier[edge_idx] = matched_idx

    for carrier in carriers:
        carrier["vertex_indices"] = sorted(set(int(v) for v in carrier["vertex_indices"]))
        carrier["boundary_intersections"] = carrier_boundary_intersections(carrier, frame)
        carrier["length_px"] = round(float(carrier["t_max"] - carrier["t_min"]), 3)

    for first_idx, first in enumerate(carriers):
        for second_idx, second in enumerate(carriers[first_idx + 1 :], start=first_idx + 1):
            intersection = carrier_intersection(first, second)
            if intersection is None:
                continue
            if frame.x_min - 1 <= intersection[0] <= frame.x_max + 1 and frame.y_min - 1 <= intersection[1] <= frame.y_max + 1:
                payload = {
                    "carrier_id": second_idx,
                    "x": round(float(intersection[0]), 3),
                    "y": round(float(intersection[1]), 3),
                }
                first["carrier_intersections"].append(payload)
                second["carrier_intersections"].append({**payload, "carrier_id": first_idx})

    return carriers, edge_to_carrier


def describe_edges(
    vertices: np.ndarray,
    edges: np.ndarray,
    assignments: np.ndarray,
    edge_to_carrier: dict[int, int],
    issue: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    style = line_style_for_issue(issue)
    ambiguous = issue == "ambiguous_mv"
    for edge_idx, (edge, assignment) in enumerate(zip(edges, assignments)):
        assignment_int = int(assignment)
        observed_assignment = "U" if ambiguous and assignment_int != 2 else ASSIGNMENT_NAMES.get(assignment_int, "U")
        rows.append(
            {
                "id": edge_idx,
                "vertices": [int(edge[0]), int(edge[1])],
                "type": "border" if assignment_int == 2 else "crease",
                "carrier_id": None if assignment_int == 2 else edge_to_carrier.get(edge_idx),
                "line_style": "solid" if assignment_int == 2 else style,
                "latent_assignment": ASSIGNMENT_NAMES.get(assignment_int, "U"),
                "observed_assignment": observed_assignment,
                "assignment_observation": "ambiguous" if ambiguous and assignment_int != 2 else "observed",
                "length_px": round(float(np.linalg.norm(vertices[int(edge[0])] - vertices[int(edge[1])])), 3),
            }
        )
    return rows


def describe_boundary(
    vertex_rows: list[dict[str, Any]],
    edges: np.ndarray,
    assignments: np.ndarray,
) -> dict[str, Any]:
    sides: dict[str, list[dict[str, Any]]] = {side: [] for side in ["top", "right", "bottom", "left"]}
    for row in vertex_rows:
        side = row.get("boundary_side")
        if side in sides:
            sides[str(side)].append(
                {
                    "vertex_id": row["id"],
                    "coordinate": row["boundary_coordinate"],
                    "type": row["type"],
                }
            )
    for side in sides:
        sides[side].sort(key=lambda item: float(item["coordinate"]))
    return {
        "sides": sides,
        "border_edge_indices": [int(idx) for idx, assignment in enumerate(assignments) if int(assignment) == 2],
        "border_edges": [
            [int(edges[idx][0]), int(edges[idx][1])]
            for idx, assignment in enumerate(assignments)
            if int(assignment) == 2
        ],
    }


def describe_assignment_constraints(
    edges: np.ndarray,
    assignments: np.ndarray,
    issue: str,
) -> dict[str, Any]:
    ambiguous = issue == "ambiguous_mv"
    observed: list[dict[str, Any]] = []
    for edge_idx, assignment in enumerate(assignments):
        assignment_int = int(assignment)
        observed_label = "U" if ambiguous and assignment_int != 2 else ASSIGNMENT_NAMES.get(assignment_int, "U")
        observed.append(
            {
                "edge_id": edge_idx,
                "vertices": [int(edges[edge_idx][0]), int(edges[edge_idx][1])],
                "latent_assignment": ASSIGNMENT_NAMES.get(assignment_int, "U"),
                "observed_assignment": observed_label,
                "source": "ambiguous_visual" if ambiguous and assignment_int != 2 else "rendered_visual",
            }
        )
    return {
        "policy": "preserve_B_mark_nonborder_U" if ambiguous else "observed_visual_labels",
        "edges": observed,
    }


def spacing_stats(vertices: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    lengths = [
        float(np.linalg.norm(vertices[int(v1)] - vertices[int(v2)]))
        for v1, v2 in edges
        if int(v1) != int(v2)
    ]
    if len(vertices) > 1:
        distances = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=2)
        distances[distances <= 1e-6] = np.inf
        closest = float(np.min(distances))
    else:
        closest = 0.0
    return {
        "min_edge_length_px": round(min(lengths), 3) if lengths else 0.0,
        "median_edge_length_px": round(float(np.median(lengths)), 3) if lengths else 0.0,
        "closest_vertex_spacing_px": round(closest, 3),
    }


def line_style_for_issue(issue: str) -> str:
    if issue == "dashed_line_support":
        return "dashed"
    if issue == "faint_low_contrast":
        return "faint_solid"
    if issue == "ambiguous_mv":
        return "monochrome_solid"
    return "solid"


def line_parameters(p0: np.ndarray, p1: np.ndarray) -> dict[str, float]:
    direction = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    theta = float(np.arctan2(direction[1], direction[0]) % np.pi)
    unit = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    normal = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    rho = float(np.dot(normal, p0))
    if rho < 0:
        rho = -rho
        theta = float((theta + np.pi) % np.pi)
        unit = -unit
    return {
        "theta": theta,
        "rho": rho,
        "t_min": min(float(np.dot(unit, p0)), float(np.dot(unit, p1))),
        "t_max": max(float(np.dot(unit, p0)), float(np.dot(unit, p1))),
    }


def carrier_boundary_intersections(
    carrier: dict[str, Any],
    frame: SquareFrame,
) -> list[dict[str, Any]]:
    theta = float(carrier["theta"])
    rho = float(carrier["rho"])
    normal = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    points: list[dict[str, Any]] = []
    sides = [
        ("top", frame.y_min),
        ("bottom", frame.y_max),
    ]
    for side, y in sides:
        if abs(normal[0]) <= 1e-9:
            continue
        x = (rho - normal[1] * y) / normal[0]
        if frame.x_min - 1e-3 <= x <= frame.x_max + 1e-3:
            points.append(
                {
                    "side": side,
                    "coordinate": round(float((x - frame.x_min) / frame.width), 6),
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                }
            )
    sides = [
        ("left", frame.x_min),
        ("right", frame.x_max),
    ]
    for side, x in sides:
        if abs(normal[1]) <= 1e-9:
            continue
        y = (rho - normal[0] * x) / normal[1]
        if frame.y_min - 1e-3 <= y <= frame.y_max + 1e-3:
            points.append(
                {
                    "side": side,
                    "coordinate": round(float((y - frame.y_min) / frame.height), 6),
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                }
            )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for point in points:
        key = (str(point["side"]), round(float(point["coordinate"]), 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


def carrier_intersection(
    first: dict[str, Any],
    second: dict[str, Any],
) -> np.ndarray | None:
    theta_a = float(first["theta"])
    theta_b = float(second["theta"])
    normal_a = np.array([-np.sin(theta_a), np.cos(theta_a)], dtype=np.float64)
    normal_b = np.array([-np.sin(theta_b), np.cos(theta_b)], dtype=np.float64)
    matrix = np.stack([normal_a, normal_b], axis=0)
    det = float(np.linalg.det(matrix))
    if abs(det) <= 1e-8:
        return None
    rhs = np.array([float(first["rho"]), float(second["rho"])], dtype=np.float64)
    return np.linalg.solve(matrix, rhs).astype(np.float32)


def angle_delta(a: float, b: float) -> float:
    delta = abs((a - b) % np.pi)
    return float(min(delta, np.pi - delta))


def draw_v2_label_overlay(
    image: np.ndarray,
    sidecar: dict[str, Any],
) -> np.ndarray:
    """Draw carriers, boundary-contact types, and carrier ids for visual QA."""
    canvas = np.asarray(image, dtype=np.uint8).copy()
    vertices = np.array([[v["x"], v["y"]] for v in sidecar["vertices"]], dtype=np.float32)

    for carrier in sidecar["carriers"]:
        color = CARRIER_COLORS[int(carrier["id"]) % len(CARRIER_COLORS)]
        for edge_id in carrier["edge_indices"]:
            edge = sidecar["edges"][int(edge_id)]
            v1, v2 = edge["vertices"]
            cv2.line(canvas, _point(vertices[v1]), _point(vertices[v2]), color, 2, cv2.LINE_AA)
        if carrier["vertex_indices"]:
            pts = vertices[np.array(carrier["vertex_indices"], dtype=np.int64)]
            center = np.mean(pts, axis=0)
            cv2.putText(
                canvas,
                f"c{carrier['id']}",
                _point(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    for edge in sidecar["edges"]:
        if edge["type"] != "border":
            continue
        v1, v2 = edge["vertices"]
        cv2.line(canvas, _point(vertices[v1]), _point(vertices[v2]), (15, 15, 15), 3, cv2.LINE_AA)

    for vertex in sidecar["vertices"]:
        color = VERTEX_COLORS.get(str(vertex["type"]), (0, 0, 0))
        radius = 5 if vertex["type"] in {"corner", "boundary_contact"} else 4
        cv2.circle(canvas, _point(np.array([vertex["x"], vertex["y"]], dtype=np.float32)), radius, color, -1, cv2.LINE_AA)

    return canvas


def _point(point: np.ndarray) -> tuple[int, int]:
    return (int(round(float(point[0]))), int(round(float(point[1]))))
