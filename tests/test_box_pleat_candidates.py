import importlib.util
import json
import math
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "find_box_pleat_candidates.py"
)
SPEC = importlib.util.spec_from_file_location("find_box_pleat_candidates", SCRIPT_PATH)
assert SPEC and SPEC.loader
bp_candidates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bp_candidates
SPEC.loader.exec_module(bp_candidates)


def test_rotated_grid_scores_as_box_pleat_candidate(tmp_path):
    fold_path = write_fold(tmp_path / "rotated-grid.fold", grid_fold(rotation_deg=27.0))

    record = bp_candidates.score_crease_pattern(
        bp_candidates.FOLDParser().parse(fold_path),
        path=fold_path,
        fold_root=tmp_path,
    )

    assert record.orthogonal_length_ratio == 1.0
    assert record.axis_balance > 0.9
    assert record.repeated_coord_count >= 6
    assert record.bp_score > 0.75


def test_radial_fan_scores_below_box_pleat_grid(tmp_path):
    grid_path = write_fold(tmp_path / "rotated-grid.fold", grid_fold(rotation_deg=13.0))
    fan_path = write_fold(tmp_path / "fan.fold", radial_fan_fold())

    parser = bp_candidates.FOLDParser()
    grid = bp_candidates.score_crease_pattern(
        parser.parse(grid_path), path=grid_path, fold_root=tmp_path
    )
    fan = bp_candidates.score_crease_pattern(
        parser.parse(fan_path), path=fan_path, fold_root=tmp_path
    )

    assert fan.orthogonal_length_ratio < 0.5
    assert fan.bp_score < grid.bp_score


def test_fingerprints_are_path_independent_for_same_fold(tmp_path):
    first_path = write_fold(tmp_path / "sample-a.fold", grid_fold(rotation_deg=0.0))
    second_path = write_fold(tmp_path / "sample-b.fold", grid_fold(rotation_deg=0.0))
    parser = bp_candidates.FOLDParser()

    first = bp_candidates.score_crease_pattern(
        parser.parse(first_path), path=first_path, fold_root=tmp_path
    )
    second = bp_candidates.score_crease_pattern(
        parser.parse(second_path), path=second_path, fold_root=tmp_path
    )
    first_fingerprints = bp_candidates.build_fingerprints(
        [first],
        errors=[],
        candidate_records=[first],
        candidate_tiers=bp_candidates.CANDIDATE_TIERS,
    )
    second_fingerprints = bp_candidates.build_fingerprints(
        [second],
        errors=[],
        candidate_records=[second],
        candidate_tiers=bp_candidates.CANDIDATE_TIERS,
    )

    assert first.canonical_fold_sha256 == second.canonical_fold_sha256
    assert first_fingerprints == second_fingerprints


def write_fold(path, fold):
    path.write_text(json.dumps(fold) + "\n", encoding="utf-8")
    return path


def grid_fold(rotation_deg=0.0):
    size = 4
    vertices = []
    index_by_coord = {}
    for y in range(size + 1):
        for x in range(size + 1):
            index_by_coord[(x, y)] = len(vertices)
            vertices.append(rotate((float(x), float(y)), rotation_deg))

    edges = []
    assignments = []
    for y in range(size + 1):
        for x in range(size):
            edges.append([index_by_coord[(x, y)], index_by_coord[(x + 1, y)]])
            assignments.append("B" if y in {0, size} else "U")
    for x in range(size + 1):
        for y in range(size):
            edges.append([index_by_coord[(x, y)], index_by_coord[(x, y + 1)]])
            assignments.append("B" if x in {0, size} else "U")

    return {
        "file_spec": 1.1,
        "file_creator": "test",
        "vertices_coords": vertices,
        "edges_vertices": edges,
        "edges_assignment": assignments,
    }


def radial_fan_fold():
    vertices = [[0.0, 0.0]]
    edges = []
    assignments = []
    for i in range(24):
        theta = 2.0 * math.pi * i / 24.0
        vertices.append([math.cos(theta), math.sin(theta)])
        edges.append([0, i + 1])
        assignments.append("U")
    return {
        "file_spec": 1.1,
        "file_creator": "test",
        "vertices_coords": vertices,
        "edges_vertices": edges,
        "edges_assignment": assignments,
    }


def rotate(point, degrees):
    theta = math.radians(degrees)
    x, y = point
    return [
        x * math.cos(theta) - y * math.sin(theta),
        x * math.sin(theta) + y * math.cos(theta),
    ]
