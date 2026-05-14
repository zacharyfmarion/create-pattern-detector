"""Manifest helpers for synthetic crease-pattern datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional


AssignmentVisibility = Literal["visible", "hidden"]
LabelPolicy = Literal["mv_segmentation", "visible_creases_as_unassigned"]
Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SyntheticManifestRow:
    """One rendered training example derived from one canonical FOLD graph."""

    id: str
    graph_id: str
    seed: int
    family: str
    bucket: str
    split: Split
    fold_path: str
    image_path: str
    assignment_visibility: AssignmentVisibility
    render_style: str
    label_policy: LabelPolicy
    image_size: int
    padding: int
    line_width: int
    vertices: int
    edges: int
    assignments: Dict[str, int]
    role_counts: Dict[str, int]
    bp_metadata: Optional[Dict[str, Any]]
    density_metadata: Optional[Dict[str, Any]]
    design_tree: Optional[Dict[str, Any]]
    layout_metadata: Optional[Dict[str, Any]]
    molecule_metadata: Optional[Dict[str, Any]]
    realism_metadata: Optional[Dict[str, Any]]
    validation: Dict[str, Any]

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SyntheticManifestRow":
        return cls(
            id=str(row["id"]),
            graph_id=str(row["graph_id"]),
            seed=int(row["seed"]),
            family=str(row["family"]),
            bucket=str(row["bucket"]),
            split=str(row["split"]),  # type: ignore[arg-type]
            fold_path=str(row["fold_path"]),
            image_path=str(row["image_path"]),
            assignment_visibility=str(row["assignment_visibility"]),  # type: ignore[arg-type]
            render_style=str(row["render_style"]),
            label_policy=str(row["label_policy"]),  # type: ignore[arg-type]
            image_size=int(row["image_size"]),
            padding=int(row["padding"]),
            line_width=int(row["line_width"]),
            vertices=int(row["vertices"]),
            edges=int(row["edges"]),
            assignments={str(k): int(v) for k, v in dict(row["assignments"]).items()},
            role_counts={str(k): int(v) for k, v in dict(row.get("role_counts", row.get("roleCounts", {}))).items()},
            bp_metadata=dict(row["bp_metadata"] if "bp_metadata" in row else row["bpMetadata"])
            if row.get("bp_metadata") is not None or row.get("bpMetadata") is not None
            else None,
            density_metadata=dict(row["density_metadata"] if "density_metadata" in row else row["densityMetadata"])
            if row.get("density_metadata") is not None or row.get("densityMetadata") is not None
            else None,
            design_tree=dict(row["design_tree"] if "design_tree" in row else row["designTree"])
            if row.get("design_tree") is not None or row.get("designTree") is not None
            else None,
            layout_metadata=dict(row["layout_metadata"] if "layout_metadata" in row else row["layoutMetadata"])
            if row.get("layout_metadata") is not None or row.get("layoutMetadata") is not None
            else None,
            molecule_metadata=dict(row["molecule_metadata"] if "molecule_metadata" in row else row["moleculeMetadata"])
            if row.get("molecule_metadata") is not None or row.get("moleculeMetadata") is not None
            else None,
            realism_metadata=dict(row["realism_metadata"] if "realism_metadata" in row else row["realismMetadata"])
            if row.get("realism_metadata") is not None or row.get("realismMetadata") is not None
            else None,
            validation=dict(row["validation"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSON Lines, ignoring blank lines."""

    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write manifest rows as deterministic JSON Lines."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def load_manifest(path: str | Path, split: Optional[str] = None) -> List[SyntheticManifestRow]:
    rows = [SyntheticManifestRow.from_dict(row) for row in load_jsonl(path)]
    if split is not None:
        rows = [row for row in rows if row.split == split]
    return rows


def resolve_manifest_path(manifest_path: str | Path, relative_path: str | Path) -> Path:
    """Resolve a path stored relative to the manifest directory."""

    path = Path(relative_path)
    if path.is_absolute():
        return path
    return Path(manifest_path).parent / path
