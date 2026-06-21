"""Roadmap-native CPLineNet training data from real FOLD geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from src.data.cpline_augmentations import normalize_augment_profile, render_augmented_cpline_sample
from src.data.fold_parser import FOLDParser, transform_coords

WEIGHTED_FAMILY_SAMPLING_PRESETS: dict[str, tuple[tuple[str, str, float], ...]] = {
    # Preserve the historical dense-edge base mix (TreeMaker and Rabbit Ear at
    # equal weight inside the non-tessellation 85%) while adding 15%
    # tessellations split 80/20 orthogonal-BP-grid/Miura.
    "v3-tessellation-15pct": (
        ("family", "treemaker-tree", 0.425),
        ("family", "rabbit-ear-fold-program", 0.425),
        ("sourceDataset", "tessellation_orthogonal_bp_grid_v2_15pct", 0.12),
        ("sourceDataset", "tessellation_miura_ori_v2_15pct", 0.03),
    ),
}
SUPPORTED_FAMILY_SAMPLING = frozenset(
    {"natural", "balanced", *WEIGHTED_FAMILY_SAMPLING_PRESETS.keys()}
)


@dataclass(frozen=True)
class CplineSample:
    image: np.ndarray
    line_prob: np.ndarray
    angle: np.ndarray
    junction_heatmap: np.ndarray
    junction_offset: np.ndarray
    junction_mask: np.ndarray
    assignment: np.ndarray
    pixel_vertices: np.ndarray
    edges: np.ndarray
    assignments: np.ndarray
    v2_non_crease_mask: np.ndarray
    v2_target_line_mask: np.ndarray
    v2_line_style: np.ndarray
    v2_observed_assignment: np.ndarray
    v2_boundary_contact_heatmap: np.ndarray
    v2_vertex_type: np.ndarray
    v2_boundary_side: np.ndarray
    v2_boundary_offset: np.ndarray
    v2_boundary_mask: np.ndarray
    v2_boundary_coord: np.ndarray
    metadata: dict[str, Any]


def load_manifest_records(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load CPLineNet raw-manifest JSONL rows."""
    path = Path(manifest_path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if "foldPath" not in record:
                raise ValueError(f"Expected raw-manifest row with foldPath at {path}:{line_number}")
            records.append(record)
    return records


def resolve_fold_path(record: dict[str, Any], manifest_path: str | Path) -> Path:
    path = Path(record["foldPath"])
    if path.is_absolute():
        return path
    return Path(manifest_path).parent / path


def select_records(
    records: list[dict[str, Any]],
    *,
    split: str,
    limit: int | None,
    max_edges: int | None,
    seed: int | None = None,
    family_sampling: str = "natural",
) -> list[dict[str, Any]]:
    if family_sampling not in SUPPORTED_FAMILY_SAMPLING:
        raise ValueError(f"Unsupported family_sampling={family_sampling!r}")
    filtered = [
        record
        for record in records
        if record.get("split") == split and (max_edges is None or int(record["edges"]) <= max_edges)
    ]
    if limit is None:
        return filtered
    if limit >= len(filtered) and family_sampling in {"natural", "balanced"}:
        return filtered
    if family_sampling == "balanced":
        return _sample_family_balanced(filtered, limit=limit, seed=seed)
    if family_sampling in WEIGHTED_FAMILY_SAMPLING_PRESETS:
        return _sample_weighted_groups(
            filtered,
            limit=limit,
            seed=seed,
            specs=WEIGHTED_FAMILY_SAMPLING_PRESETS[family_sampling],
        )
    rng = np.random.default_rng(0 if seed is None else seed)
    selected = rng.choice(len(filtered), size=limit, replace=False)
    return [filtered[int(index)] for index in selected]


def _sample_family_balanced(
    records: list[dict[str, Any]], *, limit: int, seed: int | None
) -> list[dict[str, Any]]:
    if not records or limit <= 0:
        return []
    rng = np.random.default_rng(0 if seed is None else seed)
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(str(record.get("family", "")), []).append(index)
    family_names = sorted(grouped)
    base = limit // len(family_names)
    remainder = limit % len(family_names)
    chosen: list[int] = []
    for family_offset, family_name in enumerate(family_names):
        quota = base + int(family_offset < remainder)
        family_indices = grouped[family_name]
        replace = quota > len(family_indices)
        sampled = rng.choice(family_indices, size=quota, replace=replace)
        chosen.extend(int(index) for index in sampled)
    rng.shuffle(chosen)
    return [records[index] for index in chosen]


def _sample_weighted_groups(
    records: list[dict[str, Any]],
    *,
    limit: int,
    seed: int | None,
    specs: tuple[tuple[str, str, float], ...],
) -> list[dict[str, Any]]:
    if not records or limit <= 0:
        return []
    total_weight = sum(weight for _, _, weight in specs)
    if total_weight <= 0:
        raise ValueError("Weighted family sampling requires a positive total weight")

    grouped: list[list[int]] = [[] for _ in specs]
    for index, record in enumerate(records):
        matches = [
            spec_index
            for spec_index, (field, expected, _) in enumerate(specs)
            if str(record.get(field, "")) == expected
        ]
        if len(matches) > 1:
            labels = ", ".join(f"{specs[match][0]}={specs[match][1]}" for match in matches)
            raise ValueError(
                f"Record {record.get('id', index)!r} matches multiple sampling groups: {labels}"
            )
        if matches:
            grouped[matches[0]].append(index)

    quotas = _weighted_quotas(limit=limit, weights=[weight for _, _, weight in specs])
    rng = np.random.default_rng(0 if seed is None else seed)
    chosen: list[int] = []
    for quota, indices, (field, expected, weight) in zip(quotas, grouped, specs, strict=True):
        if quota <= 0:
            continue
        if not indices:
            raise ValueError(
                f"Weighted family sampling requested {quota} records for {field}={expected!r} "
                f"(weight={weight:g}), but no filtered records matched"
            )
        replace = quota > len(indices)
        sampled = rng.choice(indices, size=quota, replace=replace)
        chosen.extend(int(index) for index in sampled)
    rng.shuffle(chosen)
    return [records[index] for index in chosen]


def _weighted_quotas(*, limit: int, weights: list[float]) -> list[int]:
    total = sum(weights)
    raw = [limit * weight / total for weight in weights]
    quotas = [int(np.floor(value)) for value in raw]
    remainder = limit - sum(quotas)
    order = sorted(
        range(len(weights)),
        key=lambda index: (raw[index] - quotas[index], weights[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    source_datasets: dict[str, int] = {}
    family_sources: dict[str, int] = {}
    for record in records:
        family = str(record.get("family", ""))
        source_dataset = str(record.get("sourceDataset", ""))
        families[family] = families.get(family, 0) + 1
        source_datasets[source_dataset] = source_datasets.get(source_dataset, 0) + 1
        family_source = f"{family}::{source_dataset}"
        family_sources[family_source] = family_sources.get(family_source, 0) + 1
    return {
        "families": dict(sorted(families.items())),
        "source_datasets": dict(sorted(source_datasets.items())),
        "family_sources": dict(sorted(family_sources.items())),
    }


class CplineFoldDataset(Dataset):
    """Dataset that renders CPLineNet inputs and dense targets from raw-manifest `.fold` rows."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        limit: int | None = None,
        max_edges: int | None = 250,
        image_size: int = 256,
        padding: int | None = None,
        line_width: int | None = None,
        augment_profile: str = "clean",
        render_noise: str | None = None,
        seed: int | None = None,
        family_sampling: str = "natural",
        junction_sigma_px: float | None = None,
        junction_offset_radius_px: float = 0.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.junction_sigma_px = junction_sigma_px
        self.junction_offset_radius_px = junction_offset_radius_px
        self.padding = padding if padding is not None else max(8, int(32 * image_size / 1024))
        self.line_width = (
            line_width if line_width is not None else max(1, int(2 * image_size / 768))
        )
        self.augment_profile = normalize_augment_profile(augment_profile, render_noise=render_noise)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._rng_worker_seed: int | None = None
        self.parser = FOLDParser()
        self._geometry_cache: dict[int, tuple[Any, np.ndarray]] = {}
        self.records = select_records(
            load_manifest_records(self.manifest_path),
            split=split,
            limit=limit,
            max_edges=max_edges,
            seed=seed,
            family_sampling=family_sampling,
        )
        if not self.records:
            raise ValueError(f"No records selected from {manifest_path} for split={split}")
        self.selection_summary = summarize_records(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cp, base_pixel_vertices = self._load_base_geometry(index)
        fold_path = resolve_fold_path(record, self.manifest_path)
        sample = render_cpline_sample(
            cp,
            image_size=self.image_size,
            padding=self.padding,
            line_width=self.line_width,
            augment_profile=self.augment_profile,
            rng=self._next_rng(),
            base_pixel_vertices=base_pixel_vertices,
            junction_sigma_px=self.junction_sigma_px,
            junction_offset_radius_px=self.junction_offset_radius_px,
        )
        item = {
            "image": torch.from_numpy(sample.image).permute(2, 0, 1).float() / 255.0,
            "line_prob": torch.from_numpy(sample.line_prob).unsqueeze(0).float(),
            "angle": torch.from_numpy(sample.angle).permute(2, 0, 1).float(),
            "junction_heatmap": torch.from_numpy(sample.junction_heatmap).unsqueeze(0).float(),
            "junction_offset": torch.from_numpy(sample.junction_offset).permute(2, 0, 1).float(),
            "junction_mask": torch.from_numpy(sample.junction_mask).bool(),
            "assignment": torch.from_numpy(sample.assignment).long(),
            "v2_non_crease_mask": torch.from_numpy(sample.v2_non_crease_mask).unsqueeze(0).float(),
            "v2_target_line_mask": torch.from_numpy(sample.v2_target_line_mask)
            .unsqueeze(0)
            .float(),
            "v2_line_style": torch.from_numpy(sample.v2_line_style).long(),
            "v2_observed_assignment": torch.from_numpy(sample.v2_observed_assignment).long(),
            "v2_boundary_contact_heatmap": torch.from_numpy(sample.v2_boundary_contact_heatmap)
            .unsqueeze(0)
            .float(),
            "v2_vertex_type": torch.from_numpy(sample.v2_vertex_type).long(),
            "v2_boundary_side": torch.from_numpy(sample.v2_boundary_side).long(),
            "v2_boundary_offset": torch.from_numpy(sample.v2_boundary_offset)
            .permute(2, 0, 1)
            .float(),
            "v2_boundary_mask": torch.from_numpy(sample.v2_boundary_mask).bool(),
            "v2_boundary_coord": torch.from_numpy(sample.v2_boundary_coord).unsqueeze(0).float(),
            "graph": {
                "vertices": torch.from_numpy(sample.pixel_vertices).float(),
                "edges": torch.from_numpy(sample.edges).long(),
                "assignments": torch.from_numpy(sample.assignments).long(),
            },
            "meta": {
                "id": str(record["id"]),
                "bucket": str(record.get("bucket", "")),
                "split": str(record.get("split", "")),
                "family": str(record.get("family", "")),
                "fold_path": str(fold_path),
                "edges": int(record.get("edges", len(sample.edges))),
                "augmentation": sample.metadata,
            },
        }
        return item

    def _load_base_geometry(self, index: int) -> tuple[Any, np.ndarray]:
        if index not in self._geometry_cache:
            record = self.records[index]
            cp = self.parser.parse(resolve_fold_path(record, self.manifest_path))
            pixel_vertices, _ = transform_coords(
                cp.vertices, image_size=self.image_size, padding=self.padding
            )
            self._geometry_cache[index] = (cp, pixel_vertices)
        return self._geometry_cache[index]

    def _next_rng(self) -> np.random.Generator:
        worker_info = get_worker_info()
        if worker_info is not None and self._rng_worker_seed != worker_info.seed:
            # Dataset instances are copied into DataLoader workers with identical
            # RNG state. Re-seed from PyTorch's per-worker seed so parallel
            # augmentation streams do not duplicate each other.
            self._rng = np.random.default_rng(int(worker_info.seed) % np.iinfo(np.uint32).max)
            self._rng_worker_seed = int(worker_info.seed)
        seed = int(self._rng.integers(0, np.iinfo(np.int32).max))
        return np.random.default_rng(seed)


def render_cpline_sample(
    cp,
    *,
    image_size: int,
    padding: int,
    line_width: int,
    augment_profile: str = "clean",
    render_noise: str | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    style_variant: str | None = None,
    square_symmetry: str | None = None,
    base_pixel_vertices: np.ndarray | None = None,
    junction_sigma_px: float | None = None,
    junction_offset_radius_px: float = 0.0,
) -> CplineSample:
    sample = render_augmented_cpline_sample(
        cp,
        image_size=image_size,
        padding=padding,
        line_width=line_width,
        augment_profile=augment_profile,
        render_noise=render_noise,
        seed=seed,
        rng=rng,
        style_variant=style_variant,
        square_symmetry=square_symmetry,
        base_pixel_vertices=base_pixel_vertices,
        junction_sigma_px=junction_sigma_px,
        junction_offset_radius_px=junction_offset_radius_px,
    )

    return CplineSample(
        image=sample.image,
        line_prob=sample.line_prob,
        angle=sample.angle,
        junction_heatmap=sample.junction_heatmap,
        junction_offset=sample.junction_offset,
        junction_mask=sample.junction_mask,
        assignment=sample.assignment,
        pixel_vertices=sample.pixel_vertices,
        edges=sample.edges,
        assignments=sample.assignments,
        v2_non_crease_mask=sample.v2_non_crease_mask,
        v2_target_line_mask=sample.v2_target_line_mask,
        v2_line_style=sample.v2_line_style,
        v2_observed_assignment=sample.v2_observed_assignment,
        v2_boundary_contact_heatmap=sample.v2_boundary_contact_heatmap,
        v2_vertex_type=sample.v2_vertex_type,
        v2_boundary_side=sample.v2_boundary_side,
        v2_boundary_offset=sample.v2_boundary_offset,
        v2_boundary_mask=sample.v2_boundary_mask,
        v2_boundary_coord=sample.v2_boundary_coord,
        metadata=sample.metadata,
    )


def render_input_image(
    cp,
    *,
    image_size: int,
    padding: int,
    line_width: int,
    render_noise: str = "clean",
) -> np.ndarray:
    """Compatibility wrapper for callers that still use render_noise."""
    profile = normalize_augment_profile(render_noise=render_noise)
    return render_augmented_cpline_sample(
        cp,
        image_size=image_size,
        padding=padding,
        line_width=line_width,
        augment_profile=profile,
    ).image


def cpline_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    stack_keys = [
        "image",
        "line_prob",
        "angle",
        "junction_heatmap",
        "junction_offset",
        "junction_mask",
        "assignment",
        "v2_non_crease_mask",
        "v2_target_line_mask",
        "v2_line_style",
        "v2_observed_assignment",
        "v2_boundary_contact_heatmap",
        "v2_vertex_type",
        "v2_boundary_side",
        "v2_boundary_offset",
        "v2_boundary_mask",
        "v2_boundary_coord",
    ]
    result = {key: torch.stack([item[key] for item in batch]) for key in stack_keys}
    result["graph"] = [item["graph"] for item in batch]
    result["meta"] = [item["meta"] for item in batch]
    return result
