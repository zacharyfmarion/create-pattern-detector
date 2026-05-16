"""Roadmap-native CPLineNet training data from real FOLD geometry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.cpline_augmentations import normalize_augment_profile, render_augmented_cpline_sample
from src.data.fold_parser import FOLDParser, transform_coords


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    metadata: dict[str, Any]


def load_manifest_records(manifest_path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return list(data["records"])


def resolve_fold_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def select_records(
    records: list[dict[str, Any]],
    *,
    split: str,
    train_count: int,
    val_count: int,
    max_edges: int | None,
) -> list[dict[str, Any]]:
    filtered = [record for record in records if max_edges is None or int(record["edges"]) <= max_edges]
    if split == "train":
        return filtered[:train_count]
    if split == "val":
        return filtered[train_count : train_count + val_count]
    raise ValueError(f"Unsupported split: {split}")


class CplineFoldDataset(Dataset):
    """Dataset that renders CPLineNet inputs and dense targets from real `.fold` files."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str,
        train_count: int = 8,
        val_count: int = 4,
        max_edges: int | None = 250,
        image_size: int = 256,
        padding: int | None = None,
        line_width: int | None = None,
        augment_profile: str = "clean",
        render_noise: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.padding = padding if padding is not None else max(8, int(32 * image_size / 1024))
        self.line_width = line_width if line_width is not None else max(1, int(2 * image_size / 768))
        self.augment_profile = normalize_augment_profile(augment_profile, render_noise=render_noise)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.parser = FOLDParser()
        self._geometry_cache: dict[int, tuple[Any, np.ndarray]] = {}
        self.records = select_records(
            load_manifest_records(self.manifest_path),
            split=split,
            train_count=train_count,
            val_count=val_count,
            max_edges=max_edges,
        )
        if not self.records:
            raise ValueError(f"No records selected from {manifest_path} for split={split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cp, base_pixel_vertices = self._load_base_geometry(index)
        fold_path = resolve_fold_path(record["path"])
        sample = render_cpline_sample(
            cp,
            image_size=self.image_size,
            padding=self.padding,
            line_width=self.line_width,
            augment_profile=self.augment_profile,
            rng=self._next_rng(),
            base_pixel_vertices=base_pixel_vertices,
        )
        item = {
            "image": torch.from_numpy(sample.image).permute(2, 0, 1).float() / 255.0,
            "line_prob": torch.from_numpy(sample.line_prob).unsqueeze(0).float(),
            "angle": torch.from_numpy(sample.angle).permute(2, 0, 1).float(),
            "junction_heatmap": torch.from_numpy(sample.junction_heatmap).unsqueeze(0).float(),
            "junction_offset": torch.from_numpy(sample.junction_offset).permute(2, 0, 1).float(),
            "junction_mask": torch.from_numpy(sample.junction_mask).bool(),
            "assignment": torch.from_numpy(sample.assignment).long(),
            "graph": {
                "vertices": torch.from_numpy(sample.pixel_vertices).float(),
                "edges": torch.from_numpy(sample.edges).long(),
                "assignments": torch.from_numpy(sample.assignments).long(),
            },
            "meta": {
                "id": str(record["id"]),
                "bucket": str(record.get("bucket", "")),
                "fold_path": str(fold_path),
                "edges": int(record.get("edges", len(sample.edges))),
                "augmentation": sample.metadata,
            },
        }
        return item

    def _load_base_geometry(self, index: int) -> tuple[Any, np.ndarray]:
        if index not in self._geometry_cache:
            record = self.records[index]
            cp = self.parser.parse(resolve_fold_path(record["path"]))
            pixel_vertices, _ = transform_coords(cp.vertices, image_size=self.image_size, padding=self.padding)
            self._geometry_cache[index] = (cp, pixel_vertices)
        return self._geometry_cache[index]

    def _next_rng(self) -> np.random.Generator:
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
    base_pixel_vertices: np.ndarray | None = None,
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
        base_pixel_vertices=base_pixel_vertices,
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
    ]
    result = {key: torch.stack([item[key] for item in batch]) for key in stack_keys}
    result["graph"] = [item["graph"] for item in batch]
    result["meta"] = [item["meta"] for item in batch]
    return result
