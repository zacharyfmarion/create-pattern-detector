"""Manifest-based PyTorch dataset for rendered synthetic crease patterns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .manifest import SyntheticManifestRow, load_manifest, resolve_manifest_path

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - lets the module import before deps are installed.
    class Dataset:  # type: ignore[no-redef]
        pass


class SyntheticManifestDataset(Dataset):
    """Load rendered synthetic images and canonical FOLD graph labels by manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        transform: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.rows = load_manifest(self.manifest_path, split=split)
        self.transform = transform

        if not self.rows:
            raise ValueError(f"No rows found in {manifest_path} for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import numpy as np
        from PIL import Image
        try:
            import torch
        except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight smoke environments.
            torch = None  # type: ignore[assignment]

        from ..annotations import GroundTruthGenerator
        from ..fold_parser import CreasePattern, FOLDParser

        row = self.rows[idx]
        fold_path = resolve_manifest_path(self.manifest_path, row.fold_path)
        image_path = resolve_manifest_path(self.manifest_path, row.image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        parser = FOLDParser()
        canonical = parser.parse(fold_path)
        target_cp = self._target_pattern(canonical, row, CreasePattern)
        gt_generator = GroundTruthGenerator(
            image_size=row.image_size,
            padding=row.padding,
            line_width=row.line_width,
        )
        gt = gt_generator.generate(target_cp)

        if self.transform:
            transformed = self.transform(
                image=image,
                segmentation=gt["segmentation"],
                orientation=gt["orientation"],
                junction_heatmap=gt["junction_heatmap"],
                junction_offset=gt["junction_offset"],
                junction_mask=gt["junction_mask"],
                vertices=gt["vertices"],
                edges=gt["edges"],
                assignments=gt["assignments"],
            )
            image = transformed["image"]
            for key in (
                "segmentation",
                "orientation",
                "junction_heatmap",
                "junction_offset",
                "junction_mask",
                "vertices",
                "edges",
                "assignments",
            ):
                gt[key] = transformed[key]

        if torch is None:
            sample = {
                "image": np.transpose(image, (2, 0, 1)).astype("float32") / 255.0,
                "segmentation": gt["segmentation"].astype("int64"),
                "orientation": np.transpose(gt["orientation"], (2, 0, 1)).astype("float32"),
                "junction_heatmap": gt["junction_heatmap"][None, ...].astype("float32"),
                "edge_distance": gt["edge_distance"][None, ...].astype("float32"),
                "junction_offset": np.transpose(gt["junction_offset"], (2, 0, 1)).astype("float32"),
                "junction_mask": gt["junction_mask"].astype(bool),
                "graph": {
                    "vertices": gt["vertices"].astype("float32"),
                    "edges": canonical.edges.astype("int64"),
                    "assignments": canonical.assignments.astype("int64"),
                },
                "meta": row.to_dict(),
            }
            return sample

        sample = {
            "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
            "segmentation": torch.from_numpy(gt["segmentation"]).long(),
            "orientation": torch.from_numpy(gt["orientation"]).permute(2, 0, 1).float(),
            "junction_heatmap": torch.from_numpy(gt["junction_heatmap"]).unsqueeze(0).float(),
            "edge_distance": torch.from_numpy(gt["edge_distance"]).unsqueeze(0).float(),
            "junction_offset": torch.from_numpy(gt["junction_offset"]).permute(2, 0, 1).float(),
            "junction_mask": torch.from_numpy(gt["junction_mask"]).bool(),
            "graph": {
                "vertices": torch.from_numpy(gt["vertices"]).float(),
                "edges": torch.from_numpy(canonical.edges).long(),
                "assignments": torch.from_numpy(canonical.assignments).long(),
            },
            "meta": row.to_dict(),
        }
        return sample

    @staticmethod
    def _target_pattern(canonical: Any, row: SyntheticManifestRow, crease_pattern_cls: Any) -> Any:
        """Apply the pixel-label policy while preserving canonical graph labels elsewhere."""

        if row.label_policy != "visible_creases_as_unassigned":
            return canonical

        assignments = canonical.assignments.copy()
        assignments[(assignments == 0) | (assignments == 1)] = 3
        return crease_pattern_cls(
            vertices=canonical.vertices,
            edges=canonical.edges,
            assignments=assignments,
        )
