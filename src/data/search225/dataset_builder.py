"""Build a standard synthetic dataset root from local SEARCH-22.5 FOLD files.

SEARCH-22.5 / ExplOri 22.5 (https://225.designorigami.net) is a database of
mathematically exact, flat-foldable crease patterns on the 22.5-degree grid. The
per-(N, symmetry) SQLite databases (``tilings_{N}_{sym}.db``) were provided directly
by the project's author.

IMPORTANT: this pipeline is strictly offline. Do NOT fetch from the public site —
it is served from a single home machine. Acquisition is:

1. Convert the local ``.db`` files to FOLD with ``db_to_fold.py`` in the SEARCH-22.5
   repo (requires that repo's venv with the compiled ``math225_core`` extension)::

       cd ~/Documents/code/SEARCH-22.5
       .venv/bin/python db_to_fold.py <tilings_N_sym.db> --out <staging-dir>

2. Run :func:`build_search225_dataset_from_folds` (or the CLI wrapper
   ``scripts/data/build_search225_dataset.py``) over the staging dir(s) to produce
   the project's standard dataset-root layout::

    <out>/
      folds/<id>.fold
      metadata/<id>.json
      raw-manifest.jsonl
      recipe.json
      qa.json

The result drops straight into ``scripts/data/build_synthetic_training_mix.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Callable

from src.data.fold_parser import FOLDParser

FAMILY = "search225-tiling"
DATASET_VERSION = "search225/v0.2.0-local-db"

DEFAULT_SPLITS = {"train": 0.85, "val": 0.10, "test": 0.05}

SYM_CHARS = {"none": "n", "diag": "d", "book": "b"}

# db_to_fold.py output naming: search225_{N}{sym}{tiling_id}.fold
FOLD_NAME_RE = re.compile(r"^search225_(\d+)(none|diag|book)(\d+)$")

# Bucket by total edge count. Aligns with the complexity buckets in the existing
# recipes (which counted creases); "tiny" captures the smallest N=2/3 patterns.
BUCKET_THRESHOLDS = [
    ("tiny", 0, 60),
    ("small", 60, 180),
    ("medium", 180, 500),
    ("dense", 500, 1200),
    ("superdense", 1200, 10**9),
]


@dataclass(frozen=True)
class Combo:
    """A (grid size N, symmetry) database combination, e.g. N=3, sym='diag'."""

    N: int
    sym: str

    @property
    def key(self) -> str:
        return f"{self.N}_{self.sym}"

    @property
    def char(self) -> str:
        return SYM_CHARS.get(self.sym, self.sym[:1])

    def view_id(self, tiling_id: int) -> str:
        return f"{self.N}{self.char}{tiling_id}"


def bucket_for_edges(num_edges: int) -> str:
    for name, lo, hi in BUCKET_THRESHOLDS:
        if lo <= num_edges < hi:
            return name
    return "superdense"


def split_for_index(index: int, splits: dict[str, float]) -> str:
    """Deterministic golden-ratio split, matching build_synthetic_training_mix."""
    total = sum(splits.values())
    position = (index * 0.6180339887498949) % 1
    train_cutoff = splits["train"] / total
    val_cutoff = (splits["train"] + splits["val"]) / total
    if position < train_cutoff:
        return "train"
    if position < val_cutoff:
        return "val"
    return "test"


def parse_fold_name(path: Path) -> tuple[Combo, int] | None:
    match = FOLD_NAME_RE.match(path.stem)
    if match is None:
        return None
    return Combo(N=int(match.group(1)), sym=match.group(2)), int(match.group(3))


@dataclass
class BuildSummary:
    out: str
    scanned: int
    saved: int
    skipped_unparseable_name: int = 0
    skipped_invalid: int = 0
    per_combo: dict[str, int] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    assignment_totals: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "out": self.out,
            "scanned": self.scanned,
            "saved": self.saved,
            "skippedUnparseableName": self.skipped_unparseable_name,
            "skippedInvalid": self.skipped_invalid,
            "perCombo": self.per_combo,
            "bucketCounts": self.bucket_counts,
            "assignmentTotals": self.assignment_totals,
            "datasetVersion": DATASET_VERSION,
        }


def build_search225_dataset_from_folds(
    out: str | Path,
    fold_dirs: list[str | Path],
    *,
    dataset_name: str = "search225_v1",
    splits: dict[str, float] | None = None,
    progress: Callable[[str], None] = print,
) -> BuildSummary:
    """Ingest local db_to_fold output dirs into a standard dataset root.

    Args:
        out: dataset root directory to create.
        fold_dirs: directories containing ``search225_{N}{sym}{id}.fold`` files
            produced by the SEARCH-22.5 repo's ``db_to_fold.py``.
        dataset_name: sample id prefix (also recorded in recipe.json).
        splits: split fractions (default 85/10/5 train/val/test).
    """
    out = Path(out).expanduser()
    folds_dir = out / "folds"
    metadata_dir = out / "metadata"
    qa_dir = out / "qa"
    for d in (folds_dir, metadata_dir, qa_dir):
        d.mkdir(parents=True, exist_ok=True)
    splits = dict(splits or DEFAULT_SPLITS)

    sources: list[tuple[Combo, int, Path]] = []
    summary = BuildSummary(out=str(out), scanned=0, saved=0)
    for fold_dir in fold_dirs:
        fold_dir = Path(fold_dir).expanduser()
        for path in sorted(fold_dir.glob("*.fold")):
            summary.scanned += 1
            parsed = parse_fold_name(path)
            if parsed is None:
                summary.skipped_unparseable_name += 1
                progress(f"  warn: cannot parse combo/id from {path.name}; skipped")
                continue
            combo, tiling_id = parsed
            sources.append((combo, tiling_id, path))

    # Deterministic order regardless of input dir order; dedupe on (combo, id).
    sources.sort(key=lambda item: (item[0].N, item[0].sym, item[1]))
    seen: set[tuple[str, int]] = set()

    parser = FOLDParser()
    rows: list[dict[str, Any]] = []
    for combo, tiling_id, path in sources:
        key = (combo.key, tiling_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            fold = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            summary.skipped_invalid += 1
            progress(f"  warn: {path.name}: {exc}")
            continue
        sample_id = f"{dataset_name}-{combo.view_id(tiling_id)}"
        row = _write_sample(
            fold=fold,
            sample_id=sample_id,
            combo=combo,
            tiling_id=tiling_id,
            fold_path=folds_dir / f"{sample_id}.fold",
            metadata_dir=metadata_dir,
            parser=parser,
        )
        if row is None:
            summary.skipped_invalid += 1
            progress(f"  warn: {path.name}: invalid geometry; skipped")
            continue
        rows.append(row)
        summary.saved += 1
        summary.per_combo[combo.key] = summary.per_combo.get(combo.key, 0) + 1
        summary.bucket_counts[row["bucket"]] = summary.bucket_counts.get(row["bucket"], 0) + 1
        for assign, count in row["assignments"].items():
            summary.assignment_totals[assign] = summary.assignment_totals.get(assign, 0) + count
        if summary.saved % 500 == 0:
            progress(f"  saved {summary.saved}")

    for index, row in enumerate(rows):
        row["split"] = split_for_index(index, splits)

    _write_manifest(out / "raw-manifest.jsonl", rows)
    _write_recipe(out, dataset_name, fold_dirs, splits, summary)
    _write_qa(out, qa_dir, rows, summary)

    progress(
        f"\nDone. scanned={summary.scanned} saved={summary.saved} "
        f"skipped={summary.skipped_unparseable_name + summary.skipped_invalid} -> {out}"
    )
    return summary


def _write_sample(
    *,
    fold: dict,
    sample_id: str,
    combo: Combo,
    tiling_id: int,
    fold_path: Path,
    metadata_dir: Path,
    parser: FOLDParser,
) -> dict[str, Any] | None:
    """Validate a fold, write fold + metadata files, return the manifest row."""
    try:
        cp = parser.parse_dict(fold)
    except Exception:  # noqa: BLE001 - malformed geometry, skip
        return None
    if cp.num_vertices == 0 or cp.num_edges == 0:
        return None

    assignment_counts = Counter(fold["edges_assignment"])
    # A handful of db rows are the "empty tiling" of their symmetry class:
    # a bare border square with zero creases. Not a crease pattern; skip.
    if assignment_counts.get("M", 0) + assignment_counts.get("V", 0) == 0:
        return None
    assignments = {k: int(v) for k, v in sorted(assignment_counts.items())}
    num_edges = len(fold["edges_vertices"])
    num_vertices = len(fold["vertices_coords"])
    bucket = bucket_for_edges(num_edges)

    label_policy = {
        "assignmentSource": "search225-exact",
        "geometrySource": "search225-exact",
        "labelSource": "search225-exact",
        "trainingEligible": True,
        "notes": [
            "Exact 22.5-degree flat-foldable tiling from ExplOri 22.5 (SEARCH-22.5), "
            "reconstructed offline from author-provided tilings databases.",
            "M/V/B assignments are mathematically valid by construction; hinge "
            "creases with undetermined M/V are preserved as F.",
        ],
    }
    search225_metadata = {
        "adapterVersion": DATASET_VERSION,
        "N": combo.N,
        "symmetry": combo.sym,
        "tilingId": tiling_id,
        "viewId": combo.view_id(tiling_id),
        "viewUrl": fold.get("file_source"),
    }
    validation = {
        "valid": True,
        "passed": ["schema", "edge-geometry", "search225-exact"],
        "failed": [],
        "errors": [],
        "metrics": {"vertices": num_vertices, "edges": num_edges},
    }

    fold_path.write_text(json.dumps(fold, separators=(",", ":")), encoding="utf-8")

    metadata = {
        "id": sample_id,
        "config": {
            "id": sample_id,
            "family": FAMILY,
            "source": "search225",
            "N": combo.N,
            "symmetry": combo.sym,
            "tilingId": tiling_id,
        },
        "validation": validation,
        "fold": fold,
        "search225Metadata": search225_metadata,
        "labelPolicy": label_policy,
    }
    metadata_path = metadata_dir / f"{sample_id}.json"
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")

    return {
        "id": sample_id,
        "family": FAMILY,
        "bucket": bucket,
        "foldPath": f"folds/{fold_path.name}",
        "metadataPath": f"metadata/{metadata_path.name}",
        "vertices": num_vertices,
        "edges": num_edges,
        "assignments": assignments,
        "seed": tiling_id,
        "labelPolicy": label_policy,
        "search225Metadata": search225_metadata,
        "validation": validation,
        "split": "train",
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_recipe(
    out: Path,
    dataset_name: str,
    fold_dirs: list[str | Path],
    splits: dict[str, float],
    summary: BuildSummary,
) -> None:
    recipe = {
        "name": dataset_name,
        "source": "search225-local-db",
        "adapterVersion": DATASET_VERSION,
        "foldDirs": [str(Path(d).expanduser()) for d in fold_dirs],
        "splits": splits,
        "family": FAMILY,
        "buckets": [{"name": n, "minEdges": lo, "maxEdges": hi} for n, lo, hi in BUCKET_THRESHOLDS],
        "perCombo": summary.per_combo,
        "notes": [
            "Exact flat-foldable 22.5-degree tilings reconstructed OFFLINE from "
            "author-provided SEARCH-22.5 tilings databases via db_to_fold.py.",
            "Never fetch from the public site; it runs on the author's single home machine.",
        ],
    }
    (out / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")


def _write_qa(out: Path, qa_dir: Path, rows: list[dict[str, Any]], summary: BuildSummary) -> None:
    qa = {
        "accepted": len(rows),
        "datasetVersion": DATASET_VERSION,
        "familyCounts": _clean_counter(Counter(r.get("family") for r in rows)),
        "bucketCounts": _clean_counter(Counter(r.get("bucket") for r in rows)),
        "splitCounts": _clean_counter(Counter(r.get("split") for r in rows)),
        "assignmentTotals": summary.assignment_totals,
        "perCombo": summary.per_combo,
    }
    payload = json.dumps(qa, indent=2, sort_keys=True) + "\n"
    (out / "qa.json").write_text(payload, encoding="utf-8")
    (qa_dir / "build-summary.json").write_text(payload, encoding="utf-8")


def _clean_counter(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items()) if k is not None}
