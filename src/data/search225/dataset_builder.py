"""Build a standard synthetic dataset root from ExplOri 22.5 tilings.

Output layout (mirrors ``treemaker_tree_v1`` / ``rabbit_ear_fold_program_v1`` so the
result drops straight into ``scripts/data/build_synthetic_training_mix.py``)::

    <out>/
      folds/<id>.fold
      metadata/<id>.json
      raw-manifest.jsonl
      recipe.json
      qa.json

Acquisition is a polite, sequential, stratified random sample across the populated
(N, sym) databases — not a full mirror — because the full site is ~2.5M patterns and
the server is single-flight. Weighting presets let you bias toward higher-N/denser
patterns (the harder cases for the detector).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

from src.data.fold_parser import FOLDParser

from .fetch import ALL_COMBOS, Combo, TilingClient, TilingNotFound

FAMILY = "search225-tiling"
DATASET_VERSION = "search225/v0.1.0"

DEFAULT_SPLITS = {"train": 0.85, "val": 0.10, "test": 0.05}

# Per-combo sampling weights for each preset. Keys are ``Combo.key`` ("N_sym").
# Combos absent from a preset get weight 0. Weights are normalized at runtime.
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    # Favor N=4,5,6 / denser patterns (the detector's hard cases).
    "higher-n": {
        "3_none": 0.03, "3_diag": 0.04, "3_book": 0.03,
        "4_none": 0.12, "4_diag": 0.12, "4_book": 0.11,
        "5_diag": 0.25, "5_book": 0.20,
        "6_book": 0.10,
    },
    # Favor N=2,3,4 / simpler patterns.
    "lower-n": {
        "2_none": 0.01, "2_diag": 0.01, "2_book": 0.01,
        "3_none": 0.13, "3_diag": 0.14, "3_book": 0.13,
        "4_none": 0.16, "4_diag": 0.16, "4_book": 0.15,
        "5_diag": 0.05, "5_book": 0.03, "6_book": 0.02,
    },
    # Roughly even across the useful (N>=3) combos.
    "balanced": {
        "3_none": 0.11, "3_diag": 0.11, "3_book": 0.11,
        "4_none": 0.11, "4_diag": 0.11, "4_book": 0.11,
        "5_diag": 0.12, "5_book": 0.11,
        "6_book": 0.11,
    },
}

# Bucket by total edge count. Aligns with the complexity buckets in the existing
# recipes (which counted creases); "tiny" captures the smallest N=2/3 patterns.
BUCKET_THRESHOLDS = [
    ("tiny", 0, 60),
    ("small", 60, 180),
    ("medium", 180, 500),
    ("dense", 500, 1200),
    ("superdense", 1200, 10 ** 9),
]


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


def allocate_counts(weights: dict[str, float], total: int) -> dict[str, int]:
    """Split ``total`` across combos proportionally to (normalized) weights."""
    active = {k: w for k, w in weights.items() if w > 0}
    weight_sum = sum(active.values())
    if weight_sum <= 0:
        raise ValueError("All sampling weights are zero")
    raw = {k: (w / weight_sum) * total for k, w in active.items()}
    floored = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floored.values())
    # Hand out the rounding remainder to the largest fractional parts.
    fractions = sorted(active, key=lambda k: raw[k] - floored[k], reverse=True)
    for k in fractions[:remainder]:
        floored[k] += 1
    return {k: v for k, v in floored.items() if v > 0}


@dataclass
class BuildSummary:
    out: str
    requested: int
    saved: int
    skipped_existing: int
    per_combo: dict[str, int] = field(default_factory=dict)
    max_ids: dict[str, int] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    assignment_totals: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "out": self.out,
            "requested": self.requested,
            "saved": self.saved,
            "skippedExisting": self.skipped_existing,
            "perCombo": self.per_combo,
            "maxIds": self.max_ids,
            "bucketCounts": self.bucket_counts,
            "assignmentTotals": self.assignment_totals,
            "datasetVersion": DATASET_VERSION,
        }


def _resolve_combos(combo_keys: list[str] | None) -> dict[str, Combo]:
    by_key = {c.key: c for c in ALL_COMBOS}
    if not combo_keys:
        return by_key
    selected: dict[str, Combo] = {}
    for key in combo_keys:
        if key not in by_key:
            raise ValueError(f"Unknown combo {key!r}; choices: {sorted(by_key)}")
        selected[key] = by_key[key]
    return selected


def build_search225_dataset(
    out: str | Path,
    *,
    target_count: int = 10_000,
    weights_preset: str = "higher-n",
    weights_override: dict[str, float] | None = None,
    combo_keys: list[str] | None = None,
    dataset_name: str = "search225_v1",
    delay: float = 0.2,
    seed: int = 1234,
    max_misses_factor: int = 40,
    resume: bool = True,
    client: TilingClient | None = None,
    progress: Callable[[str], None] = print,
) -> BuildSummary:
    """Sample tilings from the site and write a standard dataset root.

    Args:
        out: dataset root directory to create/update.
        target_count: total number of patterns to fetch across all combos.
        weights_preset: one of WEIGHT_PRESETS ("higher-n", "lower-n", "balanced").
        weights_override: explicit ``{combo_key: weight}`` map (wins over preset).
        combo_keys: restrict to these combos (e.g. ["4_diag", "5_book"]).
        delay: per-request politeness delay in seconds (sequential client).
        seed: RNG seed for reproducible id sampling.
        max_misses_factor: give up on a combo after this * its target consecutive
            missing-id draws (guards against over-estimated id ranges).
        resume: skip ids whose .fold already exists on disk.
    """
    out = Path(out).expanduser()
    folds_dir = out / "folds"
    metadata_dir = out / "metadata"
    qa_dir = out / "qa"
    for d in (folds_dir, metadata_dir, qa_dir):
        d.mkdir(parents=True, exist_ok=True)

    combos = _resolve_combos(combo_keys)
    weights = dict(weights_override or WEIGHT_PRESETS.get(weights_preset, {}))
    if not weights:
        raise ValueError(f"Unknown weights preset {weights_preset!r}")
    # Keep only weights for combos we actually have / selected.
    weights = {k: w for k, w in weights.items() if k in combos}
    per_combo_targets = allocate_counts(weights, target_count)

    client = client or TilingClient(delay=delay)
    parser = FOLDParser()
    rng = random.Random(seed)

    manifest_path = out / "raw-manifest.jsonl"
    rows: list[dict[str, Any]] = []
    existing_ids: set[str] = set()
    if resume and manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        existing_ids = {r["id"] for r in rows}
        progress(f"Resuming: {len(rows)} existing rows in manifest")

    summary = BuildSummary(out=str(out), requested=target_count, saved=0, skipped_existing=0)

    for combo_key, target in per_combo_targets.items():
        combo = combos[combo_key]
        progress(f"\n=== {combo_key}: target {target} ===")
        max_id = client.discover_max_id(combo)  # uses MAX_ID_HINTS, no requests
        summary.max_ids[combo_key] = max_id
        progress(f"  id sampling ceiling = {max_id}")

        seen_ids: set[int] = set()
        got = 0
        misses = 0
        miss_limit = max(200, target * max_misses_factor)
        t0 = time.monotonic()

        while got < target and misses < miss_limit:
            tiling_id = rng.randint(1, max_id)
            if tiling_id in seen_ids:
                continue
            seen_ids.add(tiling_id)

            sample_id = f"{dataset_name}-{combo.view_id(tiling_id)}"
            fold_path = folds_dir / f"{sample_id}.fold"
            if sample_id in existing_ids or fold_path.exists():
                summary.skipped_existing += 1
                got += 1
                continue

            try:
                fold = client.fetch_fold(tiling_id, combo)
            except TilingNotFound:
                misses += 1
                continue
            except Exception as exc:  # noqa: BLE001 - log and keep going
                progress(f"  warn: {sample_id}: {exc}")
                misses += 1
                continue

            if fold is None:
                misses += 1
                continue

            row = _write_sample(
                fold=fold,
                sample_id=sample_id,
                combo=combo,
                tiling_id=tiling_id,
                fold_path=fold_path,
                metadata_dir=metadata_dir,
                parser=parser,
            )
            if row is None:
                misses += 1
                continue

            rows.append(row)
            existing_ids.add(sample_id)
            summary.saved += 1
            summary.per_combo[combo_key] = summary.per_combo.get(combo_key, 0) + 1
            summary.bucket_counts[row["bucket"]] = summary.bucket_counts.get(row["bucket"], 0) + 1
            for assign, count in row["assignments"].items():
                summary.assignment_totals[assign] = summary.assignment_totals.get(assign, 0) + count
            got += 1
            misses = 0

            if summary.saved % 100 == 0:
                rate = summary.saved / max(1e-6, time.monotonic() - t0)
                progress(f"  saved {summary.saved} total ({rate:.1f}/s)")

        if misses >= miss_limit:
            progress(f"  stopped {combo_key} at {got}/{target} (hit miss limit; range likely exhausted)")

    # Assign global splits deterministically, then persist everything.
    for index, row in enumerate(rows):
        row["split"] = split_for_index(index, DEFAULT_SPLITS)

    _write_manifest(manifest_path, rows)
    _write_recipe(out, dataset_name, target_count, weights_preset, weights, per_combo_targets, seed)
    _write_qa(out, qa_dir, rows, summary)

    progress(
        f"\nDone. saved={summary.saved} skipped={summary.skipped_existing} "
        f"manifest_rows={len(rows)} -> {out}"
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
            "Exact 22.5-degree flat-foldable tiling from ExplOri 22.5 (SEARCH-22.5).",
            "M/V/B assignments are mathematically valid by construction; auxiliary/hinge "
            "creases are preserved as F.",
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
    target_count: int,
    weights_preset: str,
    weights: dict[str, float],
    per_combo_targets: dict[str, int],
    seed: int,
) -> None:
    recipe = {
        "name": dataset_name,
        "source": "search225",
        "sourceUrl": "https://225.designorigami.net",
        "adapterVersion": DATASET_VERSION,
        "seed": seed,
        "targetCount": target_count,
        "weightsPreset": weights_preset,
        "weights": weights,
        "perComboTargets": per_combo_targets,
        "splits": DEFAULT_SPLITS,
        "family": FAMILY,
        "buckets": [{"name": n, "minEdges": lo, "maxEdges": hi} for n, lo, hi in BUCKET_THRESHOLDS],
        "notes": [
            "Exact flat-foldable 22.5-degree tilings scraped politely (sequentially) from "
            "the public ExplOri 22.5 API and normalized into the standard dataset-root layout.",
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
        "maxIds": summary.max_ids,
    }
    payload = json.dumps(qa, indent=2, sort_keys=True) + "\n"
    (out / "qa.json").write_text(payload, encoding="utf-8")
    (qa_dir / "build-summary.json").write_text(payload, encoding="utf-8")


def _clean_counter(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items()) if k is not None}
