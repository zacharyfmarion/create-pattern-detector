"""Render canonical synthetic FOLD graphs into image variants and manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from math import ceil, sqrt
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .manifest import SyntheticManifestRow, write_jsonl


RGB = Tuple[int, int, int]

VISIBLE_COLORS: Dict[str, RGB] = {
    "M": (224, 48, 48),
    "V": (45, 91, 220),
    "B": (20, 20, 20),
    "U": (118, 118, 118),
    "F": (155, 155, 155),
    "C": (20, 20, 20),
}

BP_VISIBLE_COLORS: Dict[str, RGB] = {
    "border": (18, 18, 18),
    "ridge": (224, 48, 48),
    "hinge": (45, 91, 220),
    "axis": (45, 91, 220),
    "stretch": (45, 91, 220),
}

HIDDEN_STYLES: Dict[str, Dict[str, Any]] = {
    "monochrome_ink": {
        "background": (250, 250, 247),
        "crease": (34, 34, 34),
        "border": (12, 12, 12),
        "jitter": 2,
        "noise": 0.012,
    },
    "faint_blueprint": {
        "background": (244, 249, 250),
        "crease": (68, 104, 126),
        "border": (34, 55, 66),
        "jitter": 1,
        "noise": 0.018,
    },
}


def render_dataset(
    root: str | Path,
    raw_manifest_path: str | Path | None = None,
    recipe_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    contact_sheet_path: str | Path | None = None,
) -> List[SyntheticManifestRow]:
    """Render every requested style variant and write `manifest.jsonl`."""

    root = Path(root)
    raw_manifest_path = Path(raw_manifest_path) if raw_manifest_path else root / "raw-manifest.jsonl"
    recipe_path = Path(recipe_path) if recipe_path else root / "recipe.json"
    manifest_path = Path(manifest_path) if manifest_path else root / "manifest.jsonl"
    contact_sheet_path = Path(contact_sheet_path) if contact_sheet_path else root / "qa" / "contact_sheet.png"

    raw_rows = _load_jsonl(raw_manifest_path)
    recipe = _load_json(recipe_path)
    variants = _expand_variants(recipe.get("renderVariants", []))
    image_size = int(recipe.get("imageSize", 512))
    padding = int(recipe.get("padding", 32))

    if not raw_rows:
        raise ValueError(f"No rows found in {raw_manifest_path}")
    if not variants:
        raise ValueError("Recipe has no renderVariants")

    manifest_rows: List[SyntheticManifestRow] = []
    for raw_row in raw_rows:
        fold_path = root / raw_row["foldPath"]
        fold = _load_json(fold_path)
        for variant_index, variant in enumerate(variants):
            render_seed = int(raw_row["seed"]) + variant_index * 104729
            line_width = _line_width(variant["name"], image_size)
            image_rel = Path("images") / raw_row["split"] / f"{raw_row['id']}--{variant['name']}.png"
            image_path = root / image_rel
            render_fold(
                fold,
                image_path,
                image_size=image_size,
                padding=padding,
                line_width=line_width,
                style=variant["name"],
                assignment_visibility=variant["assignmentVisibility"],
                seed=render_seed,
            )
            manifest_rows.append(
                SyntheticManifestRow(
                    id=f"{raw_row['id']}--{variant['name']}",
                    graph_id=str(raw_row["id"]),
                    seed=render_seed,
                    family=str(raw_row["family"]),
                    bucket=str(raw_row["bucket"]),
                    split=raw_row["split"],
                    fold_path=str(raw_row["foldPath"]),
                    image_path=str(image_rel),
                    assignment_visibility=variant["assignmentVisibility"],
                    render_style=variant["name"],
                    label_policy=(
                        "mv_segmentation"
                        if variant["assignmentVisibility"] == "visible"
                        else "visible_creases_as_unassigned"
                    ),
                    image_size=image_size,
                    padding=padding,
                    line_width=line_width,
                    vertices=int(raw_row["vertices"]),
                    edges=int(raw_row["edges"]),
                    assignments=dict(raw_row["assignments"]),
                    role_counts=dict(raw_row.get("roleCounts", {})),
                    bp_metadata=dict(raw_row["bpMetadata"]) if raw_row.get("bpMetadata") is not None else None,
                    density_metadata=dict(raw_row["densityMetadata"]) if raw_row.get("densityMetadata") is not None else None,
                    design_tree=dict(raw_row["designTree"]) if raw_row.get("designTree") is not None else None,
                    layout_metadata=dict(raw_row["layoutMetadata"]) if raw_row.get("layoutMetadata") is not None else None,
                    molecule_metadata=dict(raw_row["moleculeMetadata"]) if raw_row.get("moleculeMetadata") is not None else None,
                    realism_metadata=dict(raw_row["realismMetadata"]) if raw_row.get("realismMetadata") is not None else None,
                    completion_metadata=dict(raw_row["completionMetadata"]) if raw_row.get("completionMetadata") is not None else None,
                    graph_label_policy=dict(raw_row["labelPolicy"]) if raw_row.get("labelPolicy") is not None else None,
                    bp_studio_summary=dict(raw_row["bpStudioSummary"]) if raw_row.get("bpStudioSummary") is not None else None,
                    validation=dict(raw_row["validation"]),
                )
            )

    write_jsonl(manifest_path, (row.to_dict() for row in manifest_rows))
    write_qa(root, manifest_rows)
    write_contact_sheet(root, manifest_rows, contact_sheet_path)
    return manifest_rows


def render_fold(
    fold: Mapping[str, Any],
    image_path: str | Path,
    image_size: int,
    padding: int,
    line_width: int,
    style: str,
    assignment_visibility: str,
    seed: int,
) -> None:
    """Render one FOLD graph to one RGB PNG."""

    from PIL import Image, ImageDraw, ImageFilter

    rng = random.Random(seed)
    config = HIDDEN_STYLES.get(style, {})
    background = tuple(config.get("background", (255, 255, 255)))
    scale = 3
    canvas_size = image_size * scale
    img = Image.new("RGB", (canvas_size, canvas_size), background)
    draw = ImageDraw.Draw(img)

    vertices = _transform_vertices(fold["vertices_coords"], image_size, padding)
    vertices = [(x * scale, y * scale) for x, y in vertices]
    assignments = list(fold.get("edges_assignment", ["U"] * len(fold["edges_vertices"])))
    roles = list(fold.get("edges_bpRole", []))
    width = max(1, line_width * scale)
    jitter = int(config.get("jitter", 0)) * scale

    for edge_index, (v1_index, v2_index) in enumerate(fold["edges_vertices"]):
        assignment = assignments[edge_index] if edge_index < len(assignments) else "U"
        role = roles[edge_index] if edge_index < len(roles) else None
        color = _edge_color(assignment, assignment_visibility, config, style=style, role=role)
        v1 = _jitter(vertices[v1_index], jitter, rng)
        v2 = _jitter(vertices[v2_index], jitter, rng)
        draw.line([v1, v2], fill=color, width=width)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.15 * scale))
    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    _add_noise(img, float(config.get("noise", 0.0)), rng)

    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(image_path)


def write_qa(root: Path, rows: Iterable[SyntheticManifestRow]) -> None:
    rows = list(rows)
    qa_path = root / "qa" / "render_qa.json"
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    by_split = Counter(row.split for row in rows)
    by_style = Counter(row.render_style for row in rows)
    by_family = Counter(row.family for row in rows)
    role_counts: Counter[str] = Counter()
    grid_sizes: Counter[str] = Counter()
    bp_subfamilies: Counter[str] = Counter()
    density_buckets: Counter[str] = Counter()
    dense_subfamilies: Counter[str] = Counter()
    archetypes: Counter[str] = Counter()
    realism_scores: List[float] = []
    empty_space_ratios: List[float] = []
    density_variances: List[float] = []
    solver_ms: List[float] = []
    faces: List[int] = []
    strict_passes = 0
    for row in rows:
        role_counts.update(row.role_counts)
        if row.bp_metadata:
            grid_sizes[str(row.bp_metadata.get("gridSize"))] += 1
            bp_subfamily = row.bp_metadata.get("bpSubfamily")
            if bp_subfamily:
                bp_subfamilies[str(bp_subfamily)] += 1
        if row.density_metadata:
            density_buckets[str(row.density_metadata.get("densityBucket"))] += 1
            dense_subfamily = row.density_metadata.get("subfamily")
            if dense_subfamily:
                dense_subfamilies[str(dense_subfamily)] += 1
        if row.design_tree and row.design_tree.get("archetype"):
            archetypes[str(row.design_tree["archetype"])] += 1
        if row.realism_metadata:
            if "score" in row.realism_metadata:
                realism_scores.append(float(row.realism_metadata["score"]))
            if "emptySpaceRatio" in row.realism_metadata:
                empty_space_ratios.append(float(row.realism_metadata["emptySpaceRatio"]))
            if "localDensityVariance" in row.realism_metadata:
                density_variances.append(float(row.realism_metadata["localDensityVariance"]))
        if "rabbit-ear-solver" in row.validation.get("passed", []):
            strict_passes += 1
        metrics = row.validation.get("metrics", {})
        if "solverMs" in metrics:
            solver_ms.append(float(metrics["solverMs"]))
        if "faces" in metrics:
            faces.append(int(metrics["faces"]))
    interior_roles = sum(count for role, count in role_counts.items() if role != "border")
    qa = {
        "rows": len(rows),
        "graphs": len({row.graph_id for row in rows}),
        "splits": dict(sorted(by_split.items())),
        "styles": dict(sorted(by_style.items())),
        "families": dict(sorted(by_family.items())),
        "labelPolicies": dict(sorted(Counter(row.label_policy for row in rows).items())),
        "roleCounts": dict(sorted(role_counts.items())),
        "diagonalRidgeRatio": role_counts.get("ridge", 0) / max(1, interior_roles),
        "gridSizes": dict(sorted(grid_sizes.items())),
        "bpSubfamilies": dict(sorted(bp_subfamilies.items())),
        "densityBuckets": dict(sorted(density_buckets.items())),
        "denseSubfamilies": dict(sorted(dense_subfamilies.items())),
        "archetypes": dict(sorted(archetypes.items())),
        "realismScore": _summarize(realism_scores),
        "emptySpaceRatio": _summarize(empty_space_ratios),
        "localDensityVariance": _summarize(density_variances),
        "solverMs": _summarize(solver_ms),
        "faces": _summarize(faces),
        "rabbitEarStrictPassRate": strict_passes / max(1, len(rows)),
    }
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_contact_sheet(
    root: Path,
    rows: Iterable[SyntheticManifestRow],
    output_path: str | Path,
    max_images: int = 64,
) -> None:
    """Write a compact visual QA sheet from rendered variants."""

    from PIL import Image, ImageDraw

    rows = list(rows)[:max_images]
    if not rows:
        return

    thumb_size = 160
    label_height = 28
    cols = min(8, max(1, ceil(sqrt(len(rows)))))
    rows_count = ceil(len(rows) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows_count * (thumb_size + label_height)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        x = (index % cols) * thumb_size
        y = (index // cols) * (thumb_size + label_height)
        img = Image.open(root / row.image_path).convert("RGB")
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        sheet.paste(img, (x + (thumb_size - img.width) // 2, y))
        label = f"{row.family} / {row.render_style}"
        draw.text((x + 4, y + thumb_size + 4), label[:28], fill=(20, 20, 20))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _expand_variants(variants: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []
    for variant in variants:
        count = int(variant.get("count", 1))
        for index in range(count):
            name = str(variant["name"])
            expanded.append(
                {
                    "name": name if count == 1 else f"{name}_{index:02d}",
                    "assignmentVisibility": str(variant.get("assignmentVisibility", "hidden")),
                }
            )
    return expanded


def _transform_vertices(vertices: List[List[float]], image_size: int, padding: int) -> List[Tuple[float, float]]:
    xs = [float(vertex[0]) for vertex in vertices]
    ys = [float(vertex[1]) for vertex in vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    available = image_size - 2 * padding
    scale = available / max(width, height)
    offset_x = padding + (available - width * scale) / 2
    offset_y = padding + (available - height * scale) / 2
    return [((x - min_x) * scale + offset_x, (y - min_y) * scale + offset_y) for x, y in zip(xs, ys)]


def _edge_color(
    assignment: str,
    assignment_visibility: str,
    config: Mapping[str, Any],
    style: str,
    role: str | None,
) -> RGB:
    if assignment_visibility == "visible":
        if style.startswith("bp_color") and role:
            return BP_VISIBLE_COLORS.get(role, VISIBLE_COLORS.get(assignment, VISIBLE_COLORS["U"]))
        return VISIBLE_COLORS.get(assignment, VISIBLE_COLORS["U"])
    if assignment == "B":
        return tuple(config.get("border", (15, 15, 15)))  # type: ignore[return-value]
    return tuple(config.get("crease", (42, 42, 42)))  # type: ignore[return-value]


def _jitter(point: Tuple[float, float], amount: int, rng: random.Random) -> Tuple[float, float]:
    if amount <= 0:
        return point
    return (point[0] + rng.uniform(-amount, amount), point[1] + rng.uniform(-amount, amount))


def _add_noise(img: Any, amount: float, rng: random.Random) -> None:
    if amount <= 0:
        return
    pixels = img.load()
    width, height = img.size
    stride = max(1, int(1 / amount))
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            delta = rng.randint(-10, 10)
            r, g, b = pixels[x, y]
            pixels[x, y] = (_clamp(r + delta), _clamp(g + delta), _clamp(b + delta))


def _line_width(style: str, image_size: int) -> int:
    base = max(1, round(image_size / 220))
    if style.startswith("faint"):
        return max(1, base - 1)
    return base


def _summarize(values: List[float] | List[int]) -> Dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _clamp(value: int) -> int:
    return max(0, min(255, value))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Synthetic dataset root written by Bun generator")
    parser.add_argument("--raw-manifest", help="Path to raw-manifest.jsonl")
    parser.add_argument("--recipe", help="Path to recipe.json")
    parser.add_argument("--manifest", help="Output manifest.jsonl path")
    parser.add_argument("--contact-sheet", help="Output contact sheet path")
    args = parser.parse_args(argv)

    rows = render_dataset(
        root=args.root,
        raw_manifest_path=args.raw_manifest,
        recipe_path=args.recipe,
        manifest_path=args.manifest,
        contact_sheet_path=args.contact_sheet,
    )
    print(f"Rendered {len(rows)} manifest rows under {args.root}")


if __name__ == "__main__":
    main()
