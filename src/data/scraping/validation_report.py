"""Visual QA reports for CP crop detection runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .manifest import read_jsonl, relative_to_root, write_json, write_jsonl
from .workflow import find_run_manifest


@dataclass
class ValidationReportSummary:
    run_id: str
    manifest_source: str
    output_dir: str
    crop_manifest: str
    asset_manifest: str
    total_crops: int
    status_counts: dict[str, int]
    reason_counts: dict[str, int]
    sheets: dict[str, str]
    html_report: str
    records_manifest: str


def _resolve_path(path_value: str | None, scraped_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    rooted = scraped_root / path
    if rooted.exists():
        return rooted
    return path


def _shorten(value: object, limit: int = 46) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def _record_image_path(record: dict[str, Any], scraped_root: Path) -> Path | None:
    return _resolve_path(record.get("crop_path") or record.get("source_path"), scraped_root)


def _make_tile(
    record: dict[str, Any],
    asset: dict[str, Any] | None,
    scraped_root: Path,
    tile_width: int,
    image_height: int,
    label_height: int,
) -> Image.Image:
    tile = Image.new("RGB", (tile_width, image_height + label_height), "white")
    image_path = _record_image_path(record, scraped_root)
    draw = ImageDraw.Draw(tile)
    status = str(record.get("status") or "unknown")
    status_color = {
        "accepted": (26, 120, 52),
        "review": (176, 113, 0),
        "rejected": (180, 38, 38),
        "duplicate": (110, 75, 170),
    }.get(status, (72, 72, 72))

    try:
        if image_path is None:
            raise FileNotFoundError("missing image path")
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        if not record.get("crop_path") and record.get("bbox"):
            bbox = tuple(int(v) for v in record["bbox"])
            overlay = ImageDraw.Draw(image)
            for offset in range(3):
                overlay.rectangle(
                    (bbox[0] - offset, bbox[1] - offset, bbox[2] + offset, bbox[3] + offset),
                    outline=status_color,
                )
        thumb = ImageOps.contain(image, (tile_width, image_height), Image.Resampling.LANCZOS)
        tile.paste(thumb, ((tile_width - thumb.width) // 2, (image_height - thumb.height) // 2))
    except Exception as exc:  # noqa: BLE001 - report generation should not die on one bad asset
        draw.rectangle((0, 0, tile_width - 1, image_height - 1), outline=status_color)
        draw.text((8, 8), _shorten(f"image error: {exc}", 38), fill=(0, 0, 0))

    y = image_height + 5
    draw.rectangle((0, image_height, tile_width, image_height + 4), fill=status_color)
    score = float(record.get("cp_score") or 0.0)
    label_lines = [
        f"{status} cp={score:.2f}",
        _shorten((asset or {}).get("model_name") or record.get("asset_id"), 36),
        _shorten(", ".join(str(r) for r in (record.get("reasons") or [])) or "no reasons", 40),
    ]
    font = ImageFont.load_default()
    for line in label_lines:
        draw.text((6, y), line, fill=(0, 0, 0), font=font)
        y += 14
    return tile


def _write_sheet(
    records: list[dict[str, Any]],
    assets_by_id: dict[str, dict[str, Any]],
    scraped_root: Path,
    output_path: Path,
    max_items: int,
    cols: int,
    tile_width: int,
    image_height: int,
    label_height: int,
) -> Path | None:
    records = records[:max_items]
    if not records:
        return None
    rows = (len(records) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_width, rows * (image_height + label_height)), (242, 242, 242))
    for idx, record in enumerate(records):
        asset = assets_by_id.get(str(record.get("asset_id")))
        tile = _make_tile(record, asset, scraped_root, tile_width, image_height, label_height)
        x = (idx % cols) * tile_width
        y = (idx // cols) * (image_height + label_height)
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path


def build_validation_report(
    scraped_root: str | Path,
    run_id: str,
    manifest_source: str,
    output_dir: str | Path | None = None,
    max_per_status: int = 80,
    cols: int = 5,
    tile_width: int = 240,
    image_height: int = 220,
    label_height: int = 64,
) -> ValidationReportSummary:
    """Build status-specific contact sheets and an HTML QA page for a run."""
    scraped_root = Path(scraped_root)
    crop_manifest = find_run_manifest(scraped_root, "crops", run_id, source=manifest_source)
    asset_manifest = find_run_manifest(scraped_root, "assets", run_id, source=manifest_source)
    crops = read_jsonl(crop_manifest)
    assets = read_jsonl(asset_manifest)
    assets_by_id = {str(row.get("asset_id")): row for row in assets}

    output_dir = Path(output_dir) if output_dir is not None else scraped_root / "review" / f"validation_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for crop in crops:
        asset = assets_by_id.get(str(crop.get("asset_id")), {})
        merged = {
            **crop,
            "model_name": asset.get("model_name"),
            "author_name": asset.get("author_name"),
            "source_url": asset.get("source_url"),
            "download_variant": asset.get("download_variant"),
            "raw_path": asset.get("local_path") or crop.get("source_path"),
        }
        records.append(merged)

    records.sort(key=lambda row: (str(row.get("status")), -float(row.get("cp_score") or 0.0)))
    status_counts = Counter(str(row.get("status") or "unknown") for row in records)
    reason_counts: Counter[str] = Counter()
    for row in records:
        for reason in row.get("reasons") or []:
            reason_counts[str(reason)] += 1

    sheets: dict[str, str] = {}
    for status in ("accepted", "review", "rejected", "duplicate"):
        status_records = [row for row in records if row.get("status") == status]
        sheet = _write_sheet(
            status_records,
            assets_by_id,
            scraped_root,
            output_dir / f"{status}.png",
            max_items=max_per_status,
            cols=cols,
            tile_width=tile_width,
            image_height=image_height,
            label_height=label_height,
        )
        if sheet:
            sheets[status] = sheet.name

    all_sheet = _write_sheet(
        records,
        assets_by_id,
        scraped_root,
        output_dir / "all_statuses.png",
        max_items=max_per_status,
        cols=cols,
        tile_width=tile_width,
        image_height=image_height,
        label_height=label_height,
    )
    if all_sheet:
        sheets["all"] = all_sheet.name

    records_manifest = output_dir / "validation_records.jsonl"
    write_jsonl(records_manifest, records)

    html_path = output_dir / "index.html"
    summary = ValidationReportSummary(
        run_id=run_id,
        manifest_source=manifest_source,
        output_dir=output_dir.as_posix(),
        crop_manifest=crop_manifest.as_posix(),
        asset_manifest=asset_manifest.as_posix(),
        total_crops=len(records),
        status_counts=dict(status_counts),
        reason_counts=dict(reason_counts.most_common()),
        sheets=sheets,
        html_report=html_path.as_posix(),
        records_manifest=records_manifest.as_posix(),
    )
    _write_html_report(html_path, summary)
    write_json(output_dir / "summary.json", summary)
    return summary


def _write_html_report(path: Path, summary: ValidationReportSummary) -> None:
    def rows(mapping: dict[str, int]) -> str:
        return "\n".join(f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in mapping.items())

    sheet_sections = "\n".join(
        f"<h2>{name}</h2><img src=\"{filename}\" alt=\"{name} sheet\" />"
        for name, filename in summary.sheets.items()
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CP Detection Validation {summary.run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; }}
    td, th {{ border: 1px solid #d5d8dc; padding: 6px 10px; }}
    img {{ max-width: 100%; border: 1px solid #d5d8dc; }}
    code {{ background: #f3f4f6; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>CP Detection Validation</h1>
  <p>Run <code>{summary.run_id}</code>, manifest source <code>{summary.manifest_source}</code>.</p>
  <h2>Status Counts</h2>
  <table><tr><th>Status</th><th>Count</th></tr>{rows(summary.status_counts)}</table>
  <h2>Reason Counts</h2>
  <table><tr><th>Reason</th><th>Count</th></tr>{rows(summary.reason_counts)}</table>
  {sheet_sections}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
