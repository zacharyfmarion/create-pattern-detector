"""Workflow helpers for full real CP extraction runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from .gemini_classifier import (
    GeminiCPClassifier,
    GeminiCostEstimate,
    estimate_gemini_cost_for_images,
)
from .manifest import (
    ensure_output_tree,
    make_run_id,
    read_jsonl,
    relative_to_root,
    sanitize_slug,
    utc_now,
    write_json,
    write_jsonl,
)
from .scraper import ScrapeSummary, _run_candidates
from .sources import SourceAssetCandidate


def find_run_manifest(
    scraped_root: str | Path,
    kind: str,
    run_id: str,
    source: str | None = None,
    suffix: str = "jsonl",
) -> Path:
    """Find a run manifest by source/kind/run id."""
    manifests = Path(scraped_root) / "manifests"
    if source:
        exact = manifests / f"{source}_{kind}_{run_id}.{suffix}"
        if exact.exists():
            return exact
    matches = sorted(manifests.glob(f"*_{kind}_{run_id}.{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No {kind} manifest found for run {run_id} under {manifests}")
    if len(matches) > 1 and source is None:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple {kind} manifests match run {run_id}; pass a source. Matches: {names}")
    return matches[0]


def _candidate_from_manifest_record(record: dict[str, Any]) -> SourceAssetCandidate:
    return SourceAssetCandidate(
        source=str(record.get("source") or "cpoogle"),
        asset_id=str(record["asset_id"]),
        filename=str(record.get("filename") or record["asset_id"]),
        category=str(record.get("category") or "image"),
        mime_type=record.get("mime_type"),
        source_url=str(record.get("source_url") or ""),
        download_url=record.get("download_url"),
        drive_file_id=record.get("drive_file_id"),
        model_id=record.get("model_id"),
        model_name=record.get("model_name"),
        author_name=record.get("author_name"),
        author_id=record.get("author_id"),
        priority=int(record.get("priority") or 100),
        metadata=dict(record.get("metadata") or {}),
    )


def select_refetch_candidates(
    scraped_root: str | Path,
    screening_run: str,
    statuses: Iterable[str] = ("accepted", "review"),
    source: str = "cpoogle",
    limit_assets: int | None = None,
) -> list[SourceAssetCandidate]:
    """Select image candidates from a screening crop manifest for original refetch."""
    status_set = set(statuses)
    crop_manifest = find_run_manifest(scraped_root, "crops", screening_run, source=source)
    candidate_manifest = find_run_manifest(scraped_root, "candidates", screening_run, source=source)
    crops = read_jsonl(crop_manifest)
    candidate_rows = {str(row["asset_id"]): row for row in read_jsonl(candidate_manifest)}

    selected: dict[str, list[dict[str, Any]]] = {}
    for crop in crops:
        if crop.get("status") in status_set:
            selected.setdefault(str(crop["asset_id"]), []).append(crop)

    candidates: list[SourceAssetCandidate] = []
    for asset_id, crop_rows in selected.items():
        row = candidate_rows.get(asset_id)
        if not row or row.get("category") != "image":
            continue
        candidate = _candidate_from_manifest_record(row)
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "screening_run_id": screening_run,
                "screening_statuses": sorted({str(c.get("status")) for c in crop_rows}),
                "screening_crop_paths": [c.get("crop_path") for c in crop_rows if c.get("crop_path")],
                "screening_perceptual_hashes": [
                    c.get("perceptual_hash") for c in crop_rows if c.get("perceptual_hash")
                ],
            }
        )
        candidates.append(
            SourceAssetCandidate(
                source=candidate.source,
                asset_id=candidate.asset_id,
                filename=candidate.filename,
                category=candidate.category,
                mime_type=candidate.mime_type,
                source_url=candidate.source_url,
                download_url=candidate.download_url,
                drive_file_id=candidate.drive_file_id,
                model_id=candidate.model_id,
                model_name=candidate.model_name,
                author_name=candidate.author_name,
                author_id=candidate.author_id,
                priority=candidate.priority,
                metadata=metadata,
            )
        )

    candidates.sort(key=lambda c: (c.priority, c.model_name or "", c.filename))
    if limit_assets is not None:
        candidates = candidates[:limit_assets]
    return candidates


def refetch_cpoogle_originals_from_manifest(
    scraped_root: str | Path = "data/output/scraped",
    screening_run: str = "",
    statuses: Iterable[str] = ("accepted", "review"),
    image_download_size: int = 0,
    workers: int = 12,
    request_delay: float = 0.10,
    timeout: float = 30.0,
    retries: int = 2,
    drive_api_key: str | None = None,
    drive_prefer_api: bool = False,
    pdf_dpi: int = 220,
    max_pdf_pages: int | None = 8,
    limit_assets: int | None = None,
    run_id: str | None = None,
    resume: bool = True,
) -> ScrapeSummary:
    """Re-download original CPOogle image files selected by a screening run."""
    if not screening_run:
        raise ValueError("screening_run is required")
    output_root = ensure_output_tree(scraped_root)
    candidates = select_refetch_candidates(
        output_root,
        screening_run=screening_run,
        statuses=statuses,
        source="cpoogle",
        limit_assets=limit_assets,
    )
    run_id = run_id or make_run_id("cpoogle-originals")
    summary = _run_candidates(
        source="cpoogle_originals",
        run_id=run_id,
        output_root=output_root,
        candidates=candidates,
        dry_run=False,
        limit_assets=None,
        process_crops=True,
        process_native=False,
        timeout=timeout,
        retries=retries,
        request_delay=request_delay,
        drive_api_key=drive_api_key,
        drive_prefer_api=drive_prefer_api,
        image_download_size=image_download_size,
        pdf_dpi=pdf_dpi,
        max_pdf_pages=max_pdf_pages,
        gemini_mode="off",
        gemini_model="gemini-2.5-flash-lite",
        gemini_cost_only=False,
        gemini_max_calls=None,
        gemini_confidence_threshold=0.70,
        workers=workers,
        resume=resume,
    )
    write_json(
        output_root / "source_snapshots" / f"cpoogle_originals_run_{run_id}.json",
        {
            "run_id": run_id,
            "screening_run_id": screening_run,
            "statuses": sorted(set(statuses)),
            "image_download_size": image_download_size,
            "summary": summary,
        },
    )
    return summary


@dataclass
class ExistingCropClassificationSummary:
    input_manifest: str
    output_manifest: str
    model: str
    selected: int = 0
    classified: int = 0
    errors: int = 0
    promoted: int = 0
    rejected: int = 0
    cost_only: bool = False
    estimate: GeminiCostEstimate | None = None
    elapsed_seconds: float = 0.0


def _resolve_data_path(path_value: str | None, manifest_path: Path, scraped_root: str | Path | None = None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if scraped_root is not None:
        rooted = Path(scraped_root) / path
        if rooted.exists():
            return rooted
    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative
    return path


def classify_existing_crops(
    crop_manifest: str | Path,
    output_manifest: str | Path | None = None,
    scraped_root: str | Path | None = None,
    statuses: Iterable[str] = ("review",),
    model: str = "gemini-2.5-flash-lite",
    confidence_threshold: float = 0.70,
    cost_only: bool = False,
    max_calls: int | None = None,
    api_key: str | None = None,
) -> ExistingCropClassificationSummary:
    """Classify an existing crop manifest with Gemini and write a merged manifest."""
    start = perf_counter()
    crop_manifest = Path(crop_manifest)
    rows = read_jsonl(crop_manifest)
    status_set = set(statuses)
    selected_indices = [
        idx
        for idx, row in enumerate(rows)
        if row.get("status") in status_set and row.get("crop_path")
    ]
    if max_calls is not None:
        selected_indices = selected_indices[:max_calls]

    image_paths: list[Path] = []
    for idx in selected_indices:
        image_path = _resolve_data_path(str(rows[idx].get("crop_path")), crop_manifest, scraped_root)
        if image_path is not None:
            image_paths.append(image_path)
    estimate = estimate_gemini_cost_for_images(image_paths, model=model) if image_paths else None

    summary = ExistingCropClassificationSummary(
        input_manifest=crop_manifest.as_posix(),
        output_manifest="",
        model=model,
        selected=len(selected_indices),
        cost_only=cost_only,
        estimate=estimate,
    )

    if not cost_only and selected_indices:
        classifier = GeminiCPClassifier(model=model, api_key=api_key)
        for idx in selected_indices:
            row = rows[idx]
            image_path = _resolve_data_path(str(row.get("crop_path")), crop_manifest, scraped_root)
            if image_path is None:
                continue
            classification = classifier.classify_image(image_path)
            row["gemini"] = {
                "status": classification.status,
                "is_crease_pattern": classification.is_crease_pattern,
                "confidence": classification.confidence,
                "label": classification.label,
                "reason": classification.reason,
                "error": classification.error,
                "model": model,
                "classified_at": utc_now(),
            }
            if classification.status != "classified":
                summary.errors += 1
                continue
            summary.classified += 1
            confidence = classification.confidence or 0.0
            reasons = list(row.get("reasons") or [])
            if confidence < confidence_threshold:
                row["needs_review"] = True
                reasons.append("gemini_low_confidence")
            elif classification.is_crease_pattern:
                if row.get("status") == "review":
                    row["status"] = "accepted"
                    row["needs_review"] = False
                    reasons.append("gemini_promoted")
                    summary.promoted += 1
            else:
                row["status"] = "rejected"
                row["needs_review"] = False
                reasons.append("gemini_rejected")
                summary.rejected += 1
            row["reasons"] = reasons

    if output_manifest is None:
        output_manifest = crop_manifest.with_name(f"{crop_manifest.stem}_gemini_{make_run_id('classified')}.jsonl")
    output_manifest = Path(output_manifest)
    write_jsonl(output_manifest, rows)
    summary.output_manifest = output_manifest.as_posix()
    summary.elapsed_seconds = perf_counter() - start
    write_json(output_manifest.with_suffix(".summary.json"), summary)
    return summary


@dataclass
class FinalDatasetSummary:
    output_root: str
    native_rows: int = 0
    input_crop_rows: int = 0
    usable_images: int = 0
    review_images: int = 0
    rejected_images: int = 0
    duplicate_groups: int = 0
    manifests: dict[str, str] | None = None


def _load_manifest_optional(scraped_root: Path, kind: str, run_id: str | None) -> list[dict[str, Any]]:
    if not run_id:
        return []
    return read_jsonl(find_run_manifest(scraped_root, kind, run_id))


def _asset_index(scraped_root: Path, run_id: str | None) -> dict[str, dict[str, Any]]:
    return {str(row["asset_id"]): row for row in _load_manifest_optional(scraped_root, "assets", run_id)}


def _quality_rank(record: dict[str, Any]) -> tuple[int, int, float]:
    variant = record.get("download_variant")
    quality = 1 if variant == "drive_thumbnail" else 0
    status_rank = 0 if record.get("status") == "accepted" else 1
    return (quality, status_rank, -float(record.get("cp_score") or 0.0))


def _dedupe_keys(record: dict[str, Any]) -> list[str]:
    page = record.get("page_index")
    page_key = "asset" if page is None else f"p{page}"
    keys: list[str] = []
    for field in ("content_sha256", "normalized_crop_hash", "perceptual_hash"):
        value = record.get(field)
        if value:
            keys.append(f"{field}:{value}")
    if record.get("source_url"):
        keys.append(f"source_url:{record['source_url']}:{page_key}")
    if record.get("drive_file_id"):
        keys.append(f"drive_file_id:{record['drive_file_id']}:{page_key}")
    return keys


def _final_record(
    crop: dict[str, Any],
    asset: dict[str, Any] | None,
    dataset_source: str,
    scraped_root: Path,
) -> dict[str, Any]:
    asset = asset or {}
    metadata = dict(asset.get("metadata") or {})
    dataset_id = sanitize_slug(
        f"{dataset_source}-{crop.get('asset_id')}-{crop.get('page_index', 'asset')}-{crop.get('perceptual_hash') or 'crop'}",
        max_len=160,
    )
    return {
        "dataset_id": dataset_id,
        "source": asset.get("source") or dataset_source,
        "dataset_source": dataset_source,
        "asset_id": crop.get("asset_id"),
        "model_id": asset.get("model_id"),
        "model_name": asset.get("model_name"),
        "author_name": asset.get("author_name"),
        "category": asset.get("category"),
        "mime_type": asset.get("mime_type"),
        "source_url": asset.get("source_url"),
        "download_url": asset.get("download_url"),
        "drive_file_id": asset.get("drive_file_id"),
        "raw_path": asset.get("local_path") or crop.get("source_path"),
        "crop_path": crop.get("crop_path"),
        "page_index": crop.get("page_index"),
        "status": crop.get("status"),
        "needs_review": bool(crop.get("needs_review")),
        "cp_score": crop.get("cp_score"),
        "clean_digital_score": crop.get("clean_digital_score"),
        "hand_drawn_score": crop.get("hand_drawn_score"),
        "photo_text_score": crop.get("photo_text_score"),
        "perceptual_hash": crop.get("perceptual_hash"),
        "content_sha256": crop.get("content_sha256"),
        "normalized_crop_hash": crop.get("content_sha256"),
        "download_variant": asset.get("download_variant") or "original",
        "image_download_size": asset.get("image_download_size"),
        "screening_run_id": metadata.get("screening_run_id"),
        "usage_policy": asset.get("usage_policy") or "internal-only",
        "reasons": list(crop.get("reasons") or []),
        "gemini": crop.get("gemini"),
        "manifest_recorded_at": utc_now(),
    }


def build_final_real_cp_dataset(
    scraped_root: str | Path = "data/output/scraped",
    output_root: str | Path = "data/output/scraped/final",
    native_run: str | None = None,
    cpoogle_original_run: str | None = None,
    obb_run: str | None = None,
    cpoogle_screening_run: str | None = None,
    cpoogle_original_crop_manifest: str | Path | None = None,
) -> FinalDatasetSummary:
    """Build the final usable image/native manifests from completed scrape runs."""
    scraped_root = Path(scraped_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    native_rows = _load_manifest_optional(scraped_root, "native", native_run)
    native_manifest = output_root / "native_manifest.jsonl"
    write_jsonl(native_manifest, native_rows)

    inputs: list[tuple[str, list[dict[str, Any]], dict[str, dict[str, Any]]]] = []
    if cpoogle_original_run:
        original_crops = (
            read_jsonl(cpoogle_original_crop_manifest)
            if cpoogle_original_crop_manifest
            else _load_manifest_optional(scraped_root, "crops", cpoogle_original_run)
        )
        inputs.append(
            (
                "cpoogle_original",
                original_crops,
                _asset_index(scraped_root, cpoogle_original_run),
            )
        )
    if obb_run:
        inputs.append(("obb", _load_manifest_optional(scraped_root, "crops", obb_run), _asset_index(scraped_root, obb_run)))
    if cpoogle_screening_run:
        screening_assets = _asset_index(scraped_root, cpoogle_screening_run)
        pdf_crops = [
            crop
            for crop in _load_manifest_optional(scraped_root, "crops", cpoogle_screening_run)
            if (screening_assets.get(str(crop.get("asset_id"))) or {}).get("category") == "pdf"
        ]
        inputs.append(("cpoogle_pdf_screening", pdf_crops, screening_assets))

    usable_candidates: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for dataset_source, crops, assets in inputs:
        for crop in crops:
            asset = assets.get(str(crop.get("asset_id")))
            record = _final_record(crop, asset, dataset_source, scraped_root)
            if crop.get("status") in {"accepted", "review"} and crop.get("crop_path"):
                usable_candidates.append(record)
            else:
                rejects.append(record)

    usable_candidates.sort(key=_quality_rank)
    seen: dict[str, dict[str, Any]] = {}
    usable: list[dict[str, Any]] = []
    dedupe_groups: list[dict[str, Any]] = []
    for record in usable_candidates:
        keys = _dedupe_keys(record)
        matches = [seen[key] for key in keys if key in seen]
        if matches:
            kept = matches[0]
            duplicate = dict(record)
            duplicate["status"] = "duplicate"
            duplicate["needs_review"] = True
            duplicate["reasons"] = list(duplicate.get("reasons") or []) + [f"duplicate_of:{kept['dataset_id']}"]
            rejects.append(duplicate)
            dedupe_groups.append(
                {
                    "kept_dataset_id": kept["dataset_id"],
                    "duplicate_dataset_id": record["dataset_id"],
                    "matching_keys": [key for key in keys if key in seen],
                }
            )
            continue
        usable.append(record)
        for key in keys:
            seen[key] = record

    usable_manifest = output_root / "final_usable_images.jsonl"
    rejects_manifest = output_root / "final_rejects.jsonl"
    dedupe_manifest = output_root / "dedupe_groups.jsonl"
    summary_path = output_root / "summary.json"
    write_jsonl(usable_manifest, usable)
    write_jsonl(rejects_manifest, rejects)
    write_jsonl(dedupe_manifest, dedupe_groups)

    summary = FinalDatasetSummary(
        output_root=output_root.as_posix(),
        native_rows=len(native_rows),
        input_crop_rows=sum(len(crops) for _, crops, _ in inputs),
        usable_images=len(usable),
        review_images=sum(1 for row in usable if row.get("status") == "review"),
        rejected_images=len(rejects),
        duplicate_groups=len(dedupe_groups),
        manifests={
            "native": relative_to_root(native_manifest, output_root) or native_manifest.as_posix(),
            "usable_images": relative_to_root(usable_manifest, output_root) or usable_manifest.as_posix(),
            "rejects": relative_to_root(rejects_manifest, output_root) or rejects_manifest.as_posix(),
            "dedupe": relative_to_root(dedupe_manifest, output_root) or dedupe_manifest.as_posix(),
            "summary": relative_to_root(summary_path, output_root) or summary_path.as_posix(),
        },
    )
    write_json(summary_path, summary)
    return summary
