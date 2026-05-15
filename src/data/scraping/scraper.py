"""High-level real CP scraping orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import time
from time import perf_counter
from typing import Any

from .downloader import download_drive_file, download_url
from .gemini_classifier import (
    GeminiCPClassifier,
    GeminiCostEstimate,
    estimate_gemini_cost_for_images,
)
from .manifest import (
    classify_asset,
    ensure_output_tree,
    make_run_id,
    read_jsonl,
    relative_to_root,
    sanitize_slug,
    sha256_file,
    utc_now,
    unique_path,
    write_json,
    write_jsonl,
)
from .native_import import NativeImportResult, import_native_asset
from .sources import (
    CPOOGLE_MODELS_URL,
    OBB_URL,
    SourceAssetCandidate,
    fetch_text,
    iter_cpoogle_assets,
    iter_obb_assets,
    load_obb_html,
    parse_obb_html,
)


@dataclass
class ScrapeSummary:
    source: str
    run_id: str
    output_root: str
    candidates: int = 0
    downloaded: int = 0
    failed_downloads: int = 0
    native_assets: int = 0
    converted_native: int = 0
    crop_assets: int = 0
    accepted_crops: int = 0
    review_crops: int = 0
    rejected_crops: int = 0
    duplicate_crops: int = 0
    gemini_mode: str = "off"
    gemini_estimated_cost_usd: float = 0.0
    gemini_images: int = 0
    gemini_classified: int = 0
    gemini_errors: int = 0
    reused_assets: int = 0
    elapsed_seconds: float = 0.0
    manifests: dict[str, str] | None = None


@dataclass
class CandidateProcessResult:
    asset_record: dict[str, Any]
    crop_results: list[Any]
    native_results: list[NativeImportResult]


def _candidate_to_record(candidate: SourceAssetCandidate, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seen_at": utc_now(),
        "source": candidate.source,
        "asset_id": candidate.asset_id,
        "filename": candidate.filename,
        "category": candidate.category,
        "mime_type": candidate.mime_type,
        "source_url": candidate.source_url,
        "download_url": candidate.download_url,
        "drive_file_id": candidate.drive_file_id,
        "model_id": candidate.model_id,
        "model_name": candidate.model_name,
        "author_name": candidate.author_name,
        "author_id": candidate.author_id,
        "priority": candidate.priority,
        "metadata": candidate.metadata,
        "usage_policy": "internal-only",
    }


def _candidate_output_name(candidate: SourceAssetCandidate, image_download_size: int) -> str:
    """Return the deterministic raw-asset filename for a candidate."""
    name_parts = [candidate.asset_id, candidate.model_name or "", candidate.filename]
    output_name = "-".join(sanitize_slug(p, 64) for p in name_parts if p)
    suffix = Path(output_name).suffix
    if candidate.category == "pdf" and not suffix:
        output_name = f"{output_name}.pdf"
    elif candidate.category == "native" and not suffix:
        native_suffix = Path(candidate.filename).suffix.lower()
        if native_suffix:
            output_name = f"{output_name}{native_suffix}"
    elif candidate.category == "image":
        image_suffix = suffix or Path(candidate.filename).suffix.lower() or ".img"
        if candidate.drive_file_id and image_download_size > 0:
            output_name = f"{Path(output_name).stem}-thumb{image_download_size}{image_suffix}"
        else:
            output_name = f"{Path(output_name).stem}-original{image_suffix}"
    return output_name


def _download_variant_for(candidate: SourceAssetCandidate, image_download_size: int) -> tuple[str, int | None]:
    if candidate.category == "image" and candidate.drive_file_id and image_download_size > 0:
        return "drive_thumbnail", image_download_size
    return "original", None


def _resume_cache_key(candidate: SourceAssetCandidate, image_download_size: int) -> tuple[str, str, str, int | None]:
    variant, size = _download_variant_for(candidate, image_download_size)
    return (candidate.asset_id, candidate.category, variant, size)


def _record_matches_resume_request(record: dict[str, Any], candidate: SourceAssetCandidate, image_download_size: int) -> bool:
    variant, size = _download_variant_for(candidate, image_download_size)
    return (
        record.get("asset_id") == candidate.asset_id
        and record.get("category") == candidate.category
        and (record.get("download_variant") or "original") == variant
        and (record.get("image_download_size") if variant == "drive_thumbnail" else None) == size
    )


def _build_resume_cache(output_root: Path) -> dict[tuple[str, str, str, int | None], dict[str, Any]]:
    """Index previously downloaded assets that still exist on disk."""
    cache: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    manifests_dir = output_root / "manifests"
    if not manifests_dir.exists():
        return cache
    for manifest_path in sorted(manifests_dir.glob("*_assets_*.jsonl")):
        try:
            rows = read_jsonl(manifest_path)
        except ValueError:
            continue
        for row in rows:
            if row.get("download_status") not in {"downloaded", "reused"}:
                continue
            local_path_value = row.get("local_path")
            if not local_path_value:
                continue
            local_path = Path(local_path_value)
            if not local_path.is_absolute():
                local_path = output_root / local_path
            if not local_path.exists() or local_path.stat().st_size <= 0:
                continue
            variant = row.get("download_variant") or "original"
            size = row.get("image_download_size") if variant == "drive_thumbnail" else None
            key = (str(row.get("asset_id")), str(row.get("category")), str(variant), size)
            cached = dict(row)
            cached["local_path"] = relative_to_root(local_path, output_root)
            cached["bytes"] = local_path.stat().st_size
            cache[key] = cached
    return cache


def _reuse_record(
    candidate: SourceAssetCandidate,
    output_root: Path,
    local_path: Path,
    image_download_size: int,
    source: str,
    cached_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    variant, size = _download_variant_for(candidate, image_download_size)
    record = {
        **_candidate_to_record(candidate, ""),
        "download_status": "reused",
        "download_error": None,
        "attempted_url": (cached_record or {}).get("attempted_url"),
        "local_path": relative_to_root(local_path, output_root),
        "content_type": (cached_record or {}).get("content_type"),
        "bytes": local_path.stat().st_size,
        "content_sha256": sha256_file(local_path),
        "download_seconds": 0.0,
        "download_method": (cached_record or {}).get("download_method") or "resume_cache",
        "download_variant": variant,
        "image_download_size": size,
        "resume_source": source,
        "resumed_from_run_id": (cached_record or {}).get("run_id"),
    }
    return record


def _download_candidate(
    candidate: SourceAssetCandidate,
    output_root: Path,
    drive_api_key: str | None,
    drive_prefer_api: bool,
    image_download_size: int,
    resume: bool,
    resume_cache: dict[tuple[str, str, str, int | None], dict[str, Any]] | None,
    timeout: float,
    retries: int,
    request_delay: float,
) -> dict[str, Any]:
    start = perf_counter()
    raw_dir = output_root / "raw_assets" / candidate.source / candidate.category
    output_name = _candidate_output_name(candidate, image_download_size)
    destination = raw_dir / output_name

    if resume:
        cache_key = _resume_cache_key(candidate, image_download_size)
        cached_record = (resume_cache or {}).get(cache_key)
        if cached_record and _record_matches_resume_request(cached_record, candidate, image_download_size):
            cached_path_value = cached_record.get("local_path")
            cached_path = Path(cached_path_value) if cached_path_value else None
            if cached_path and not cached_path.is_absolute():
                cached_path = output_root / cached_path
            if cached_path and cached_path.exists() and cached_path.stat().st_size > 0:
                return _reuse_record(candidate, output_root, cached_path, image_download_size, "manifest", cached_record)
        if destination.exists() and destination.stat().st_size > 0:
            return _reuse_record(candidate, output_root, destination, image_download_size, "raw_asset")

    destination = unique_path(raw_dir, output_name)

    if candidate.drive_file_id:
        result = download_drive_file(
            candidate.drive_file_id,
            candidate.category,
            destination,
            api_key=drive_api_key,
            prefer_api=drive_prefer_api,
            image_download_size=image_download_size,
            timeout=timeout,
            retries=retries,
            delay=request_delay,
        )
    elif candidate.download_url:
        result = download_url(
            candidate.download_url,
            destination,
            timeout=timeout,
            retries=retries,
            delay=request_delay,
        )
    else:
        result = None

    if request_delay:
        time.sleep(request_delay)

    if result is None:
        return {
            **_candidate_to_record(candidate, ""),
            "download_status": "failed",
            "download_error": "no_download_url",
        }

    attempted_url = result.attempted_url or ""
    download_variant = (
        "drive_thumbnail"
        if candidate.category == "image" and "drive.google.com/thumbnail" in attempted_url
        else "original"
    )
    record = {
        **_candidate_to_record(candidate, ""),
        "download_status": result.status,
        "download_error": result.error,
        "attempted_url": result.attempted_url,
        "local_path": relative_to_root(result.local_path, output_root) if result.local_path else None,
        "content_type": result.content_type,
        "bytes": result.bytes_written,
        "content_sha256": sha256_file(result.local_path) if result.local_path and result.local_path.exists() else None,
        "download_seconds": perf_counter() - start,
        "download_method": "drive_api" if "googleapis.com/drive/v3/files/" in attempted_url else "public_url",
        "download_variant": download_variant,
        "image_download_size": image_download_size if download_variant == "drive_thumbnail" else None,
        "resume_source": None,
        "resumed_from_run_id": None,
    }
    return record


def _dedupe_crop_results(
    crop_results: list[CropDetectionResult],
) -> tuple[list[CropDetectionResult], list[dict[str, Any]]]:
    seen: dict[tuple[str | None, str | None], CropDetectionResult] = {}
    groups: list[dict[str, Any]] = []
    for result in crop_results:
        if result.status != "accepted":
            continue
        key = (result.perceptual_hash, result.content_sha256)
        if key == (None, None):
            continue
        first = seen.get(key)
        if first is None:
            seen[key] = result
            continue
        result.status = "duplicate"
        result.needs_review = True
        result.reasons.append("duplicate_crop")
        groups.append(
            {
                "perceptual_hash": result.perceptual_hash,
                "content_sha256": result.content_sha256,
                "kept_asset_id": first.asset_id,
                "duplicate_asset_id": result.asset_id,
                "kept_crop_path": first.crop_path,
                "duplicate_crop_path": result.crop_path,
            }
        )
    return crop_results, groups


def _gemini_candidates(crop_results: list[CropDetectionResult], mode: str) -> list[CropDetectionResult]:
    if mode == "off":
        return []
    if mode == "review":
        return [r for r in crop_results if r.status == "review" and r.crop_path]
    if mode == "accepted-review":
        return [r for r in crop_results if r.status in {"accepted", "review"} and r.crop_path]
    if mode == "all":
        return [r for r in crop_results if r.crop_path]
    raise ValueError(f"Unknown Gemini mode: {mode}")


def _estimate_and_maybe_apply_gemini(
    crop_results: list[CropDetectionResult],
    mode: str,
    model: str,
    cost_only: bool,
    max_calls: int | None,
    confidence_threshold: float,
) -> tuple[GeminiCostEstimate | None, int, int]:
    selected = _gemini_candidates(crop_results, mode)
    if max_calls is not None:
        selected = selected[:max_calls]
    image_paths = [r.crop_path for r in selected if r.crop_path]
    estimate = estimate_gemini_cost_for_images(image_paths, model=model) if image_paths else None
    if cost_only or not image_paths:
        return estimate, 0, 0

    classifier = GeminiCPClassifier(model=model)
    classified = 0
    errors = 0
    for result in selected:
        if not result.crop_path:
            continue
        classification = classifier.classify_image(result.crop_path)
        result.gemini = {
            "status": classification.status,
            "is_crease_pattern": classification.is_crease_pattern,
            "confidence": classification.confidence,
            "label": classification.label,
            "reason": classification.reason,
            "error": classification.error,
            "model": model,
        }
        if classification.status != "classified":
            errors += 1
            continue
        classified += 1
        confidence = classification.confidence or 0.0
        if confidence < confidence_threshold:
            result.needs_review = True
            result.reasons.append("gemini_low_confidence")
            continue
        if classification.is_crease_pattern:
            if result.status == "review":
                result.status = "accepted"
                result.needs_review = False
                result.reasons.append("gemini_promoted")
        else:
            result.status = "rejected"
            result.needs_review = False
            result.reasons.append("gemini_rejected")
    return estimate, classified, errors


def _write_run_outputs(
    output_root: Path,
    source: str,
    run_id: str,
    candidate_records: list[dict[str, Any]],
    asset_records: list[dict[str, Any]],
    crop_results: list[CropDetectionResult],
    native_results: list[NativeImportResult],
    dedupe_groups: list[dict[str, Any]],
    summary: ScrapeSummary,
) -> ScrapeSummary:
    manifests_dir = output_root / "manifests"
    paths = {
        "candidates": manifests_dir / f"{source}_candidates_{run_id}.jsonl",
        "assets": manifests_dir / f"{source}_assets_{run_id}.jsonl",
        "crops": manifests_dir / f"{source}_crops_{run_id}.jsonl",
        "native": manifests_dir / f"{source}_native_{run_id}.jsonl",
        "dedupe": manifests_dir / f"{source}_dedupe_groups_{run_id}.jsonl",
        "summary": manifests_dir / f"{source}_summary_{run_id}.json",
    }
    write_jsonl(paths["candidates"], candidate_records)
    write_jsonl(paths["assets"], asset_records)
    write_jsonl(paths["crops"], crop_results)
    write_jsonl(paths["native"], native_results)
    write_jsonl(paths["dedupe"], dedupe_groups)

    contact_sheet_records = [r for r in crop_results if r.status in {"accepted", "review"}]
    if contact_sheet_records:
        from .crop_detector import write_contact_sheet

        contact_sheet = write_contact_sheet(
            contact_sheet_records,
            output_root / "review" / f"{source}_contact_sheet_{run_id}.png",
        )
        if contact_sheet:
            paths["contact_sheet"] = contact_sheet
    summary.manifests = {key: relative_to_root(value, output_root) or "" for key, value in paths.items()}
    write_json(paths["summary"], summary)
    return summary


def _process_candidate(
    candidate: SourceAssetCandidate,
    output_root: Path,
    run_id: str,
    process_crops: bool,
    process_native: bool,
    timeout: float,
    retries: int,
    request_delay: float,
    drive_api_key: str | None,
    drive_prefer_api: bool,
    image_download_size: int,
    resume: bool,
    resume_cache: dict[tuple[str, str, str, int | None], dict[str, Any]] | None,
    pdf_dpi: int,
    max_pdf_pages: int | None,
) -> CandidateProcessResult:
    record = _download_candidate(
        candidate,
        output_root,
        drive_api_key=drive_api_key,
        drive_prefer_api=drive_prefer_api,
        image_download_size=image_download_size,
        resume=resume,
        resume_cache=resume_cache,
        timeout=timeout,
        retries=retries,
        request_delay=request_delay,
    )
    record["run_id"] = run_id

    local_path_value = record.get("local_path")
    if record.get("download_status") not in {"downloaded", "reused"} or not local_path_value:
        return CandidateProcessResult(record, [], [])

    local_path = output_root / local_path_value
    category = candidate.category or classify_asset(local_path.name, record.get("content_type"))
    crop_results: list[Any] = []
    native_results: list[NativeImportResult] = []

    if category == "native" and process_native:
        native_start = perf_counter()
        native_result = import_native_asset(local_path, output_root / "native", candidate.source, candidate.asset_id)
        record["native_import_seconds"] = perf_counter() - native_start
        native_results.append(native_result)

    if category in {"image", "pdf"} and process_crops:
        try:
            from .crop_detector import PDFRenderingUnavailable, process_asset_for_crops

            crop_start = perf_counter()
            crop_results.extend(
                process_asset_for_crops(
                    local_path,
                    asset_id=candidate.asset_id,
                    crop_root=output_root / "crops" / candidate.source,
                    pdf_dpi=pdf_dpi,
                    max_pdf_pages=max_pdf_pages,
                )
            )
            record["crop_processing_seconds"] = perf_counter() - crop_start
        except (PDFRenderingUnavailable, Exception) as exc:  # noqa: BLE001 - capture per-asset failures in manifests
            from .crop_detector import CropDetectionResult

            crop_results.append(
                CropDetectionResult(
                    asset_id=candidate.asset_id,
                    source_path=local_path.as_posix(),
                    crop_path=None,
                    status="rejected",
                    needs_review=False,
                    bbox=None,
                    page_index=None,
                    cp_score=0.0,
                    clean_digital_score=0.0,
                    hand_drawn_score=0.0,
                    photo_text_score=0.0,
                    perceptual_hash=None,
                    content_sha256=None,
                    reasons=[f"asset_processing_failed: {exc}"],
                    metrics={},
                )
            )
            record["crop_processing_seconds"] = 0.0

    return CandidateProcessResult(record, crop_results, native_results)


def _run_candidates(
    source: str,
    run_id: str,
    output_root: Path,
    candidates: list[SourceAssetCandidate],
    dry_run: bool,
    limit_assets: int | None,
    process_crops: bool,
    process_native: bool,
    timeout: float,
    retries: int,
    request_delay: float,
    drive_api_key: str | None,
    drive_prefer_api: bool,
    image_download_size: int,
    pdf_dpi: int,
    max_pdf_pages: int | None,
    gemini_mode: str,
    gemini_model: str,
    gemini_cost_only: bool,
    gemini_max_calls: int | None,
    gemini_confidence_threshold: float,
    workers: int,
    resume: bool,
) -> ScrapeSummary:
    run_start = perf_counter()
    if limit_assets is not None:
        candidates = candidates[:limit_assets]

    candidate_records = [_candidate_to_record(candidate, run_id) for candidate in candidates]
    summary = ScrapeSummary(
        source=source,
        run_id=run_id,
        output_root=output_root.as_posix(),
        candidates=len(candidates),
        gemini_mode=gemini_mode,
    )
    if dry_run:
        summary.manifests = {}
        write_jsonl(output_root / "manifests" / f"{source}_candidates_{run_id}.jsonl", candidate_records)
        write_json(output_root / "manifests" / f"{source}_summary_{run_id}.json", summary)
        return summary

    asset_records: list[dict[str, Any]] = []
    crop_results: list[CropDetectionResult] = []
    native_results: list[NativeImportResult] = []
    dedupe_groups: list[dict[str, Any]] = []
    resume_cache = _build_resume_cache(output_root) if resume else {}

    process_kwargs = {
        "output_root": output_root,
        "run_id": run_id,
        "process_crops": process_crops,
        "process_native": process_native,
        "timeout": timeout,
        "retries": retries,
        "request_delay": request_delay,
        "drive_api_key": drive_api_key,
        "drive_prefer_api": drive_prefer_api,
        "image_download_size": image_download_size,
        "resume": resume,
        "resume_cache": resume_cache,
        "pdf_dpi": pdf_dpi,
        "max_pdf_pages": max_pdf_pages,
    }
    if workers <= 1:
        results = [_process_candidate(candidate, **process_kwargs) for candidate in candidates]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_candidate, candidate, **process_kwargs) for candidate in candidates]
            for future in as_completed(futures):
                results.append(future.result())

    for result in results:
        record = result.asset_record
        asset_records.append(record)
        local_path_value = record.get("local_path")
        if record.get("download_status") not in {"downloaded", "reused"} or not local_path_value:
            summary.failed_downloads += 1
            continue
        summary.downloaded += 1
        if record.get("download_status") == "reused":
            summary.reused_assets += 1

        category = record.get("category")
        if category == "native":
            summary.native_assets += 1
        if category in {"image", "pdf"} and process_crops:
            summary.crop_assets += 1

        native_results.extend(result.native_results)
        crop_results.extend(result.crop_results)

    summary.converted_native = sum(1 for r in native_results if r.status == "converted")

    crop_results, dedupe_groups = _dedupe_crop_results(crop_results)
    gemini_estimate, gemini_classified, gemini_errors = _estimate_and_maybe_apply_gemini(
        crop_results=crop_results,
        mode=gemini_mode,
        model=gemini_model,
        cost_only=gemini_cost_only,
        max_calls=gemini_max_calls,
        confidence_threshold=gemini_confidence_threshold,
    )
    if gemini_estimate:
        summary.gemini_images = gemini_estimate.images
        summary.gemini_estimated_cost_usd = gemini_estimate.estimated_cost_usd
    summary.gemini_classified = gemini_classified
    summary.gemini_errors = gemini_errors

    # Re-run crop counts after optional Gemini promotion/rejection.
    summary.accepted_crops = sum(1 for r in crop_results if r.status == "accepted")
    summary.review_crops = sum(1 for r in crop_results if r.status == "review")
    summary.rejected_crops = sum(1 for r in crop_results if r.status == "rejected")
    summary.duplicate_crops = sum(1 for r in crop_results if r.status == "duplicate")
    summary.elapsed_seconds = perf_counter() - run_start
    return _write_run_outputs(
        output_root=output_root,
        source=source,
        run_id=run_id,
        candidate_records=candidate_records,
        asset_records=asset_records,
        crop_results=crop_results,
        native_results=native_results,
        dedupe_groups=dedupe_groups,
        summary=summary,
    )


def scrape_cpoogle(
    output_root: str | Path = "data/output/scraped",
    source_json: str | Path | None = None,
    dry_run: bool = False,
    limit_assets: int | None = None,
    include_images: bool = True,
    include_pdfs: bool = True,
    include_native: bool = True,
    include_other: bool = False,
    process_crops: bool = True,
    process_native: bool = True,
    timeout: float = 60.0,
    retries: int = 2,
    request_delay: float = 0.15,
    drive_api_key: str | None = None,
    drive_prefer_api: bool = False,
    image_download_size: int = 1024,
    pdf_dpi: int = 220,
    max_pdf_pages: int | None = 8,
    gemini_mode: str = "off",
    gemini_model: str = "gemini-2.5-flash-lite",
    gemini_cost_only: bool = False,
    gemini_max_calls: int | None = None,
    gemini_confidence_threshold: float = 0.70,
    workers: int = 1,
    resume: bool = True,
) -> ScrapeSummary:
    """Scrape CPOogle candidates, assets, crops, and native imports."""
    output_root = ensure_output_tree(output_root)
    run_id = make_run_id("cpoogle")
    if source_json:
        snapshot_text = Path(source_json).read_text(encoding="utf-8")
        models = json.loads(snapshot_text)
    else:
        snapshot_text = fetch_text(CPOOGLE_MODELS_URL)
        models = json.loads(snapshot_text)
    snapshot_path = output_root / "source_snapshots" / f"cpoogle_models_{run_id}.json"
    snapshot_path.write_text(snapshot_text, encoding="utf-8")

    candidates = iter_cpoogle_assets(
        models,
        include_images=include_images,
        include_pdfs=include_pdfs,
        include_native=include_native,
        include_other=include_other,
    )
    summary = _run_candidates(
        source="cpoogle",
        run_id=run_id,
        output_root=output_root,
        candidates=candidates,
        dry_run=dry_run,
        limit_assets=limit_assets,
        process_crops=process_crops,
        process_native=process_native,
        timeout=timeout,
        retries=retries,
        request_delay=request_delay,
        drive_api_key=drive_api_key,
        drive_prefer_api=drive_prefer_api,
        image_download_size=image_download_size,
        pdf_dpi=pdf_dpi,
        max_pdf_pages=max_pdf_pages,
        gemini_mode=gemini_mode,
        gemini_model=gemini_model,
        gemini_cost_only=gemini_cost_only,
        gemini_max_calls=gemini_max_calls,
        gemini_confidence_threshold=gemini_confidence_threshold,
        workers=workers,
        resume=resume,
    )
    write_json(
        output_root / "source_snapshots" / f"cpoogle_run_{run_id}.json",
        {"run_id": run_id, "source_url": CPOOGLE_MODELS_URL, "snapshot": relative_to_root(snapshot_path, output_root), "summary": summary},
    )
    return summary


def scrape_obb(
    output_root: str | Path = "data/output/scraped",
    source_html: str | Path | None = None,
    dry_run: bool = False,
    limit_assets: int | None = None,
    process_crops: bool = True,
    timeout: float = 60.0,
    retries: int = 2,
    request_delay: float = 0.15,
    drive_api_key: str | None = None,
    drive_prefer_api: bool = False,
    image_download_size: int = 1024,
    gemini_mode: str = "off",
    gemini_model: str = "gemini-2.5-flash-lite",
    gemini_cost_only: bool = False,
    gemini_max_calls: int | None = None,
    gemini_confidence_threshold: float = 0.70,
    workers: int = 1,
    resume: bool = True,
) -> ScrapeSummary:
    """Scrape the Origami By Boice crease-pattern gallery."""
    output_root = ensure_output_tree(output_root)
    run_id = make_run_id("obb")
    if source_html:
        html = load_obb_html(source_html)
    else:
        html = load_obb_html()
    snapshot_path = output_root / "source_snapshots" / f"obb_gallery_{run_id}.html"
    snapshot_path.write_text(html, encoding="utf-8")

    items = parse_obb_html(html)
    candidates = iter_obb_assets(items, include_images=True)
    summary = _run_candidates(
        source="obb",
        run_id=run_id,
        output_root=output_root,
        candidates=candidates,
        dry_run=dry_run,
        limit_assets=limit_assets,
        process_crops=process_crops,
        process_native=False,
        timeout=timeout,
        retries=retries,
        request_delay=request_delay,
        drive_api_key=drive_api_key,
        drive_prefer_api=drive_prefer_api,
        image_download_size=image_download_size,
        pdf_dpi=220,
        max_pdf_pages=None,
        gemini_mode=gemini_mode,
        gemini_model=gemini_model,
        gemini_cost_only=gemini_cost_only,
        gemini_max_calls=gemini_max_calls,
        gemini_confidence_threshold=gemini_confidence_threshold,
        workers=workers,
        resume=resume,
    )
    write_json(
        output_root / "source_snapshots" / f"obb_run_{run_id}.json",
        {"run_id": run_id, "source_url": OBB_URL, "snapshot": relative_to_root(snapshot_path, output_root), "item_count": len(items), "summary": summary},
    )
    return summary


__all__ = ["ScrapeSummary", "scrape_cpoogle", "scrape_obb"]
