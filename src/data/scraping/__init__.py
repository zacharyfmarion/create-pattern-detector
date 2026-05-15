"""Utilities for collecting and preparing real origami crease pattern data."""

__all__ = [
    "CPCropDetector",
    "CropDetectionResult",
    "SourceAssetCandidate",
    "build_final_real_cp_dataset",
    "build_validation_report",
    "classify_existing_crops",
    "import_native_asset",
    "iter_cpoogle_assets",
    "parse_obb_html",
    "process_asset_for_crops",
    "refetch_cpoogle_originals_from_manifest",
    "scrape_cpoogle",
    "scrape_obb",
    "select_refetch_candidates",
]


def __getattr__(name: str):
    """Lazy-load optional image/PDF dependencies only when needed."""
    if name in {"CPCropDetector", "CropDetectionResult", "process_asset_for_crops"}:
        from .crop_detector import CPCropDetector, CropDetectionResult, process_asset_for_crops

        values = {
            "CPCropDetector": CPCropDetector,
            "CropDetectionResult": CropDetectionResult,
            "process_asset_for_crops": process_asset_for_crops,
        }
        return values[name]
    if name == "import_native_asset":
        from .native_import import import_native_asset

        return import_native_asset
    if name in {"scrape_cpoogle", "scrape_obb"}:
        from .scraper import scrape_cpoogle, scrape_obb

        return {"scrape_cpoogle": scrape_cpoogle, "scrape_obb": scrape_obb}[name]
    if name in {"SourceAssetCandidate", "iter_cpoogle_assets", "parse_obb_html"}:
        from .sources import SourceAssetCandidate, iter_cpoogle_assets, parse_obb_html

        values = {
            "SourceAssetCandidate": SourceAssetCandidate,
            "iter_cpoogle_assets": iter_cpoogle_assets,
            "parse_obb_html": parse_obb_html,
        }
        return values[name]
    if name == "build_validation_report":
        from .validation_report import build_validation_report

        return build_validation_report
    if name in {
        "build_final_real_cp_dataset",
        "classify_existing_crops",
        "refetch_cpoogle_originals_from_manifest",
        "select_refetch_candidates",
    }:
        from .workflow import (
            build_final_real_cp_dataset,
            classify_existing_crops,
            refetch_cpoogle_originals_from_manifest,
            select_refetch_candidates,
        )

        values = {
            "build_final_real_cp_dataset": build_final_real_cp_dataset,
            "classify_existing_crops": classify_existing_crops,
            "refetch_cpoogle_originals_from_manifest": refetch_cpoogle_originals_from_manifest,
            "select_refetch_candidates": select_refetch_candidates,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
