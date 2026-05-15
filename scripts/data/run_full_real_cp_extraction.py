#!/usr/bin/env python3
"""Run the staged full real CP extraction workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.scraping.manifest import ensure_output_tree, make_run_id, relative_to_root, write_json
from src.data.scraping.scraper import scrape_cpoogle, scrape_obb
from src.data.scraping.workflow import (
    build_final_real_cp_dataset,
    classify_existing_crops,
    find_run_manifest,
    refetch_cpoogle_originals_from_manifest,
)


ALL_STAGES = ("native", "screening", "obb", "refetch", "gemini", "final")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/output/scraped"))
    parser.add_argument("--final-output-root", type=Path, default=Path("data/output/scraped/final"))
    parser.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=list(ALL_STAGES))
    parser.add_argument("--native-run")
    parser.add_argument("--screening-run")
    parser.add_argument("--obb-run")
    parser.add_argument("--original-run")
    parser.add_argument("--classified-crop-manifest", type=Path)
    parser.add_argument("--limit-assets", type=int, help="Smoke-test limit for scrape/refetch stages")
    parser.add_argument("--screening-image-download-size", type=int, default=2048)
    parser.add_argument("--native-workers", type=int, default=16)
    parser.add_argument("--screening-workers", type=int, default=16)
    parser.add_argument("--obb-workers", type=int, default=8)
    parser.add_argument("--refetch-workers", type=int, default=12)
    parser.add_argument("--native-request-delay", type=float, default=0.05)
    parser.add_argument("--screening-request-delay", type=float, default=0.05)
    parser.add_argument("--obb-request-delay", type=float, default=0.10)
    parser.add_argument("--refetch-request-delay", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore existing raw assets/manifests during scrape and refetch stages.",
    )
    parser.add_argument("--gemini-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--gemini-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--gemini-cost-only", action="store_true")
    parser.add_argument("--gemini-max-calls", type=int)
    parser.add_argument(
        "--drive-api-key",
        default=os.environ.get("GOOGLE_DRIVE_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
    )
    parser.add_argument(
        "--gemini-api-key",
        default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
    )
    args = parser.parse_args()

    output_root = ensure_output_tree(args.output_root)
    workflow_run_id = make_run_id("real-cp-extraction")
    state_path = output_root / "source_snapshots" / f"full_extraction_{workflow_run_id}.json"
    state: dict[str, object] = {
        "workflow_run_id": workflow_run_id,
        "stages": list(args.stages),
        "output_root": output_root.as_posix(),
        "final_output_root": args.final_output_root.as_posix(),
    }

    def save_state() -> None:
        write_json(state_path, state)

    if "native" in args.stages:
        summary = scrape_cpoogle(
            output_root=output_root,
            limit_assets=args.limit_assets,
            include_images=False,
            include_pdfs=False,
            include_native=True,
            request_delay=args.native_request_delay,
            timeout=args.timeout,
            retries=args.retries,
            workers=args.native_workers,
            drive_api_key=args.drive_api_key,
            resume=not args.force_download,
        )
        args.native_run = summary.run_id
        state["native_run"] = summary.run_id
        state["native_summary"] = summary
        save_state()

    if "screening" in args.stages:
        summary = scrape_cpoogle(
            output_root=output_root,
            limit_assets=args.limit_assets,
            include_images=True,
            include_pdfs=True,
            include_native=False,
            image_download_size=args.screening_image_download_size,
            request_delay=args.screening_request_delay,
            timeout=args.timeout,
            retries=args.retries,
            workers=args.screening_workers,
            drive_api_key=args.drive_api_key,
            gemini_mode="off",
            resume=not args.force_download,
        )
        args.screening_run = summary.run_id
        state["screening_run"] = summary.run_id
        state["screening_summary"] = summary
        save_state()

    if "obb" in args.stages:
        summary = scrape_obb(
            output_root=output_root,
            limit_assets=args.limit_assets,
            request_delay=args.obb_request_delay,
            timeout=args.timeout,
            retries=args.retries,
            workers=args.obb_workers,
            gemini_mode="off",
            resume=not args.force_download,
        )
        args.obb_run = summary.run_id
        state["obb_run"] = summary.run_id
        state["obb_summary"] = summary
        save_state()

    if "refetch" in args.stages:
        if not args.screening_run:
            raise ValueError("--screening-run is required for the refetch stage")
        summary = refetch_cpoogle_originals_from_manifest(
            scraped_root=output_root,
            screening_run=args.screening_run,
            statuses=("accepted", "review"),
            image_download_size=0,
            workers=args.refetch_workers,
            request_delay=args.refetch_request_delay,
            timeout=args.timeout,
            retries=args.retries,
            drive_api_key=args.drive_api_key,
            limit_assets=args.limit_assets,
            resume=not args.force_download,
        )
        args.original_run = summary.run_id
        state["original_run"] = summary.run_id
        state["original_summary"] = summary
        save_state()

    if "gemini" in args.stages:
        if not args.original_run:
            raise ValueError("--original-run is required for the gemini stage")
        crop_manifest = find_run_manifest(output_root, "crops", args.original_run)
        output_manifest = crop_manifest.with_name(f"{crop_manifest.stem}_gemini.jsonl")
        summary = classify_existing_crops(
            crop_manifest=crop_manifest,
            output_manifest=output_manifest,
            scraped_root=output_root,
            statuses=("review",),
            model=args.gemini_model,
            confidence_threshold=args.gemini_confidence_threshold,
            cost_only=args.gemini_cost_only,
            max_calls=args.gemini_max_calls,
            api_key=args.gemini_api_key,
        )
        args.classified_crop_manifest = Path(summary.output_manifest)
        state["classified_crop_manifest"] = relative_to_root(args.classified_crop_manifest, output_root)
        state["gemini_summary"] = summary
        save_state()

    if "final" in args.stages:
        if not any([args.native_run, args.original_run, args.obb_run, args.screening_run]):
            raise ValueError("At least one run id is required for the final stage")
        summary = build_final_real_cp_dataset(
            scraped_root=output_root,
            output_root=args.final_output_root,
            native_run=args.native_run,
            cpoogle_original_run=args.original_run,
            cpoogle_original_crop_manifest=args.classified_crop_manifest,
            obb_run=args.obb_run,
            cpoogle_screening_run=args.screening_run,
        )
        state["final_summary"] = summary
        save_state()

    print(state_path)


if __name__ == "__main__":
    main()
