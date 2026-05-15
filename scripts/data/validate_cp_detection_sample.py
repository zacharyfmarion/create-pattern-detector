#!/usr/bin/env python3
"""Run or report on a visual CP-detection validation sample."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.scraping.manifest import ensure_output_tree, make_run_id, relative_to_root, write_json
from src.data.scraping.scraper import _run_candidates
from src.data.scraping.sources import (
    CPOOGLE_MODELS_URL,
    OBB_URL,
    fetch_text,
    iter_cpoogle_assets,
    iter_obb_assets,
    parse_obb_html,
)
from src.data.scraping.validation_report import build_validation_report


def _sample_candidates(candidates, sample_size: int, mode: str, seed: int):
    if mode == "first":
        return candidates[:sample_size]
    rng = random.Random(seed)
    if sample_size >= len(candidates):
        return list(candidates)
    return rng.sample(candidates, sample_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/output/scraped"))
    parser.add_argument("--source", choices=["cpoogle", "obb"], default="cpoogle")
    parser.add_argument("--run-id", help="Skip scraping and generate a report for an existing run id")
    parser.add_argument("--manifest-source", help="Manifest source name for --run-id")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--sample-mode", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-pdfs", action="store_true")
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--source-html", type=Path)
    parser.add_argument("--image-download-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-per-status", type=int, default=80)
    args = parser.parse_args()

    output_root = ensure_output_tree(args.output_root)
    if args.run_id:
        manifest_source = args.manifest_source or args.source
        summary = build_validation_report(
            scraped_root=output_root,
            run_id=args.run_id,
            manifest_source=manifest_source,
            max_per_status=args.max_per_status,
        )
        print(summary)
        return

    manifest_source = f"{args.source}_validation"
    run_id = make_run_id(manifest_source)
    if args.source == "cpoogle":
        if args.source_json:
            snapshot_text = args.source_json.read_text(encoding="utf-8")
        else:
            snapshot_text = fetch_text(CPOOGLE_MODELS_URL)
        models = json.loads(snapshot_text)
        candidates = iter_cpoogle_assets(
            models,
            include_images=True,
            include_pdfs=args.include_pdfs,
            include_native=False,
            include_other=False,
        )
        snapshot_path = output_root / "source_snapshots" / f"cpoogle_validation_models_{run_id}.json"
        snapshot_url = CPOOGLE_MODELS_URL
    else:
        if args.source_html:
            snapshot_text = args.source_html.read_text(encoding="utf-8")
        else:
            snapshot_text = fetch_text(OBB_URL)
        items = parse_obb_html(snapshot_text)
        candidates = iter_obb_assets(items, include_images=True)
        snapshot_path = output_root / "source_snapshots" / f"obb_validation_gallery_{run_id}.html"
        snapshot_url = OBB_URL

    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    sampled = _sample_candidates(candidates, args.sample_size, args.sample_mode, args.seed)
    scrape_summary = _run_candidates(
        source=manifest_source,
        run_id=run_id,
        output_root=output_root,
        candidates=sampled,
        dry_run=False,
        limit_assets=None,
        process_crops=True,
        process_native=False,
        timeout=args.timeout,
        retries=args.retries,
        request_delay=args.request_delay,
        drive_api_key=None,
        drive_prefer_api=False,
        image_download_size=args.image_download_size,
        pdf_dpi=220,
        max_pdf_pages=8,
        gemini_mode="off",
        gemini_model="gemini-2.5-flash-lite",
        gemini_cost_only=False,
        gemini_max_calls=None,
        gemini_confidence_threshold=0.70,
        workers=args.workers,
        resume=not args.force_download,
    )
    write_json(
        output_root / "source_snapshots" / f"{manifest_source}_run_{run_id}.json",
        {
            "run_id": run_id,
            "source": args.source,
            "source_url": snapshot_url,
            "snapshot": relative_to_root(snapshot_path, output_root),
            "sample_size": args.sample_size,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "summary": scrape_summary,
        },
    )
    report_summary = build_validation_report(
        scraped_root=output_root,
        run_id=run_id,
        manifest_source=manifest_source,
        max_per_status=args.max_per_status,
    )
    print(report_summary)


if __name__ == "__main__":
    main()
