#!/usr/bin/env python3
"""Scrape Origami By Boice crease-pattern gallery candidates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.scraping.gemini_classifier import GEMINI_PRICING_URL
from src.data.scraping.scraper import scrape_obb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/output/scraped"))
    parser.add_argument("--source-html", type=Path, help="Use a frozen OBB gallery HTML file")
    parser.add_argument("--dry-run", action="store_true", help="Only write source snapshot and candidates manifest")
    parser.add_argument("--limit-assets", type=int, help="Limit candidates for smoke runs")
    parser.add_argument("--no-crops", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="Parallel download/process workers")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore existing raw assets/manifests and download candidates again.",
    )
    parser.add_argument(
        "--image-download-size",
        type=int,
        default=1024,
        help="Accepted for CLI symmetry; only Google Drive image candidates use thumbnail screening.",
    )
    parser.add_argument(
        "--drive-api-key",
        default=os.environ.get("GOOGLE_DRIVE_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        help="Accepted for CLI symmetry; OBB image downloads do not use Google Drive.",
    )
    parser.add_argument("--no-drive-api", action="store_true")
    parser.add_argument("--drive-media-api", action="store_true")
    parser.add_argument(
        "--gemini-mode",
        choices=["off", "review", "accepted-review", "all"],
        default="off",
        help="Optional Gemini classifier stage. Default is off, so scrape costs $0.",
    )
    parser.add_argument("--gemini-model", default="gemini-2.5-flash-lite")
    parser.add_argument(
        "--gemini-cost-only",
        action="store_true",
        help="Estimate Gemini cost from generated crops without calling the model.",
    )
    parser.add_argument("--gemini-max-calls", type=int)
    parser.add_argument("--gemini-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--gemini-pricing-url", action="store_true")
    args = parser.parse_args()

    if args.gemini_pricing_url:
        print(GEMINI_PRICING_URL)

    summary = scrape_obb(
        output_root=args.output_root,
        source_html=args.source_html,
        dry_run=args.dry_run,
        limit_assets=args.limit_assets,
        process_crops=not args.no_crops,
        timeout=args.timeout,
        retries=args.retries,
        request_delay=args.request_delay,
        drive_api_key=args.drive_api_key,
        drive_prefer_api=args.drive_media_api and not args.no_drive_api,
        image_download_size=args.image_download_size,
        gemini_mode=args.gemini_mode,
        gemini_model=args.gemini_model,
        gemini_cost_only=args.gemini_cost_only,
        gemini_max_calls=args.gemini_max_calls,
        gemini_confidence_threshold=args.gemini_confidence_threshold,
        workers=args.workers,
        resume=not args.force_download,
    )
    print(summary)


if __name__ == "__main__":
    main()
