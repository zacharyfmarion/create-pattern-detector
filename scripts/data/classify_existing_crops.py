#!/usr/bin/env python3
"""Classify an existing crop manifest with Gemini and write a merged manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.scraping.gemini_classifier import GEMINI_PRICING_URL
from src.data.scraping.workflow import classify_existing_crops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--scraped-root", type=Path, default=Path("data/output/scraped"))
    parser.add_argument("--status", dest="statuses", nargs="+", default=["review"])
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--cost-only", action="store_true")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
    )
    parser.add_argument("--gemini-pricing-url", action="store_true")
    args = parser.parse_args()

    if args.gemini_pricing_url:
        print(GEMINI_PRICING_URL)

    summary = classify_existing_crops(
        crop_manifest=args.crop_manifest,
        output_manifest=args.output_manifest,
        scraped_root=args.scraped_root,
        statuses=args.statuses,
        model=args.model,
        confidence_threshold=args.confidence_threshold,
        cost_only=args.cost_only,
        max_calls=args.max_calls,
        api_key=args.api_key,
        workers=args.workers,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(summary)


if __name__ == "__main__":
    main()
