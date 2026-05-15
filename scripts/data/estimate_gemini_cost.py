#!/usr/bin/env python3
"""Estimate Gemini classifier cost for already-generated crop images."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.scraping.gemini_classifier import (
    DEFAULT_PRICING,
    GEMINI_PRICING_URL,
    GEMINI_TOKENS_URL,
    estimate_gemini_cost_for_images,
)
from src.data.scraping.manifest import IMAGE_EXTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Images or directories of images")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", choices=sorted(DEFAULT_PRICING))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    image_paths: list[Path] = []
    for path in args.paths or [Path("data/output/scraped/crops")]:
        if path.is_dir():
            image_paths.extend(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        elif path.is_file():
            image_paths.append(path)

    image_paths = sorted(image_paths)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    estimate = estimate_gemini_cost_for_images(image_paths, model=args.model)
    pricing = DEFAULT_PRICING[args.model]
    print(f"Model: {estimate.model}")
    print(f"Images: {estimate.images}")
    print(f"Estimated input tokens: {estimate.input_tokens:,}")
    print(f"Estimated output tokens: {estimate.output_tokens:,}")
    print(f"Pricing assumption: ${pricing.input_per_million}/1M input, ${pricing.output_per_million}/1M output")
    print(f"Estimated cost: ${estimate.estimated_cost_usd:.6f}")
    print(f"Pricing URL: {GEMINI_PRICING_URL}")
    print(f"Token docs: {GEMINI_TOKENS_URL}")


if __name__ == "__main__":
    main()
