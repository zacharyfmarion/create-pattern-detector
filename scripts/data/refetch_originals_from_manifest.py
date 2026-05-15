#!/usr/bin/env python3
"""Re-fetch original CPOogle images selected by a screening crop manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.scraping.workflow import refetch_cpoogle_originals_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/output/scraped"))
    parser.add_argument("--screening-run", required=True, help="CPOogle screening run id")
    parser.add_argument("--statuses", nargs="+", default=["accepted", "review"])
    parser.add_argument(
        "--image-download-size",
        type=int,
        default=0,
        help="Set to 0 for original Drive files. Nonzero values use Drive thumbnails.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--request-delay", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit-assets", type=int)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore existing original raw assets/manifests and download again.",
    )
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument("--max-pdf-pages", type=int, default=8)
    parser.add_argument(
        "--drive-api-key",
        default=os.environ.get("GOOGLE_DRIVE_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
    )
    parser.add_argument("--drive-media-api", action="store_true")
    args = parser.parse_args()

    summary = refetch_cpoogle_originals_from_manifest(
        scraped_root=args.output_root,
        screening_run=args.screening_run,
        statuses=args.statuses,
        image_download_size=args.image_download_size,
        workers=args.workers,
        request_delay=args.request_delay,
        timeout=args.timeout,
        retries=args.retries,
        drive_api_key=args.drive_api_key,
        drive_prefer_api=args.drive_media_api,
        pdf_dpi=args.pdf_dpi,
        max_pdf_pages=args.max_pdf_pages,
        limit_assets=args.limit_assets,
        resume=not args.force_download,
    )
    print(summary)


if __name__ == "__main__":
    main()
