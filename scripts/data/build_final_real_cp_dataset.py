#!/usr/bin/env python3
"""Build final real CP image/native manifests from completed scrape runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.scraping.workflow import build_final_real_cp_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scraped-root", type=Path, default=Path("data/output/scraped"))
    parser.add_argument("--output-root", type=Path, default=Path("data/output/scraped/final"))
    parser.add_argument("--native-run")
    parser.add_argument("--cpoogle-original-run")
    parser.add_argument(
        "--cpoogle-original-crop-manifest",
        type=Path,
        help="Optional Gemini-merged crop manifest to use instead of the original run crop manifest.",
    )
    parser.add_argument("--obb-run")
    parser.add_argument(
        "--cpoogle-screening-run",
        help="Optional CPOogle screening run; PDF crops from this run are included in the final dataset.",
    )
    args = parser.parse_args()

    summary = build_final_real_cp_dataset(
        scraped_root=args.scraped_root,
        output_root=args.output_root,
        native_run=args.native_run,
        cpoogle_original_run=args.cpoogle_original_run,
        cpoogle_original_crop_manifest=args.cpoogle_original_crop_manifest,
        obb_run=args.obb_run,
        cpoogle_screening_run=args.cpoogle_screening_run,
    )
    print(summary)


if __name__ == "__main__":
    main()
