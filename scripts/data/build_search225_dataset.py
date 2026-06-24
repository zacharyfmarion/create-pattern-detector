#!/usr/bin/env python3
"""Build a SEARCH-22.5 (ExplOri 22.5) synthetic dataset root.

Politely (sequentially) samples exact flat-foldable 22.5-degree tilings from
https://225.designorigami.net and writes them in the project's standard dataset-root
layout (folds/, metadata/, raw-manifest.jsonl, recipe.json, qa.json). The result can
be merged with the other synthetic sources via build_synthetic_training_mix.py.

Examples:
  # Curated ~10k, weighted toward higher-N / denser patterns (default)
  python scripts/data/build_search225_dataset.py \
      --out ~/Documents/datasets/create-pattern-detector/synthetic/search225_v1 \
      --count 10000 --weights higher-n

  # Tiny smoke run against a couple of small combos
  python scripts/data/build_search225_dataset.py --out /tmp/search225_smoke \
      --count 20 --combo 3_book 3_diag --delay 0.3

  # Then merge into the training mix:
  python scripts/data/build_synthetic_training_mix.py --recompute-splits \
      --out ~/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v2 \
      ~/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1 \
      ~/Documents/datasets/create-pattern-detector/synthetic/rabbit_ear_fold_program_v1 \
      ~/Documents/datasets/create-pattern-detector/synthetic/search225_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.search225.dataset_builder import (  # noqa: E402
    WEIGHT_PRESETS,
    build_search225_dataset,
)
from src.data.search225.fetch import ALL_COMBOS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True, help="Dataset root to create/update")
    parser.add_argument("--count", type=int, default=10_000, help="Total patterns to sample")
    parser.add_argument(
        "--weights",
        choices=sorted(WEIGHT_PRESETS),
        default="higher-n",
        help="Sampling weight preset across (N, sym) combos (default: higher-n)",
    )
    parser.add_argument(
        "--combo",
        nargs="+",
        metavar="N_sym",
        help=f"Restrict to these combos. Choices: {', '.join(c.key for c in ALL_COMBOS)}",
    )
    parser.add_argument("--dataset-name", default="search225_v1", help="Sample id prefix")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Per-request politeness delay in seconds (sequential; default 0.2)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for id sampling")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing manifest/folds and re-fetch from scratch",
    )
    args = parser.parse_args()

    summary = build_search225_dataset(
        out=args.out,
        target_count=args.count,
        weights_preset=args.weights,
        combo_keys=args.combo,
        dataset_name=args.dataset_name,
        delay=args.delay,
        seed=args.seed,
        resume=not args.no_resume,
    )
    print()
    import json

    print(json.dumps(summary.as_dict(), indent=2))


if __name__ == "__main__":
    main()
