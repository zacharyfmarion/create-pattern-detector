#!/usr/bin/env python3
"""Build a SEARCH-22.5 (ExplOri 22.5) dataset root from local FOLD files.

STRICTLY OFFLINE: consumes FOLD files reconstructed from the author-provided
``tilings_{N}_{sym}.db`` SQLite files. Never fetch from the public site — it is
served from a single machine in the author's home.

Workflow:

  # 1. Convert the local .db files in the SEARCH-22.5 repo (its venv has the
  #    compiled math225_core extension):
  cd ~/Documents/code/SEARCH-22.5
  for db in ~/Documents/open\\ source/origami-designer/explori_db/*.db; do
    .venv/bin/python db_to_fold.py "$db" --out /tmp/search225_staging
  done

  # 2. Ingest the staging dir into a standard dataset root:
  python scripts/data/build_search225_dataset.py \\
      --folds /tmp/search225_staging \\
      --out ~/Documents/datasets/create-pattern-detector/synthetic/search225_v1

  # 3. Merge into a new training mix alongside the existing sources:
  python scripts/data/build_synthetic_training_mix.py --recompute-splits \\
      --out ~/Documents/datasets/create-pattern-detector/synthetic/cp_training_mix_v4 \\
      ~/Documents/datasets/create-pattern-detector/synthetic/treemaker_tree_v1 \\
      ~/Documents/datasets/create-pattern-detector/synthetic/rabbit_ear_fold_program_v1 \\
      ~/Documents/datasets/create-pattern-detector/synthetic/search225_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.search225.dataset_builder import (  # noqa: E402
    build_search225_dataset_from_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--folds",
        type=Path,
        nargs="+",
        required=True,
        help="Dir(s) of search225_{N}{sym}{id}.fold files from db_to_fold.py",
    )
    parser.add_argument("--out", type=Path, required=True, help="Dataset root to create")
    parser.add_argument("--dataset-name", default="search225_v1", help="Sample id prefix")
    args = parser.parse_args()

    summary = build_search225_dataset_from_folds(
        out=args.out,
        fold_dirs=list(args.folds),
        dataset_name=args.dataset_name,
    )
    print()
    print(json.dumps(summary.as_dict(), indent=2))


if __name__ == "__main__":
    main()
