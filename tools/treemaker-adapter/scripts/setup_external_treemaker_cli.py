#!/usr/bin/env python3
"""Build the external TreeMaker headless CLI outside this repository.

The GPL TreeMaker source is intentionally not vendored here. This script clones
the legacy TreeMaker environment into a local cache, builds this repo's thin
JSON wrapper against that source, and prints the TREEMAKER_CLI environment
settings needed by synthetic generation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_URL = "https://github.com/AndrewKvalheim/treemaker.git"
BRANCH = "legacy-environment"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=cwd, check=True)


def replace_once(path: Path, before: str, after: str) -> None:
    text = path.read_text(errors="replace")
    if after in text:
        return
    if before not in text:
        raise RuntimeError(f"expected source snippet not found in {path}: {before!r}")
    path.write_text(text.replace(before, after))


def patch_legacy_sources(source_root: Path) -> None:
    """Apply the minimal modern-Clang fixes needed by the legacy source."""
    tm_array = source_root / "src" / "Source" / "tmModel" / "tmPtrClasses" / "tmArray.h"
    replace_once(tm_array, "  insert(this->begin(), t);", "  this->insert(this->begin(), t);")
    replace_once(tm_array, "  if (find(this->begin(), this->end(), t) == this->end()) push_back(t);", "  if (find(this->begin(), this->end(), t) == this->end()) this->push_back(t);")
    replace_once(tm_array, "  erase(this->begin() + ptrdiff_t(n) - 1);", "  this->erase(this->begin() + ptrdiff_t(n) - 1);")
    replace_once(tm_array, "  erase(remove(this->begin(), this->end(), t), this->end());", "  this->erase(remove(this->begin(), this->end(), t), this->end());")
    replace_once(tm_array, "  insert(this->begin() + ptrdiff_t(n) - 1, t);", "  this->insert(this->begin() + ptrdiff_t(n) - 1, t);")
    replace_once(tm_array, "  erase(this->begin());", "  this->erase(this->begin());")
    replace_once(tm_array, "  erase(this->rbegin());", "  this->erase(this->rbegin());")
    replace_once(tm_array, "  insert(this->end(), aList.begin(), aList.end());", "  this->insert(this->end(), aList.begin(), aList.end());")

    tm_dpptr_array = source_root / "src" / "Source" / "tmModel" / "tmPtrClasses" / "tmDpptrArray.h"
    replace_once(tm_dpptr_array, "  if (!contains(pt)) push_back(pt);", "  if (!this->contains(pt)) this->push_back(pt);")
    replace_once(tm_dpptr_array, "  if (contains(pt)) {", "  if (this->contains(pt)) {")

    tm_tree = source_root / "src" / "Source" / "tmModel" / "tmTreeClasses" / "tmTree.h"
    replace_once(
        tm_tree,
        '#include "tmModel_fwd.h"\n#include "tmPart.h"',
        '#include "tmModel_fwd.h"\n#include "tmTreeCleaner.h"\n#include "tmPart.h"',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path.home() / ".cache" / "cp-detector" / "treemaker-legacy",
        help="Directory for the external GPL checkout and build artifacts.",
    )
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the external root.")
    parser.add_argument("--no-smoke", action="store_true", help="Skip the @optimized smoke test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    wrapper_dir = repo_root / "tools" / "treemaker-adapter" / "headless"
    external_root = args.external_root.expanduser().resolve()
    source_root = external_root / "source"
    build_root = external_root / "build"

    if args.force and external_root.exists():
        shutil.rmtree(external_root)
    external_root.mkdir(parents=True, exist_ok=True)

    if not (source_root / ".git").exists():
        run([
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            BRANCH,
            REPO_URL,
            str(source_root),
        ])
    else:
        run(["git", "fetch", "--depth", "1", "origin", BRANCH], cwd=source_root)
        run(["git", "checkout", BRANCH], cwd=source_root)

    patch_legacy_sources(source_root)

    build_root.mkdir(parents=True, exist_ok=True)
    run([
        "cmake",
        "-S",
        str(wrapper_dir),
        "-B",
        str(build_root),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DTREEMAKER_SOURCE_DIR={source_root / 'src'}",
    ])
    run(["cmake", "--build", str(build_root), "-j", str(os.cpu_count() or 4)])

    cli = build_root / "treemaker-json-cli"
    if not cli.exists():
        raise FileNotFoundError(cli)

    if not args.no_smoke:
        smoke_out = external_root / "smoke-optimized.json"
        run([str(cli), "--in", "@optimized", "--out", str(smoke_out)])
        smoke = json.loads(smoke_out.read_text())
        if not smoke.get("ok") or smoke.get("stats", {}).get("creases", 0) < 50:
            raise RuntimeError(f"TreeMaker smoke did not produce a full CP: {smoke}")

    print(f"export TREEMAKER_CLI={cli}")
    print("export TREEMAKER_CLI_ARGS=--triangulate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
