#!/usr/bin/env python3
"""Resolve the current promoted CP detector checkpoint from the tracked pointer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
POINTER_PATH = REPO_ROOT / "artifacts/checkpoints/current-browser-model.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def current_manifest_path() -> Path:
    pointer = load_json(POINTER_PATH)
    return REPO_ROOT / pointer["checkpointManifest"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        choices=["checkpoint", "checkpoint-manifest", "id", "onnx-sha256"],
        default="checkpoint",
    )
    args = parser.parse_args()
    manifest_path = current_manifest_path()
    manifest = load_json(manifest_path)

    if args.field == "checkpoint":
        print(manifest["checkpoint"]["relativePath"])
    elif args.field == "checkpoint-manifest":
        print(manifest_path.relative_to(REPO_ROOT))
    elif args.field == "id":
        print(manifest["id"])
    elif args.field == "onnx-sha256":
        print(manifest["inference"]["downstreamOnnxSha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
