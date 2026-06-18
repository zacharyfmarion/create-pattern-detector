#!/usr/bin/env python3
"""Fail-fast checks for CPLineNet training run_config.json files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    key, expected = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"expected non-empty KEY in {value!r}")
    return key, expected


def lookup(config: dict[str, Any], key: str) -> Any:
    current: Any = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--expect-str", action="append", type=parse_key_value, default=[])
    parser.add_argument("--expect-float", action="append", type=parse_key_value, default=[])
    parser.add_argument("--expect-suffix", action="append", type=parse_key_value, default=[])
    args = parser.parse_args()

    config_path = args.run_config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    for key, expected in args.expect_str:
        try:
            actual = lookup(config, key)
        except KeyError:
            errors.append(f"{key}: missing, expected {expected!r}")
            continue
        if str(actual) != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    for key, expected_raw in args.expect_float:
        try:
            expected = float(expected_raw)
        except ValueError:
            errors.append(f"{key}: expected float value is invalid: {expected_raw!r}")
            continue
        try:
            actual = lookup(config, key)
            actual_float = float(actual)
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key}: missing or non-numeric, expected {expected:g}")
            continue
        if not math.isclose(actual_float, expected, rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"{key}: expected {expected:g}, got {actual!r}")

    for key, expected_suffix in args.expect_suffix:
        try:
            actual = lookup(config, key)
        except KeyError:
            errors.append(f"{key}: missing, expected suffix {expected_suffix!r}")
            continue
        if not str(actual).endswith(expected_suffix):
            errors.append(f"{key}: expected suffix {expected_suffix!r}, got {actual!r}")

    if errors:
        print(f"run_config verification failed: {config_path}")
        for error in errors:
            print(f"- {error}")
        return 2

    print(f"run_config verification ok: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
