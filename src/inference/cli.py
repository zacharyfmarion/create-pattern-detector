"""Command-line entrypoint for Phase 5 `cp-detect` inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.inference.pipeline import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CHECKPOINT_MANIFEST,
    CPDetectPipeline,
    InferenceConfig,
    write_batch_summary,
)

DEFAULT_OUTPUT_DIR = Path("outputs/cp-detect")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except NotImplementedError as exc:
        print(f"cp-detect: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"cp-detect: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a rectified origami crease-pattern image and export FOLD.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input image path(s).")
    parser.add_argument(
        "--rectified",
        action="store_true",
        help="Input contains a readable CP or visible CP panel ready for border-based normalization.",
    )
    parser.add_argument("--output", type=Path, help="Single-output .fold path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, help="Single-output report JSON path.")
    parser.add_argument("--debug-dir", type=Path, help="Debug artifact directory.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batchnorm-mode", choices=["batch-stats", "eval"], default="batch-stats")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--alpha-matte", choices=["auto", "white", "black"], default="auto")
    parser.add_argument("--infer-assignments", action="store_true")
    parser.add_argument("--repair-near-endpoint-crossings", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument(
        "--no-verify-checkpoint",
        action="store_true",
        help="Skip local checksum/size verification before loading the checkpoint.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if not args.rectified:
        raise NotImplementedError(
            "Automatic input discovery is not implemented in Phase 5. Use --rectified "
            "for readable CP inputs with a visible CP panel; full photo/document "
            "rectification is a Phase 6 deliverable."
        )

    inputs = [Path(path) for path in args.inputs]
    for path in inputs:
        if path.is_dir():
            raise ValueError(f"Directory inputs are not supported in Phase 5: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
    if len(inputs) > 1 and args.output is not None:
        raise ValueError("--output can only be used with a single input")
    if len(inputs) > 1 and args.report is not None:
        raise ValueError("--report can only be used with a single input")

    config = InferenceConfig(
        checkpoint=args.checkpoint,
        checkpoint_manifest=args.checkpoint_manifest,
        device=args.device,
        threshold=args.threshold,
        batchnorm_mode=args.batchnorm_mode,
        rectified=args.rectified,
        alpha_matte=args.alpha_matte,
        infer_assignments=args.infer_assignments,
        repair_near_endpoint_crossings=args.repair_near_endpoint_crossings,
        include_debug=not args.no_debug,
        verify_checkpoint=not args.no_verify_checkpoint,
    )
    pipeline = CPDetectPipeline(config)

    output_specs = _output_specs(
        inputs,
        output=args.output,
        output_dir=args.output_dir,
        report=args.report,
        debug_dir=args.debug_dir,
        include_debug=not args.no_debug,
    )
    results = []
    for input_path, fold_path, report_path, debug_path in output_specs:
        result = pipeline.detect(
            input_path,
            output_fold=fold_path,
            report_path=report_path,
            debug_dir=debug_path,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "input": str(input_path),
                    "status": result.status,
                    "fold": None if result.output_fold is None else str(result.output_fold),
                    "fold_written": result.report_payload["outputs"]["fold_written"],
                    "report": None if result.report_path is None else str(result.report_path),
                    "debug_dir": None if result.debug_dir is None else str(result.debug_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(results) > 1:
        write_batch_summary(results, args.output_dir / "batch.summary.json", config=config)
    return 1 if any(result.status == "failed" for result in results) else 0


def _output_specs(
    inputs: list[Path],
    *,
    output: Path | None,
    output_dir: Path,
    report: Path | None,
    debug_dir: Path | None,
    include_debug: bool,
) -> list[tuple[Path, Path, Path, Path | None]]:
    if len(inputs) == 1:
        input_path = inputs[0]
        fold_path = output or output_dir / f"{input_path.stem}.fold"
        report_path = report or output_dir / f"{input_path.stem}.report.json"
        debug_path = None
        if include_debug:
            debug_path = debug_dir or output_dir / f"{input_path.stem}.debug"
        return [(input_path, fold_path, report_path, debug_path)]

    seen: dict[str, int] = {}
    specs = []
    for input_path in inputs:
        stem = _unique_stem(input_path.stem, seen)
        fold_path = output_dir / f"{stem}.fold"
        report_path = output_dir / f"{stem}.report.json"
        debug_path = None if not include_debug else (debug_dir or output_dir) / f"{stem}.debug"
        specs.append((input_path, fold_path, report_path, debug_path))
    return specs


def _unique_stem(stem: str, seen: dict[str, int]) -> str:
    count = seen.get(stem, 0) + 1
    seen[stem] = count
    if count == 1:
        return stem
    return f"{stem}_{count}"


if __name__ == "__main__":
    raise SystemExit(main())
