#!/usr/bin/env python3
"""Compare two Stage 4 eval directories against the preregistered promotion gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

STATUS_QUALITY = {
    "valid": 1.0,
    "repaired": 0.9,
    "ambiguous": 0.55,
    "outside_v1_envelope": 0.35,
    "failed": 0.0,
}
PRIMARY_METRICS = (
    "edge_precision",
    "edge_recall",
    "vertex_precision",
    "vertex_recall",
    "assignment_accuracy",
    "border_f1",
    "structural_validity_rate",
)
TARGET_ID = "rabbit_ear_fold_program_v1-5wk08-000155"
TARGET_PROFILE = "clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-min-delta", type=float, default=0.002)
    parser.add_argument("--max-aggregate-primary-drop", type=float, default=0.01)
    parser.add_argument("--max-profile-score-drop", type=float, default=0.005)
    parser.add_argument("--max-family-score-drop", type=float, default=0.005)
    parser.add_argument("--min-family-files", type=int, default=25)
    parser.add_argument("--target-id", default=TARGET_ID)
    parser.add_argument("--target-profile", default=TARGET_PROFILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_run(args.baseline_dir)
    candidate = load_run(args.candidate_dir)
    comparison = compare_runs(baseline, candidate, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    write_visual_contact_sheet(comparison, args.output_dir / "visual_examples.png")
    write_markdown_report(comparison, args.output_dir / "comparison.md")
    print(json.dumps({"accepted": comparison["gate"]["accepted"], "gate": comparison["gate"]}))


def load_run(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    rows_path = path / "per_sample_metrics.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json: {summary_path}")
    if not rows_path.exists():
        raise SystemExit(f"Missing per_sample_metrics.jsonl: {rows_path}")
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "dir": path,
        "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        "rows": rows,
        "rows_by_key": {row_key(row): row for row in rows},
    }


def compare_runs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    aggregate = compare_block(baseline_summary["aggregate"], candidate_summary["aggregate"])
    by_profile = compare_named_blocks(baseline_summary["by_profile"], candidate_summary["by_profile"])
    by_family = compare_named_blocks(baseline_summary["by_family"], candidate_summary["by_family"])
    by_profile_family = compare_named_blocks(
        baseline_summary.get("by_profile_family", {}),
        candidate_summary.get("by_profile_family", {}),
    )
    sample_deltas = compare_samples(baseline, candidate)
    visual_examples = visual_example_specs(
        baseline,
        candidate,
        sample_deltas,
        target_id=args.target_id,
        target_profile=args.target_profile,
    )
    target = target_report(
        baseline,
        candidate,
        target_id=args.target_id,
        target_profile=args.target_profile,
    )
    gate = evaluate_gate(aggregate, by_profile, by_family, target, args)
    return {
        "baseline_dir": str(baseline["dir"]),
        "candidate_dir": str(candidate["dir"]),
        "gate": gate,
        "aggregate": aggregate,
        "by_profile": by_profile,
        "by_family": by_family,
        "by_profile_family": by_profile_family,
        "target_sample": target,
        "sample_deltas": {
            "largest_improvements": sample_deltas[:10],
            "worst_regressions": sorted(sample_deltas, key=lambda item: item["score_delta"])[:10],
        },
        "visual_examples": visual_examples,
    }


def compare_block(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_score = balanced_score(baseline)
    candidate_score = balanced_score(candidate)
    metrics = sorted({*baseline.keys(), *candidate.keys()} & set(PRIMARY_METRICS))
    metric_deltas = {
        metric: float(candidate.get(metric, 0.0)) - float(baseline.get(metric, 0.0))
        for metric in metrics
    }
    return {
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "score_delta": candidate_score - baseline_score,
        "baseline": compact_metrics(baseline),
        "candidate": compact_metrics(candidate),
        "metric_deltas": metric_deltas,
        "files": int(candidate.get("files", baseline.get("files", 0))),
    }


def compare_named_blocks(
    baseline_blocks: dict[str, dict[str, Any]],
    candidate_blocks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    compared = {}
    for name in sorted(set(baseline_blocks) | set(candidate_blocks)):
        compared[name] = compare_block(baseline_blocks.get(name, {}), candidate_blocks.get(name, {}))
    return compared


def compare_samples(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    deltas = []
    for key, baseline_row in baseline["rows_by_key"].items():
        candidate_row = candidate["rows_by_key"].get(key)
        if candidate_row is None:
            continue
        baseline_score = row_score(baseline_row)
        candidate_score = row_score(candidate_row)
        deltas.append(
            {
                "key": key,
                "id": baseline_row["id"],
                "profile": baseline_row["profile"],
                "sample_index": int(baseline_row["sample_index"]),
                "family": baseline_row.get("family", ""),
                "score_delta": candidate_score - baseline_score,
                "edge_recall_delta": float(candidate_row["edge_recall"])
                - float(baseline_row["edge_recall"]),
                "edge_precision_delta": float(candidate_row["edge_precision"])
                - float(baseline_row["edge_precision"]),
                "assignment_accuracy_delta": float(candidate_row["assignment_accuracy"])
                - float(baseline_row["assignment_accuracy"]),
                "baseline_status": baseline_row["status"],
                "candidate_status": candidate_row["status"],
            }
        )
    return sorted(deltas, key=lambda item: item["score_delta"], reverse=True)


def evaluate_gate(
    aggregate: dict[str, Any],
    by_profile: dict[str, Any],
    by_family: dict[str, Any],
    target: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    failures = []
    if aggregate["score_delta"] < args.score_min_delta:
        failures.append(
            f"aggregate balanced score delta {aggregate['score_delta']:.6f} < {args.score_min_delta:.6f}"
        )
    for metric, delta in aggregate["metric_deltas"].items():
        if delta < -args.max_aggregate_primary_drop:
            failures.append(
                f"aggregate {metric} delta {delta:.6f} < -{args.max_aggregate_primary_drop:.6f}"
            )
    baseline_failed = int(aggregate["baseline"]["status_counts"].get("failed", 0))
    candidate_failed = int(aggregate["candidate"]["status_counts"].get("failed", 0))
    if candidate_failed > baseline_failed:
        failures.append(f"failed count increased from {baseline_failed} to {candidate_failed}")
    for name, block in by_profile.items():
        if block["score_delta"] < -args.max_profile_score_drop:
            failures.append(
                f"profile {name} score delta {block['score_delta']:.6f} < -{args.max_profile_score_drop:.6f}"
            )
    for name, block in by_family.items():
        if block["files"] < args.min_family_files:
            continue
        if block["score_delta"] < -args.max_family_score_drop:
            failures.append(
                f"family {name} score delta {block['score_delta']:.6f} < -{args.max_family_score_drop:.6f}"
            )
    if not target["present"]:
        failures.append(target["message"])
    return {
        "accepted": not failures,
        "failures": failures,
        "thresholds": {
            "score_min_delta": args.score_min_delta,
            "max_aggregate_primary_drop": args.max_aggregate_primary_drop,
            "max_profile_score_drop": args.max_profile_score_drop,
            "max_family_score_drop": args.max_family_score_drop,
            "min_family_files": args.min_family_files,
        },
    }


def target_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    target_id: str,
    target_profile: str,
) -> dict[str, Any]:
    key = None
    for row_key_value, row in baseline["rows_by_key"].items():
        if row["id"] == target_id and row["profile"] == target_profile:
            key = row_key_value
            break
    if key is None:
        return {
            "present": False,
            "message": f"target {target_profile}/{target_id} not present in baseline rows",
        }
    baseline_row = baseline["rows_by_key"][key]
    candidate_row = candidate["rows_by_key"].get(key)
    if candidate_row is None:
        return {
            "present": False,
            "message": f"target {target_profile}/{target_id} not present in candidate rows",
        }
    return {
        "present": True,
        "message": (
            "per-sample metrics are present; exact edge-53/edge-68 recovery is covered by the "
            "checkpoint regression test because eval JSONL does not store edge-level matches"
        ),
        "key": key,
        "baseline": compact_row(baseline_row),
        "candidate": compact_row(candidate_row),
        "deltas": {
            "score": row_score(candidate_row) - row_score(baseline_row),
            "edge_recall": float(candidate_row["edge_recall"]) - float(baseline_row["edge_recall"]),
            "edge_precision": float(candidate_row["edge_precision"]) - float(baseline_row["edge_precision"]),
            "pred_edges": int(candidate_row["pred_edges"]) - int(baseline_row["pred_edges"]),
            "matched_edges": int(candidate_row["matched_edges"]) - int(baseline_row["matched_edges"]),
        },
    }


def visual_example_specs(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    sample_deltas: list[dict[str, Any]],
    *,
    target_id: str,
    target_profile: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    target = next(
        (
            item
            for item in sample_deltas
            if item["id"] == target_id and item["profile"] == target_profile
        ),
        None,
    )
    if target is not None:
        selected.append({"reason": "target", **target})
    selected.extend({"reason": "largest_improvement", **item} for item in sample_deltas[:3])
    selected.extend(
        {"reason": "worst_regression", **item}
        for item in sorted(sample_deltas, key=lambda item: item["score_delta"])[:3]
    )
    deduped = []
    seen = set()
    for item in selected:
        key = item["key"]
        if key in seen:
            continue
        seen.add(key)
        baseline_path = example_path(baseline["dir"], item)
        candidate_path = example_path(candidate["dir"], item)
        deduped.append(
            {
                **item,
                "baseline_image": str(baseline_path) if baseline_path.exists() else None,
                "candidate_image": str(candidate_path) if candidate_path.exists() else None,
            }
        )
    return deduped


def write_visual_contact_sheet(comparison: dict[str, Any], path: Path) -> None:
    pairs = [
        item
        for item in comparison["visual_examples"]
        if item.get("baseline_image") and item.get("candidate_image")
    ]
    if not pairs:
        return
    thumb_w, thumb_h = 420, 276
    label_h = 36
    canvas = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * len(pairs)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_idx, item in enumerate(pairs):
        y = row_idx * (thumb_h + label_h)
        for col, key in enumerate(("baseline_image", "candidate_image")):
            image = Image.open(item[key]).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            x = col * thumb_w + (thumb_w - image.width) // 2
            canvas.paste(image, (x, y + label_h))
        label = (
            f"{item['reason']} | {item['profile']} {item['sample_index']:03d} | "
            f"score delta {item['score_delta']:+.4f}"
        )
        draw.text((8, y + 10), label, fill=(17, 24, 39))
        draw.text((thumb_w + 8, y + 10), "candidate", fill=(17, 24, 39))
    canvas.save(path)


def write_markdown_report(comparison: dict[str, Any], path: Path) -> None:
    gate = comparison["gate"]
    lines = [
        "# Stage 4 Run Comparison",
        "",
        f"Accepted: `{gate['accepted']}`",
        f"Aggregate score delta: `{comparison['aggregate']['score_delta']:.6f}`",
        "",
        "## Gate Failures",
    ]
    lines.extend(f"- {failure}" for failure in gate["failures"] or ["none"])
    lines.extend(["", "## Target Sample", ""])
    lines.append(json.dumps(comparison["target_sample"], indent=2))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_metrics(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_precision": float(block.get("edge_precision", 0.0)),
        "edge_recall": float(block.get("edge_recall", 0.0)),
        "edge_f1": f1(float(block.get("edge_precision", 0.0)), float(block.get("edge_recall", 0.0))),
        "vertex_precision": float(block.get("vertex_precision", 0.0)),
        "vertex_recall": float(block.get("vertex_recall", 0.0)),
        "vertex_f1": f1(
            float(block.get("vertex_precision", 0.0)),
            float(block.get("vertex_recall", 0.0)),
        ),
        "assignment_accuracy": float(block.get("assignment_accuracy", 0.0)),
        "border_f1": float(block.get("border_f1", 0.0)),
        "structural_validity_rate": float(block.get("structural_validity_rate", 0.0)),
        "status_quality": status_quality(block),
        "status_counts": block.get("status_counts", {}),
    }


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": row["status"],
        "edge_precision": float(row["edge_precision"]),
        "edge_recall": float(row["edge_recall"]),
        "matched_edges": int(row["matched_edges"]),
        "pred_edges": int(row["pred_edges"]),
        "gt_edges": int(row["gt_edges"]),
        "assignment_accuracy": float(row["assignment_accuracy"]),
    }


def balanced_score(block: dict[str, Any]) -> float:
    edge_f1 = f1(float(block.get("edge_precision", 0.0)), float(block.get("edge_recall", 0.0)))
    vertex_f1 = f1(float(block.get("vertex_precision", 0.0)), float(block.get("vertex_recall", 0.0)))
    return (
        0.40 * edge_f1
        + 0.15 * vertex_f1
        + 0.15 * float(block.get("assignment_accuracy", 0.0))
        + 0.10 * float(block.get("border_f1", 0.0))
        + 0.10 * float(block.get("structural_validity_rate", 0.0))
        + 0.10 * status_quality(block)
    )


def row_score(row: dict[str, Any]) -> float:
    block = {
        **row,
        "structural_validity_rate": 1.0 if (row.get("structural_validity") or {}).get("valid") else 0.0,
        "status_counts": {row["status"]: 1},
        "files": 1,
    }
    return balanced_score(block)


def status_quality(block: dict[str, Any]) -> float:
    counts = block.get("status_counts", {})
    total = int(block.get("files", 0)) or sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    return sum(STATUS_QUALITY.get(status, 0.0) * int(count) for status, count in counts.items()) / total


def f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def row_key(row: dict[str, Any]) -> str:
    return f"{row['profile']}__{int(row['sample_index']):03d}__{row['id']}"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def example_path(run_dir: Path, item: dict[str, Any]) -> Path:
    return (
        run_dir
        / "examples"
        / f"{item['profile']}_{int(item['sample_index']):03d}_{safe_id(item['id'])}.png"
    )


if __name__ == "__main__":
    main()
