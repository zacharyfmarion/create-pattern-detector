#!/usr/bin/env python3
"""Visualize CPLineNet-V2 boundary-contact heads against dense targets."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.cpline_augmentations import AUGMENT_PROFILES
from src.data.cpline_dataset import CplineFoldDataset, cpline_collate
from src.data.v2_boundary_targets import V2_BOUNDARY_SIDE_NAMES, V2_VERTEX_TYPE_NAMES
from src.models import CPLineNet
from src.models.batchnorm import BATCHNORM_MODES, model_eval_with_batchnorm_mode

VERTEX_COLORS = {
    1: (255, 220, 40),
    2: (20, 220, 120),
    3: (255, 80, 220),
}
SIDE_COLORS = {
    0: (255, 220, 40),
    1: (30, 210, 255),
    2: (220, 80, 255),
    3: (255, 145, 40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--augment-profile", choices=AUGMENT_PROFILES, default="clean")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-edges", type=int, default=120)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batchnorm-mode", choices=BATCHNORM_MODES, default="eval")
    parser.add_argument("--pred-contact-threshold", type=float, default=0.55)
    parser.add_argument("--max-pred-contacts", type=int, default=16)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = CPLineNet(
        backbone=str(config.get("backbone", "tiny")),
        hidden_channels=int(config.get("hidden_channels", 128)),
        v2_heads=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    dataset = CplineFoldDataset(
        manifest,
        split=args.split,
        limit=args.examples,
        max_edges=args.max_edges,
        image_size=args.image_size,
        augment_profile=args.augment_profile,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=cpline_collate)

    rows: list[dict[str, Any]] = []
    with model_eval_with_batchnorm_mode(model, batchnorm_mode=args.batchnorm_mode):
        for batch in loader:
            images = batch["image"].to(device)
            outputs = model(images)
            pred_contact = torch.sigmoid(outputs["boundary_contact_logits"])[0, 0].cpu().numpy()
            pred_vertex = outputs["vertex_type_logits"][0].argmax(dim=0).cpu().numpy()
            pred_side = outputs["boundary_side_logits"][0].argmax(dim=0).cpu().numpy()
            input_image = (batch["image"][0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            target_contact = batch["v2_boundary_contact_heatmap"][0, 0].numpy()
            target_vertex = batch["v2_vertex_type"][0].numpy()
            target_side = batch["v2_boundary_side"][0].numpy()
            target_mask = batch["v2_boundary_mask"][0].numpy()
            pred_mask = _pred_contact_mask(
                pred_contact,
                threshold=args.pred_contact_threshold,
                max_contacts=args.max_pred_contacts,
            )
            rows.append(
                {
                    "id": batch["meta"][0]["id"],
                    "selected_profile": batch["meta"][0]["augmentation"].get("selected_profile"),
                    "input": input_image,
                    "target_contact": target_contact,
                    "pred_contact": pred_contact,
                    "target_vertex": target_vertex,
                    "pred_vertex": pred_vertex,
                    "target_side": target_side,
                    "pred_side": pred_side,
                    "target_mask": target_mask,
                    "pred_mask": pred_mask,
                }
            )

    _write_sheet(rows, output_path=output_path)
    _write_sidecar(
        rows,
        output_path=output_path.with_suffix(".json"),
        checkpoint_path=checkpoint_path,
        manifest=manifest,
        args=args,
        config=config,
    )
    print(f"Saved boundary-head sheet: {output_path}")
    print(f"Saved boundary-head sidecar: {output_path.with_suffix('.json')}")


def _pred_contact_mask(heatmap: np.ndarray, *, threshold: float, max_contacts: int) -> np.ndarray:
    if max_contacts <= 0:
        return np.zeros_like(heatmap, dtype=bool)
    dilated = cv2.dilate(heatmap.astype(np.float32), np.ones((5, 5), dtype=np.float32))
    local_max = (heatmap >= dilated - 1e-6) & (heatmap >= threshold)
    coords = np.argwhere(local_max)
    if len(coords) == 0:
        coords = np.argwhere(heatmap == np.max(heatmap))
    ranked = sorted(coords.tolist(), key=lambda coord: float(heatmap[coord[0], coord[1]]), reverse=True)
    mask = np.zeros_like(heatmap, dtype=bool)
    for row, col in ranked[:max_contacts]:
        mask[int(row), int(col)] = True
    return mask


def _write_sheet(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    columns = [
        "input",
        "target contact",
        "pred contact",
        "target vertex",
        "pred vertex",
        "target side",
        "pred side",
    ]
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(2.7 * len(columns), max(2.5, 2.5 * len(rows))),
        squeeze=False,
    )
    for row_idx, row in enumerate(rows):
        images = [
            row["input"],
            _heat_overlay(row["input"], row["target_contact"], color=(20, 230, 150)),
            _heat_overlay(row["input"], row["pred_contact"], color=(255, 80, 200)),
            _label_overlay(row["input"], row["target_vertex"], VERTEX_COLORS, mask=row["target_vertex"] > 0),
            _label_overlay(row["input"], row["pred_vertex"], VERTEX_COLORS, mask=row["pred_vertex"] > 0),
            _label_overlay(row["input"], row["target_side"], SIDE_COLORS, mask=row["target_mask"]),
            _label_overlay(row["input"], row["pred_side"], SIDE_COLORS, mask=row["pred_mask"]),
        ]
        for col_idx, image in enumerate(images):
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].axis("off")
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(columns[col_idx], fontsize=9)
            if col_idx == 0:
                axes[row_idx, col_idx].text(
                    0.02,
                    0.98,
                    f"{row['selected_profile']}\n{row['id']}",
                    transform=axes[row_idx, col_idx].transAxes,
                    va="top",
                    ha="left",
                    fontsize=6.5,
                    color="black",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
                )
    _add_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heat_overlay(image: np.ndarray, heatmap: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    base = image.copy().astype(np.float32)
    heat = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    color_arr = np.array(color, dtype=np.float32)
    alpha = (0.15 + 0.75 * heat)[..., None]
    mask = heat > 0.02
    base[mask] = (1.0 - alpha[mask]) * base[mask] + alpha[mask] * color_arr
    return base.clip(0, 255).astype(np.uint8)


def _label_overlay(
    image: np.ndarray,
    labels: np.ndarray,
    colors: dict[int, tuple[int, int, int]],
    *,
    mask: np.ndarray,
) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    kernel = np.ones((5, 5), dtype=np.uint8)
    for label, color in colors.items():
        label_seed = mask & (labels == label)
        label_mask = cv2.dilate(label_seed.astype(np.uint8), kernel) > 0
        if np.any(label_mask):
            overlay[label_mask] = 0.25 * overlay[label_mask] + 0.75 * np.array(color, dtype=np.float32)
    return overlay.clip(0, 255).astype(np.uint8)


def _add_legend(fig: plt.Figure) -> None:
    vertex_text = ", ".join(
        f"{idx}={name}" for idx, name in sorted(V2_VERTEX_TYPE_NAMES.items()) if idx != 0
    )
    side_text = ", ".join(f"{idx}={name}" for idx, name in sorted(V2_BOUNDARY_SIDE_NAMES.items()))
    fig.text(0.01, 0.01, f"vertex: {vertex_text}    side: {side_text}", fontsize=8)


def _write_sidecar(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    checkpoint_path: Path,
    manifest: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> None:
    payload = {
        "checkpoint": checkpoint_path.as_posix(),
        "manifest": manifest.as_posix(),
        "augment_profile": args.augment_profile,
        "image_size": args.image_size,
        "split": args.split,
        "checkpoint_config": config,
        "rows": [
            {
                "id": row["id"],
                "selected_profile": row["selected_profile"],
                "target_contact_count": int(np.count_nonzero(row["target_mask"])),
                "pred_contact_count": int(np.count_nonzero(row["pred_mask"])),
                "pred_contact_max": float(np.max(row["pred_contact"])),
                "pred_contact_mean": float(np.mean(row["pred_contact"])),
            }
            for row in rows
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
