"""Label and loss tests for the close-pair junction recovery recipe.

See docs/v2-close-pair-junction-recovery.md. The failure being fixed: GT vertex
pairs closer than ~8px fuse into a single junction-heatmap blob (sigma ~3.33px
at 1024) and the legacy sub-pixel offsets carry no information to split them.
"""

from __future__ import annotations

import numpy as np
import torch

from src.data.cpline_augmentations import _junction_offsets
from src.models.losses.cpline_loss import _penalty_reduced_focal_loss
from src.vectorization.evidence import _add_gaussian, _add_impulse


def _pair_heatmap(sigma: float, distance: float, size: int = 64) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    heatmap = np.zeros((size, size), dtype=np.float32)
    a = (size / 2 - distance / 2, size / 2)
    b = (size / 2 + distance / 2, size / 2)
    for center in (a, b):
        point = np.array(center, dtype=np.float32)
        _add_gaussian(heatmap, point, sigma=sigma, radius=4.0)
        _add_impulse(heatmap, point)
    return heatmap, a, b


def test_sigma_1_5_preserves_dip_between_close_pair() -> None:
    heatmap, a, b = _pair_heatmap(sigma=1.5, distance=5.0)
    mid = heatmap[int(round(a[1])), int(round((a[0] + b[0]) / 2))]
    peak_a = heatmap[int(round(a[1])), int(round(a[0]))]
    peak_b = heatmap[int(round(b[1])), int(round(b[0]))]
    assert peak_a == 1.0 and peak_b == 1.0, "impulses must pin both anchors at 1.0"
    assert mid < 0.5, f"sigma 1.5 must keep a real dip between 5px peaks, got {mid:.3f}"


def test_legacy_sigma_fuses_close_pair_dip() -> None:
    # Documents the failure mode: at the legacy 1024px sigma (~3.33) the dip
    # between 5px-apart peaks nearly vanishes in the label itself.
    heatmap, a, b = _pair_heatmap(sigma=3.33, distance=5.0)
    mid = heatmap[int(round(a[1])), int(round((a[0] + b[0]) / 2))]
    assert mid > 0.7


def test_radius_offsets_are_bimodal_over_close_pair() -> None:
    size = 64
    vertices = np.array([[29.5, 32.0], [34.5, 32.0]], dtype=np.float32)
    edges = np.array([[0, 1]], dtype=np.int64)
    radius = 3.0
    offset, mask = _junction_offsets(vertices, edges, size, radius_px=radius)

    assert mask.sum() > 2, "radius mode must supervise whole disks, not anchors"
    assert np.all(np.abs(offset) <= 1.0 + 1e-6), "targets must be radius-normalized"

    # A pixel on the left vertex's side points right toward x=29.5 (positive dx
    # from x=28), and one on the right side points toward x=34.5.
    left_dx = offset[32, 28, 0] * radius
    right_dx = offset[32, 36, 0] * radius
    assert mask[32, 28] and mask[32, 36]
    assert np.isclose(left_dx, 29.5 - 28.0, atol=1e-4)
    assert np.isclose(right_dx, 34.5 - 36.0, atol=1e-4)

    # Nearest-vertex assignment: the pixel between the two vertices but closer
    # to the right one must point at the right vertex.
    between_dx = offset[32, 33, 0] * radius
    assert np.isclose(between_dx, 34.5 - 33.0, atol=1e-4)


def test_radius_zero_matches_legacy_subpixel_offsets() -> None:
    size = 32
    vertices = np.array([[10.3, 12.8]], dtype=np.float32)
    edges = np.array([[0, 0]], dtype=np.int64)
    offset, mask = _junction_offsets(vertices, edges, size, radius_px=0.0)
    assert mask.sum() == 1
    assert mask[13, 10]
    assert np.isclose(offset[13, 10, 0], 0.3, atol=1e-5)
    assert np.isclose(offset[13, 10, 1], -0.2, atol=1e-5)


def test_focal_loss_penalizes_dip_fill_in() -> None:
    # One positive anchor, one dip pixel (label 0.4). Confidently predicting
    # 0.99 in the dip must cost more than predicting the label value there.
    target = torch.tensor([[[[1.0, 0.4]]]])
    logits_fill = torch.tensor([[[[6.0, 6.0]]]])  # plateau: dip filled in
    logits_dip = torch.tensor([[[[6.0, -0.4]]]])  # dip preserved
    fill = _penalty_reduced_focal_loss(logits_fill, target, alpha=2.0, beta=4.0)
    keep = _penalty_reduced_focal_loss(logits_dip, target, alpha=2.0, beta=4.0)
    assert torch.isfinite(fill) and torch.isfinite(keep)
    assert fill > keep, "filling the dip must cost more than preserving it"


def test_focal_loss_zero_when_perfect() -> None:
    target = torch.tensor([[[[1.0, 0.0]]]])
    logits = torch.tensor([[[[20.0, -20.0]]]])
    loss = _penalty_reduced_focal_loss(logits, target, alpha=2.0, beta=4.0)
    assert loss.item() < 1e-4


def _load_eval_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent.parent / "scripts" / "evals" / "eval_close_pairs.py"
    spec = importlib.util.spec_from_file_location("eval_close_pairs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offset_cluster_decode_splits_fused_blob() -> None:
    eval_mod = _load_eval_module()
    size = 32
    radius = 3.0
    a = np.array([13.5, 16.0], dtype=np.float32)
    b = np.array([18.5, 16.0], dtype=np.float32)
    # Fused blob the way the model actually emits it: one wide bell centered
    # between the pair, a single local maximum, no information in the heatmap
    # about there being two vertices.
    mid = (a + b) / 2.0
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    probs = 0.99 * np.exp(
        -((grid_x - mid[0]) ** 2 + (grid_y - mid[1]) ** 2) / (2.0 * 3.33**2)
    ).astype(np.float32)
    # Radius-normalized offsets pointing to the nearest of a/b, supervised
    # over the whole vote-collection range.
    offsets = np.zeros((2, size, size), dtype=np.float32)
    ys, xs = np.where(probs >= 0.25)
    for y, x in zip(ys, xs):
        target = a if np.hypot(x - a[0], y - a[1]) <= np.hypot(x - b[0], y - b[1]) else b
        offsets[0, y, x] = (target[0] - x) / radius
        offsets[1, y, x] = (target[1] - y) / radius

    clusters = eval_mod.decode_offset_clusters(probs, offsets, offset_scale=radius)
    assert len(clusters) == 2, f"expected the blob to split into 2 vertices, got {len(clusters)}"
    dist_a = min(np.hypot(*(c - a)) for c in clusters)
    dist_b = min(np.hypot(*(c - b)) for c in clusters)
    assert dist_a < 0.75 and dist_b < 0.75

    peaks = eval_mod.decode_peaks(probs, offsets, offset_scale=radius)
    assert len(peaks) <= 1, "legacy peak decode cannot split the plateau (documents the gap)"
