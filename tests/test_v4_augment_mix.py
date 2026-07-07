"""Guardrails for the v4 solid-geometry training mix.

The v3 replay mix trained the junction/line heads against full solid-geometry
targets while 19% of samples had every non-border crease dashed (and 27% had
dashes, text, or watermarks) — i.e. junctions labeled at points with no ink.
The v4 mix must contain no geometry-obfuscating profile.
"""

from __future__ import annotations

import numpy as np

from src.data.cpline_augmentations import AUGMENT_MIXES, normalize_augment_profile

OBFUSCATOR_MARKERS = ("dashed", "text", "watermark", "combined", "guide-grid", "junction-dots")


def test_v4_mix_is_registered_and_normalizes():
    assert normalize_augment_profile("v4-solid-geometry-replay") == "v4-solid-geometry-replay"


def test_v4_mix_has_no_obfuscating_profiles():
    mix = AUGMENT_MIXES["v4-solid-geometry-replay"]
    for profile, _, _ in mix:
        assert not any(marker in profile for marker in OBFUSCATOR_MARKERS), profile


def test_v4_mix_weights_sum_to_one():
    total = sum(weight for _, weight, _ in AUGMENT_MIXES["v4-solid-geometry-replay"])
    assert np.isclose(total, 1.0)


def test_v3_mix_obfuscator_share_documented():
    """Pin the measured v3 shares so the v4 rationale stays verifiable."""
    mix = AUGMENT_MIXES["v3-no-guide-grid-replay"]
    total = sum(w for _, w, _ in mix)
    dash = sum(w for p, w, _ in mix if "dashed" in p or "combined" in p) / total
    any_obf = (
        sum(w for p, w, _ in mix if any(m in p for m in ("dashed", "text", "watermark", "combined")))
        / total
    )
    assert np.isclose(dash, 0.19)
    assert np.isclose(any_obf, 0.27)


def test_dense_width_cap_constants():
    from src.data.cpline_augmentations import DENSE_WIDTH_CAP_MIN_EDGES, DENSE_WIDTH_CAP_PX

    # Guardrail for the density-aware stroke-width cap: fat draws on very dense
    # CPs flood the inter-pleat gaps (measured: width 5 at 8.6px pitch leaves
    # only a dot-lattice of background). If these move, re-run the gap check.
    assert DENSE_WIDTH_CAP_MIN_EDGES == 2500
    assert DENSE_WIDTH_CAP_PX == 3
