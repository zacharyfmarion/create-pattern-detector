"""Collect exact 22.5-degree flat-foldable crease patterns from ExplOri 22.5.

ExplOri 22.5 (https://225.designorigami.net), a.k.a. SEARCH-22.5, is a database of
mathematically exact, flat-foldable crease patterns built on the 22.5-degree grid.
Each pattern satisfies Maekawa/Kawasaki by construction and ships with valid M/V/B
assignments, which makes it a high-quality, distinct distribution for training the
detector alongside the TreeMaker and Rabbit Ear synthetic sources.

The public site exposes a REST endpoint:

    GET /api/fetch_tiling?id={tiling_id}&N={N}&sym={sym}

This package fetches those tilings (politely, sequentially) and normalizes them into
the project's standard synthetic dataset-root layout so they can be merged with the
existing sources via ``scripts/data/build_synthetic_training_mix.py``.
"""

from __future__ import annotations

__all__ = [
    "ALL_COMBOS",
    "Combo",
    "TilingClient",
    "cp_to_fold_dict",
    "build_search225_dataset",
]


def __getattr__(name: str):
    if name in {"ALL_COMBOS", "Combo", "TilingClient", "cp_to_fold_dict"}:
        from .fetch import ALL_COMBOS, Combo, TilingClient, cp_to_fold_dict

        return {
            "ALL_COMBOS": ALL_COMBOS,
            "Combo": Combo,
            "TilingClient": TilingClient,
            "cp_to_fold_dict": cp_to_fold_dict,
        }[name]
    if name == "build_search225_dataset":
        from .dataset_builder import build_search225_dataset

        return build_search225_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
