"""Local SEARCH-22.5 (ExplOri 22.5) dataset adapter.

ExplOri 22.5 (https://225.designorigami.net), a.k.a. SEARCH-22.5, is a database of
mathematically exact, flat-foldable crease patterns built on the 22.5-degree grid.
Each pattern satisfies Maekawa/Kawasaki by construction and ships with valid M/V/B
assignments, which makes it a high-quality, distinct distribution for training the
detector alongside the TreeMaker and Rabbit Ear synthetic sources.

This adapter is STRICTLY OFFLINE: it consumes FOLD files reconstructed from the
author-provided ``tilings_{N}_{sym}.db`` SQLite files by ``db_to_fold.py`` in the
SEARCH-22.5 repo. Never fetch from the public site — it is served from a single
machine in the author's home.
"""

from __future__ import annotations

__all__ = [
    "Combo",
    "build_search225_dataset_from_folds",
]


def __getattr__(name: str):
    if name in {"Combo", "build_search225_dataset_from_folds"}:
        from .dataset_builder import Combo, build_search225_dataset_from_folds

        return {
            "Combo": Combo,
            "build_search225_dataset_from_folds": build_search225_dataset_from_folds,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
