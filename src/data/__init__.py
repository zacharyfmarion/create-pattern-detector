"""Data loading and processing utilities."""

__all__ = [
    "FOLDParser",
    "CreasePattern",
    "GroundTruthGenerator",
    "CreasePatternDataset",
    "CplineFoldDataset",
    "AUGMENT_PROFILES",
    "render_augmented_cpline_sample",
]


def __getattr__(name: str):
    """Lazy-load optional-heavy data helpers only when requested."""
    if name in {"FOLDParser", "CreasePattern"}:
        from .fold_parser import CreasePattern, FOLDParser

        return {"FOLDParser": FOLDParser, "CreasePattern": CreasePattern}[name]
    if name == "GroundTruthGenerator":
        from .annotations import GroundTruthGenerator

        return GroundTruthGenerator
    if name == "CreasePatternDataset":
        from .dataset import CreasePatternDataset

        return CreasePatternDataset
    if name == "CplineFoldDataset":
        from .cpline_dataset import CplineFoldDataset

        return CplineFoldDataset
    if name in {"AUGMENT_PROFILES", "render_augmented_cpline_sample"}:
        from . import cpline_augmentations

        return getattr(cpline_augmentations, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
