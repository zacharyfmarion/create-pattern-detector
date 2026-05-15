"""Data loading and processing utilities."""

from .fold_parser import FOLDParser, CreasePattern

__all__ = [
    "FOLDParser",
    "CreasePattern",
    "GroundTruthGenerator",
    "CreasePatternDataset",
]


def __getattr__(name: str):
    """Lazy-load OpenCV-dependent data helpers only when requested."""
    if name == "GroundTruthGenerator":
        from .annotations import GroundTruthGenerator

        return GroundTruthGenerator
    if name == "CreasePatternDataset":
        from .dataset import CreasePatternDataset

        return CreasePatternDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
