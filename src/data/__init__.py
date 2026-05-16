"""Data loading and processing utilities."""

__all__ = [
    "FOLDParser",
    "CreasePattern",
    "GroundTruthGenerator",
    "CreasePatternDataset",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
