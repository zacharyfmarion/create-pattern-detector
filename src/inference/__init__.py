"""Production inference utilities for Phase 5."""

from .pipeline import (
    CPDetectPipeline,
    InferenceConfig,
    InferenceResult,
    build_stage4_builder,
    load_checkpoint_manifest,
    select_device,
)
from .rectifier import AlphaMattePolicy, RectificationResult, SquareRectifier

__all__ = [
    "AlphaMattePolicy",
    "CPDetectPipeline",
    "InferenceConfig",
    "InferenceResult",
    "RectificationResult",
    "SquareRectifier",
    "build_stage4_builder",
    "load_checkpoint_manifest",
    "select_device",
]
