"""BatchNorm helpers for small-batch CPLine inference/evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


BATCHNORM_MODES = ("eval", "batch-stats")


@contextmanager
def model_eval_with_batchnorm_mode(
    model: torch.nn.Module,
    *,
    batchnorm_mode: str = "eval",
) -> Iterator[None]:
    """Put a model in eval mode, optionally using per-batch BatchNorm stats.

    `batch-stats` keeps dropout and other modules in eval mode, but switches
    BatchNorm modules to train mode with zero momentum. That makes BatchNorm use
    current image/batch statistics without mutating running_mean/running_var.
    """
    if batchnorm_mode not in BATCHNORM_MODES:
        raise ValueError(f"Unsupported batchnorm_mode: {batchnorm_mode}")

    was_training = model.training
    batchnorm_states: list[tuple[torch.nn.modules.batchnorm._BatchNorm, bool, float | None]] = []
    model.eval()
    if batchnorm_mode == "batch-stats":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                batchnorm_states.append((module, module.training, module.momentum))
                module.train()
                module.momentum = 0.0

    try:
        yield
    finally:
        model.train(was_training)
        for module, training, momentum in batchnorm_states:
            module.train(training)
            module.momentum = momentum
