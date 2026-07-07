"""High-resolution crop model for VertexRefinerV1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.vertex_refiner_contract import (
    AUXILIARY_CPLINE_CHANNEL_INDICES,
    BOUNDARY_SIDE_NAMES,
    CROP_SIZE_PX,
    INCIDENT_RAY_BINS,
    INPUT_CHANNEL_COUNT,
    V2_INPUT_CHANNEL_COUNT,
    V3_INPUT_CHANNEL_COUNT,
    VERTEX_KIND_NAMES,
)


@dataclass(frozen=True)
class DecodedVertex:
    x: float
    y: float
    score: float
    kind_id: int
    kind: str
    degree_class: int
    degree: int
    ray_bins: tuple[int, ...]
    boundary_side_id: int | None = field(default=None, kw_only=True)
    boundary_side: str | None = field(default=None, kw_only=True)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvNormAct(in_channels, out_channels)
        self.fuse = ConvNormAct(out_channels + skip_channels, out_channels)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.residual(self.fuse(x))


class VertexRefinerV1(nn.Module):
    """Small full-resolution U-Net over 96x96 vertex proposal crops."""

    def __init__(
        self,
        *,
        input_channels: int = INPUT_CHANNEL_COUNT,
        base_channels: int = 48,
        crop_size: int = CROP_SIZE_PX,
        offset_limit_px: float = 4.0,
    ) -> None:
        super().__init__()
        if input_channels != INPUT_CHANNEL_COUNT:
            raise ValueError(f"VertexRefinerV1 expects {INPUT_CHANNEL_COUNT} input channels")
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.crop_size = int(crop_size)
        self.offset_limit_px = float(offset_limit_px)

        c1 = self.base_channels
        c2 = self.base_channels * 2
        c3 = self.base_channels * 4

        self.stem = ConvNormAct(input_channels, c1)
        self.enc0 = ResidualBlock(c1)
        self.down1 = ConvNormAct(c1, c2, stride=2)
        self.enc1 = ResidualBlock(c2)
        self.down2 = ConvNormAct(c2, c3, stride=2)
        self.bottleneck = nn.Sequential(ResidualBlock(c3), ResidualBlock(c3))
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, max(c1, 64 if base_channels >= 48 else c1))
        decoder_channels = max(c1, 64 if base_channels >= 48 else c1)
        self.decoder = nn.Sequential(ResidualBlock(decoder_channels), ResidualBlock(decoder_channels))

        self.vertex_heatmap_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.vertex_offset_head = nn.Conv2d(decoder_channels, 2, kernel_size=1)
        self.vertex_kind_head = nn.Conv2d(decoder_channels, len(VERTEX_KIND_NAMES), kernel_size=1)
        self.degree_head = nn.Conv2d(decoder_channels, 9, kernel_size=1)
        self.incident_ray_head = nn.Conv2d(decoder_channels, INCIDENT_RAY_BINS, kernel_size=1)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        auxiliary_dropout_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected input shape Bx{self.input_channels}xHxW, got {tuple(inputs.shape)}"
            )
        x = apply_auxiliary_channel_dropout(
            inputs,
            p=auxiliary_dropout_p,
            training=self.training,
        )
        skip0 = self.enc0(self.stem(x))
        skip1 = self.enc1(self.down1(skip0))
        x = self.bottleneck(self.down2(skip1))
        x = self.up1(x, skip1)
        x = self.up0(x, skip0)
        x = self.decoder(x)
        return {
            "vertex_heatmap": self.vertex_heatmap_head(x),
            "vertex_offset": torch.tanh(self.vertex_offset_head(x)) * self.offset_limit_px,
            "vertex_kind": self.vertex_kind_head(x),
            "degree": self.degree_head(x),
            "incident_rays": self.incident_ray_head(x),
        }

    def onnx_outputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.forward(inputs)
        return (
            outputs["vertex_heatmap"],
            outputs["vertex_offset"],
            outputs["vertex_kind"],
            outputs["degree"],
            outputs["incident_rays"],
        )


class _FrameAwareVertexRefiner(nn.Module):
    """Shared frame-aware crop model body for V2/V3 junction refinement."""

    def __init__(
        self,
        *,
        input_channels: int,
        expected_input_channels: int,
        model_name: str,
        base_channels: int = 48,
        crop_size: int = CROP_SIZE_PX,
        offset_limit_px: float = 4.0,
    ) -> None:
        super().__init__()
        if input_channels != expected_input_channels:
            raise ValueError(f"{model_name} expects {expected_input_channels} input channels")
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.crop_size = int(crop_size)
        self.offset_limit_px = float(offset_limit_px)

        c1 = self.base_channels
        c2 = self.base_channels * 2
        c3 = self.base_channels * 4

        self.stem = ConvNormAct(input_channels, c1)
        self.enc0 = ResidualBlock(c1)
        self.down1 = ConvNormAct(c1, c2, stride=2)
        self.enc1 = ResidualBlock(c2)
        self.down2 = ConvNormAct(c2, c3, stride=2)
        self.bottleneck = nn.Sequential(ResidualBlock(c3), ResidualBlock(c3))
        self.up1 = UpBlock(c3, c2, c2)
        self.up0 = UpBlock(c2, c1, max(c1, 64 if base_channels >= 48 else c1))
        decoder_channels = max(c1, 64 if base_channels >= 48 else c1)
        self.decoder = nn.Sequential(ResidualBlock(decoder_channels), ResidualBlock(decoder_channels))

        self.vertex_heatmap_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.boundary_contact_heatmap_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.vertex_offset_head = nn.Conv2d(decoder_channels, 2, kernel_size=1)
        self.vertex_kind_head = nn.Conv2d(decoder_channels, len(VERTEX_KIND_NAMES), kernel_size=1)
        self.degree_head = nn.Conv2d(decoder_channels, 9, kernel_size=1)
        self.incident_ray_head = nn.Conv2d(decoder_channels, INCIDENT_RAY_BINS, kernel_size=1)
        self.boundary_side_head = nn.Conv2d(decoder_channels, len(BOUNDARY_SIDE_NAMES), kernel_size=1)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        auxiliary_dropout_p: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        del auxiliary_dropout_p
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected input shape Bx{self.input_channels}xHxW, got {tuple(inputs.shape)}"
            )
        skip0 = self.enc0(self.stem(inputs))
        skip1 = self.enc1(self.down1(skip0))
        x = self.bottleneck(self.down2(skip1))
        x = self.up1(x, skip1)
        x = self.up0(x, skip0)
        x = self.decoder(x)
        return {
            "vertex_heatmap": self.vertex_heatmap_head(x),
            "boundary_contact_heatmap": self.boundary_contact_heatmap_head(x),
            "vertex_offset": torch.tanh(self.vertex_offset_head(x)) * self.offset_limit_px,
            "vertex_kind": self.vertex_kind_head(x),
            "degree": self.degree_head(x),
            "incident_rays": self.incident_ray_head(x),
            "boundary_side": self.boundary_side_head(x),
        }

    def onnx_outputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.forward(inputs)
        return (
            outputs["vertex_heatmap"],
            outputs["vertex_offset"],
            outputs["vertex_kind"],
            outputs["degree"],
            outputs["incident_rays"],
            outputs["boundary_contact_heatmap"],
            outputs["boundary_side"],
        )


class VertexRefinerV2(_FrameAwareVertexRefiner):
    """Frame-aware source-only crop model with the legacy source skeleton channel."""

    def __init__(
        self,
        *,
        input_channels: int = V2_INPUT_CHANNEL_COUNT,
        base_channels: int = 48,
        crop_size: int = CROP_SIZE_PX,
        offset_limit_px: float = 4.0,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            expected_input_channels=V2_INPUT_CHANNEL_COUNT,
            model_name="VertexRefinerV2",
            base_channels=base_channels,
            crop_size=crop_size,
            offset_limit_px=offset_limit_px,
        )


class VertexRefinerV3(_FrameAwareVertexRefiner):
    """Frame-aware source-only crop model without a skeleton input channel."""

    def __init__(
        self,
        *,
        input_channels: int = V3_INPUT_CHANNEL_COUNT,
        base_channels: int = 48,
        crop_size: int = CROP_SIZE_PX,
        offset_limit_px: float = 4.0,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            expected_input_channels=V3_INPUT_CHANNEL_COUNT,
            model_name="VertexRefinerV3",
            base_channels=base_channels,
            crop_size=crop_size,
            offset_limit_px=offset_limit_px,
        )


def apply_auxiliary_channel_dropout(
    inputs: torch.Tensor,
    *,
    p: float,
    training: bool,
) -> torch.Tensor:
    """Randomly zero CPLineNet auxiliary channels 3..5 per crop."""
    if not training or p <= 0.0:
        return inputs
    if p >= 1.0:
        result = inputs.clone()
        result[:, AUXILIARY_CPLINE_CHANNEL_INDICES, :, :] = 0.0
        return result
    keep = torch.rand((inputs.shape[0], 1, 1, 1), device=inputs.device) >= p
    result = inputs.clone()
    result[:, AUXILIARY_CPLINE_CHANNEL_INDICES, :, :] = (
        result[:, AUXILIARY_CPLINE_CHANNEL_INDICES, :, :] * keep.to(inputs.dtype)
    )
    return result


def decode_vertex_refiner_batch(
    outputs: dict[str, torch.Tensor],
    *,
    crop_origins_xy: list[tuple[float, float]] | None = None,
    square_frames: list[Any | None] | None = None,
    heatmap_threshold: float = 0.25,
    boundary_heatmap_threshold: float | None = None,
    nms_radius_px: int = 2,
    ray_threshold: float = 0.5,
) -> list[list[DecodedVertex]]:
    heatmap_logits = outputs["vertex_heatmap"].detach()
    if heatmap_logits.ndim != 4:
        raise ValueError("vertex_heatmap output must have shape Bx1xHxW")
    batch_size = heatmap_logits.shape[0]
    origins = crop_origins_xy or [(0.0, 0.0)] * batch_size
    if len(origins) != batch_size:
        raise ValueError("crop_origins_xy length must match batch size")
    frames = square_frames or [None] * batch_size
    if len(frames) != batch_size:
        raise ValueError("square_frames length must match batch size")
    return [
        decode_vertex_refiner_outputs(
            {key: value[index : index + 1] for key, value in outputs.items()},
            crop_origin_xy=origins[index],
            square_frame=frames[index],
            heatmap_threshold=heatmap_threshold,
            boundary_heatmap_threshold=boundary_heatmap_threshold,
            nms_radius_px=nms_radius_px,
            ray_threshold=ray_threshold,
        )
        for index in range(batch_size)
    ]


def decode_vertex_refiner_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    crop_origin_xy: tuple[float, float] = (0.0, 0.0),
    square_frame: Any | None = None,
    heatmap_threshold: float = 0.25,
    boundary_heatmap_threshold: float | None = None,
    nms_radius_px: int = 2,
    ray_threshold: float = 0.5,
) -> list[DecodedVertex]:
    """Decode one crop's VertexRefinerV1 outputs into vertex candidates."""
    heatmap = torch.sigmoid(outputs["vertex_heatmap"][0, 0])
    peaks = _peak_entries(
        heatmap,
        threshold=float(heatmap_threshold),
        nms_radius_px=nms_radius_px,
        boundary_candidate=False,
    )
    if "boundary_contact_heatmap" in outputs:
        boundary_heatmap = torch.sigmoid(outputs["boundary_contact_heatmap"][0, 0])
        peaks.extend(
            _peak_entries(
                boundary_heatmap,
                threshold=float(
                    heatmap_threshold
                    if boundary_heatmap_threshold is None
                    else boundary_heatmap_threshold
                ),
                nms_radius_px=nms_radius_px,
                boundary_candidate=True,
            )
        )
    peaks = _dedupe_peak_entries(peaks)
    if not peaks:
        return []
    offsets = outputs["vertex_offset"][0]
    kind_probs = torch.softmax(outputs["vertex_kind"][0], dim=0)
    degree_probs = torch.softmax(outputs["degree"][0], dim=0)
    ray_probs = torch.sigmoid(outputs["incident_rays"][0])
    side_probs = (
        torch.softmax(outputs["boundary_side"][0], dim=0)
        if "boundary_side" in outputs
        else None
    )
    origin_x, origin_y = crop_origin_xy
    decoded: list[DecodedVertex] = []
    for score, row, col, boundary_candidate in sorted(
        peaks,
        key=lambda item: (-float(item[0]), int(item[1]), int(item[2])),
    ):
        dx = float(offsets[0, row, col].item())
        dy = float(offsets[1, row, col].item())
        kind_id = int(torch.argmax(kind_probs[:, row, col]).item())
        if boundary_candidate and VERTEX_KIND_NAMES[kind_id] in {"background", "interior_junction"}:
            kind_id = VERTEX_KIND_NAMES.index("boundary_contact")
        degree_class = int(torch.argmax(degree_probs[:, row, col]).item())
        active_rays = tuple(
            int(index)
            for index in torch.where(ray_probs[:, row, col] >= float(ray_threshold))[0].tolist()
        )
        boundary_side_id = (
            int(torch.argmax(side_probs[:, row, col]).item()) if side_probs is not None else None
        )
        boundary_side = (
            BOUNDARY_SIDE_NAMES[boundary_side_id] if boundary_side_id is not None else None
        )
        x = float(origin_x) + float(col) + dx
        y = float(origin_y) + float(row) + dy
        if square_frame is not None and (
            boundary_candidate or VERTEX_KIND_NAMES[kind_id] == "boundary_contact"
        ):
            if boundary_side is None:
                boundary_side = _nearest_frame_side(x, y, square_frame)
                boundary_side_id = BOUNDARY_SIDE_NAMES.index(boundary_side)
            x, y = _snap_xy_to_frame(x, y, square_frame, boundary_side)
        decoded.append(
            DecodedVertex(
                x=x,
                y=y,
                score=float(score),
                kind_id=kind_id,
                kind=VERTEX_KIND_NAMES[kind_id],
                degree_class=degree_class,
                degree=degree_class,
                ray_bins=active_rays,
                boundary_side_id=boundary_side_id,
                boundary_side=boundary_side,
            )
        )
    return decoded


def _peak_entries(
    heatmap: torch.Tensor,
    *,
    threshold: float,
    nms_radius_px: int,
    boundary_candidate: bool,
) -> list[tuple[float, int, int, bool]]:
    kernel = int(nms_radius_px) * 2 + 1
    pooled = F.max_pool2d(
        heatmap[None, None],
        kernel_size=kernel,
        stride=1,
        padding=int(nms_radius_px),
    )[0, 0]
    peak_mask = (heatmap >= float(threshold)) & (heatmap >= pooled - 1e-6)
    rows, cols = torch.where(peak_mask)
    return [
        (float(heatmap[row, col].item()), int(row.item()), int(col.item()), boundary_candidate)
        for row, col in zip(rows, cols, strict=True)
    ]


def _dedupe_peak_entries(
    peaks: list[tuple[float, int, int, bool]],
) -> list[tuple[float, int, int, bool]]:
    by_cell: dict[tuple[int, int], tuple[float, int, int, bool]] = {}
    for score, row, col, boundary_candidate in peaks:
        key = (int(row), int(col))
        previous = by_cell.get(key)
        if previous is None:
            by_cell[key] = (score, row, col, boundary_candidate)
            continue
        by_cell[key] = (
            max(float(score), float(previous[0])),
            row,
            col,
            bool(boundary_candidate or previous[3]),
        )
    return list(by_cell.values())


def _nearest_frame_side(x: float, y: float, frame: Any) -> str:
    distances = {
        "top": abs(float(y) - float(frame.y_min)),
        "right": abs(float(x) - float(frame.x_max)),
        "bottom": abs(float(y) - float(frame.y_max)),
        "left": abs(float(x) - float(frame.x_min)),
    }
    return min(distances.items(), key=lambda item: item[1])[0]


def _snap_xy_to_frame(x: float, y: float, frame: Any, side: str) -> tuple[float, float]:
    if side == "top":
        return float(np.clip(x, frame.x_min, frame.x_max)), float(frame.y_min)
    if side == "right":
        return float(frame.x_max), float(np.clip(y, frame.y_min, frame.y_max))
    if side == "bottom":
        return float(np.clip(x, frame.x_min, frame.x_max)), float(frame.y_max)
    if side == "left":
        return float(frame.x_min), float(np.clip(y, frame.y_min, frame.y_max))
    return float(x), float(y)


class VertexRefinerOnnxWrapper(nn.Module):
    """Tuple-returning wrapper for stable ONNX output ordering."""

    def __init__(self, model: VertexRefinerV1) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model.onnx_outputs(inputs)


def vertex_refiner_parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def vertex_refiner_output_shapes(outputs: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in outputs.items()}


def summarize_decoded_vertices(vertices: list[DecodedVertex]) -> list[dict[str, Any]]:
    return [
        {
            "x": vertex.x,
            "y": vertex.y,
            "score": vertex.score,
            "kind": vertex.kind,
            "degree": vertex.degree,
            "ray_bins": list(vertex.ray_bins),
        }
        for vertex in vertices
    ]
