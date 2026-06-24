from __future__ import annotations

from io import BytesIO

import pytest
import torch

from src.models.losses.vertex_refiner_loss import VertexRefinerLoss
from src.models.vertex_refiner import (
    VertexRefinerOnnxWrapper,
    VertexRefinerV1,
    VertexRefinerV2,
    apply_auxiliary_channel_dropout,
    decode_vertex_refiner_outputs,
)
from src.data.vertex_refiner_targets import SquareFrame
from src.models.vertex_refiner_contract import ONNX_OUTPUT_NAMES


def test_vertex_refiner_forward_shapes() -> None:
    model = VertexRefinerV1(base_channels=8)
    outputs = model(torch.randn(2, 8, 96, 96))

    assert outputs["vertex_heatmap"].shape == (2, 1, 96, 96)
    assert outputs["vertex_offset"].shape == (2, 2, 96, 96)
    assert outputs["vertex_kind"].shape == (2, 5, 96, 96)
    assert outputs["degree"].shape == (2, 9, 96, 96)
    assert outputs["incident_rays"].shape == (2, 36, 96, 96)
    assert float(outputs["vertex_offset"].detach().abs().max()) <= model.offset_limit_px + 1e-5


def test_vertex_refiner_v2_forward_shapes() -> None:
    model = VertexRefinerV2(base_channels=8)
    outputs = model(torch.randn(2, 12, 96, 96))

    assert outputs["vertex_heatmap"].shape == (2, 1, 96, 96)
    assert outputs["boundary_contact_heatmap"].shape == (2, 1, 96, 96)
    assert outputs["vertex_offset"].shape == (2, 2, 96, 96)
    assert outputs["vertex_kind"].shape == (2, 5, 96, 96)
    assert outputs["degree"].shape == (2, 9, 96, 96)
    assert outputs["incident_rays"].shape == (2, 36, 96, 96)
    assert outputs["boundary_side"].shape == (2, 4, 96, 96)


def test_auxiliary_channel_dropout_zeroes_only_cpline_channels() -> None:
    inputs = torch.ones(2, 8, 4, 4)
    dropped = apply_auxiliary_channel_dropout(inputs, p=1.0, training=True)

    assert torch.all(dropped[:, :3] == 1.0)
    assert torch.all(dropped[:, 3:6] == 0.0)
    assert torch.all(dropped[:, 6:] == 1.0)
    assert torch.equal(apply_auxiliary_channel_dropout(inputs, p=1.0, training=False), inputs)


def test_vertex_refiner_decode_reads_offsets_classes_degree_and_rays() -> None:
    outputs = {
        "vertex_heatmap": torch.full((1, 1, 96, 96), -8.0),
        "vertex_offset": torch.zeros((1, 2, 96, 96)),
        "vertex_kind": torch.zeros((1, 5, 96, 96)),
        "degree": torch.zeros((1, 9, 96, 96)),
        "incident_rays": torch.full((1, 36, 96, 96), -8.0),
    }
    outputs["vertex_heatmap"][0, 0, 20, 30] = 8.0
    outputs["vertex_offset"][0, :, 20, 30] = torch.tensor([0.25, -0.5])
    outputs["vertex_kind"][0, 2, 20, 30] = 4.0
    outputs["degree"][0, 4, 20, 30] = 4.0
    outputs["incident_rays"][0, [0, 9, 18, 27], 20, 30] = 8.0

    decoded = decode_vertex_refiner_outputs(outputs, crop_origin_xy=(100.0, 200.0))

    assert len(decoded) == 1
    assert decoded[0].x == 130.25
    assert decoded[0].y == 219.5
    assert decoded[0].kind == "boundary_contact"
    assert decoded[0].degree == 4
    assert decoded[0].ray_bins == (0, 9, 18, 27)


def test_vertex_refiner_v2_decode_snaps_boundary_peak_to_frame() -> None:
    outputs = {
        "vertex_heatmap": torch.full((1, 1, 96, 96), -8.0),
        "boundary_contact_heatmap": torch.full((1, 1, 96, 96), -8.0),
        "vertex_offset": torch.zeros((1, 2, 96, 96)),
        "vertex_kind": torch.zeros((1, 5, 96, 96)),
        "degree": torch.zeros((1, 9, 96, 96)),
        "incident_rays": torch.full((1, 36, 96, 96), -8.0),
        "boundary_side": torch.full((1, 4, 96, 96), -8.0),
    }
    outputs["boundary_contact_heatmap"][0, 0, 47, 30] = 8.0
    outputs["vertex_offset"][0, :, 47, 30] = torch.tensor([0.25, 1.7])
    outputs["vertex_kind"][0, 1, 47, 30] = 5.0
    outputs["boundary_side"][0, 0, 47, 30] = 8.0

    decoded = decode_vertex_refiner_outputs(
        outputs,
        crop_origin_xy=(100.0, 200.0),
        square_frame=SquareFrame(100.0, 248.0, 196.0, 344.0),
        heatmap_threshold=0.25,
    )

    assert len(decoded) == 1
    assert decoded[0].kind == "boundary_contact"
    assert decoded[0].boundary_side == "top"
    assert decoded[0].x == 130.25
    assert decoded[0].y == 248.0


@pytest.mark.parametrize(
    ("side", "side_id", "peak_row", "peak_col", "expected_x", "expected_y"),
    [
        ("top", 0, 19, 40, 40.0, 20.0),
        ("right", 1, 40, 77, 76.0, 40.0),
        ("bottom", 2, 77, 40, 40.0, 76.0),
        ("left", 3, 40, 19, 20.0, 40.0),
    ],
)
def test_vertex_refiner_v2_decode_snaps_all_boundary_sides(
    side: str,
    side_id: int,
    peak_row: int,
    peak_col: int,
    expected_x: float,
    expected_y: float,
) -> None:
    del side
    outputs = {
        "vertex_heatmap": torch.full((1, 1, 96, 96), -8.0),
        "boundary_contact_heatmap": torch.full((1, 1, 96, 96), -8.0),
        "vertex_offset": torch.zeros((1, 2, 96, 96)),
        "vertex_kind": torch.zeros((1, 5, 96, 96)),
        "degree": torch.zeros((1, 9, 96, 96)),
        "incident_rays": torch.full((1, 36, 96, 96), -8.0),
        "boundary_side": torch.full((1, 4, 96, 96), -8.0),
    }
    outputs["boundary_contact_heatmap"][0, 0, peak_row, peak_col] = 8.0
    outputs["vertex_kind"][0, 2, peak_row, peak_col] = 5.0
    outputs["boundary_side"][0, side_id, peak_row, peak_col] = 8.0

    decoded = decode_vertex_refiner_outputs(
        outputs,
        square_frame=SquareFrame(20.0, 20.0, 76.0, 76.0),
        heatmap_threshold=0.25,
    )

    assert len(decoded) == 1
    assert decoded[0].x == expected_x
    assert decoded[0].y == expected_y


def test_vertex_refiner_loss_is_finite_and_backpropagates() -> None:
    model = VertexRefinerV1(base_channels=8)
    inputs = torch.randn(1, 8, 96, 96)
    targets = _tiny_targets()
    outputs = model(inputs)
    losses = VertexRefinerLoss()(outputs, targets)

    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grad_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0


def test_vertex_refiner_v2_loss_is_finite_and_backpropagates() -> None:
    model = VertexRefinerV2(base_channels=8)
    inputs = torch.randn(1, 12, 96, 96)
    targets = _tiny_targets()
    outputs = model(inputs)
    losses = VertexRefinerLoss()(outputs, targets)

    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["boundary_heatmap"])
    assert torch.isfinite(losses["boundary_side"])
    losses["total"].backward()
    grad_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0


def test_vertex_refiner_tiny_one_batch_overfit_reduces_loss() -> None:
    torch.manual_seed(11)
    model = VertexRefinerV1(base_channels=4)
    inputs = torch.randn(1, 8, 96, 96)
    targets = _tiny_targets()
    criterion = VertexRefinerLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    with torch.no_grad():
        initial = float(criterion(model(inputs), targets)["total"])
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)["total"]
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(criterion(model(inputs), targets)["total"])

    assert final < initial


def test_vertex_refiner_onnx_wrapper_exports_if_onnx_is_available() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    model = VertexRefinerV1(base_channels=4).eval()
    wrapper = VertexRefinerOnnxWrapper(model)
    buffer = BytesIO()

    torch.onnx.export(
        wrapper,
        torch.randn(1, 8, 96, 96),
        buffer,
        input_names=["refiner_input"],
        output_names=list(ONNX_OUTPUT_NAMES),
        opset_version=17,
    )

    assert buffer.getbuffer().nbytes > 0


def _tiny_targets() -> dict[str, torch.Tensor]:
    targets = {
        "vertex_heatmap": torch.zeros(1, 1, 96, 96),
        "boundary_contact_heatmap": torch.zeros(1, 1, 96, 96),
        "vertex_offset": torch.zeros(1, 2, 96, 96),
        "vertex_offset_mask": torch.zeros(1, 96, 96, dtype=torch.bool),
        "vertex_kind": torch.zeros(1, 96, 96, dtype=torch.long),
        "vertex_kind_mask": torch.zeros(1, 96, 96, dtype=torch.bool),
        "boundary_side": torch.zeros(1, 96, 96, dtype=torch.long),
        "boundary_side_mask": torch.zeros(1, 96, 96, dtype=torch.bool),
        "degree": torch.zeros(1, 96, 96, dtype=torch.long),
        "degree_mask": torch.zeros(1, 96, 96, dtype=torch.bool),
        "incident_rays": torch.zeros(1, 36, 96, 96),
        "incident_ray_mask": torch.zeros(1, 96, 96, dtype=torch.bool),
    }
    row = 48
    col = 48
    targets["vertex_heatmap"][0, 0, row, col] = 1.0
    targets["boundary_contact_heatmap"][0, 0, row, col] = 1.0
    targets["vertex_offset"][0, :, row, col] = torch.tensor([0.25, -0.25])
    targets["vertex_offset_mask"][0, row, col] = True
    targets["vertex_kind"][0, row, col] = 1
    targets["vertex_kind_mask"][0, row, col] = True
    targets["boundary_side"][0, row, col] = 0
    targets["boundary_side_mask"][0, row, col] = True
    targets["degree"][0, row, col] = 4
    targets["degree_mask"][0, row, col] = True
    targets["incident_rays"][0, [0, 9, 18, 27], row, col] = 1.0
    targets["incident_ray_mask"][0, row, col] = True
    return targets
