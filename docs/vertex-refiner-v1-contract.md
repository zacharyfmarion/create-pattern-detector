# Vertex Refiner V1 Contract

This is the Phase 0 interface contract for the high-resolution vertex refiner.
Training, ONNX export, and `tree-maker-rust` integration should import or mirror
the constants in `src/models/vertex_refiner_contract.py`.

## Ownership

`create-pattern-detector` owns:

- crop proposal and crop tensor construction
- `VertexRefinerV1` training and local ML evaluation
- checkpoint manifests under `artifacts/checkpoints/`
- ONNX export into the downstream product model directory

`tree-maker-rust` owns:

- browser/runtime ONNX execution
- Rust/WASM output decoding
- product candidate graph integration
- clean-15, box-pleat native, candidate-coverage, and exact-solve reports

## Input Tensor

ONNX input name: `refiner_input`

Shape, without batch: `8 x 96 x 96`. Batch exports add a leading dynamic or
fixed batch dimension.

Channel order:

| Index | Name | Meaning |
| ---: | --- | --- |
| 0 | `image_gray` | Original grayscale crop, normalized to `0..1`. |
| 1 | `source_ink_probability` | Processed source-image line/ink probability, normalized to `0..1`. |
| 2 | `source_distance_to_ink` | Distance to source-image ink in crop-local pixels, normalized by crop size unless the manifest says otherwise. |
| 3 | `cpline_junction_probability` | CPLineNet junction heatmap crop. Auxiliary only. |
| 4 | `cpline_junction_offset_dx` | CPLineNet junction offset `dx`, in the existing radius-normalized decoder convention. |
| 5 | `cpline_junction_offset_dy` | CPLineNet junction offset `dy`, in the existing radius-normalized decoder convention. |
| 6 | `crop_x_normalized` | Crop-local x coordinate spanning `-1` at the left edge to `+1` at the right edge. |
| 7 | `crop_y_normalized` | Crop-local y coordinate spanning `-1` at the top edge to `+1` at the bottom edge. |

Phase 4 source-only training and the first product integration must set channels
`3..5` to zero. They remain in the tensor contract as reserved auxiliary
channels for a later controlled ablation, but they must not be required for a V1
checkpoint to run.

This keeps the production architecture parallelizable:

```text
source image -> source-image proposals -> VertexRefiner crops -> graph
```

instead of the waterfall:

```text
source image -> CPLineNet full-image pass -> VertexRefiner crops -> graph
```

Rendered GT junction labels are allowed only for unit tests and explicit
label-leakage diagnostics. Real CPLineNet dense caches should be used only for a
later source-plus-dense ablation after the source-only V1 result is measured.

## Coordinate Convention

All crop tensors use image coordinates:

- x increases to the right
- y increases downward
- array row is y
- array column is x

A crop origin is the full-image coordinate of local `(col=0, row=0)`.

For a heatmap peak at local `(col, row)` with predicted offset `(dx, dy)`, the
decoded full-image point is:

```text
x = crop_origin_x + col + dx
y = crop_origin_y + row + dy
```

Offsets are in crop pixels, not normalized units.

## Output Tensors

ONNX output names, in order:

1. `vertex_heatmap`: shape `1 x 96 x 96`
2. `vertex_offset`: shape `2 x 96 x 96`
3. `vertex_kind`: shape `5 x 96 x 96`
4. `degree`: shape `9 x 96 x 96`
5. `incident_rays`: shape `36 x 96 x 96`

`vertex_kind` class order:

```text
background
interior_junction
boundary_contact
corner
endpoint_or_dangling
```

`degree` class order:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8+
```

`incident_rays` are multi-label logits over 36 bins. Each bin is 10 degrees in
image-coordinate clockwise angles:

```text
0 degrees: east
90 degrees: south
180 degrees: west
270 degrees: north
```

The target bin is the nearest 10-degree bin. Opposite directions differ by 18
bins.

## Decode Defaults

The first V1 decoder should use:

```text
heatmap threshold: 0.25
NMS radius: 2 px
cross-crop duplicate merge radius: 1 px
```

Do not merge two refined vertices only because they are spatially close if their
high-confidence incident-ray signatures strongly disagree.

## ONNX And Manifests

Stable downstream directory:

```text
tree-maker-rust/apps/web/public/models/cp-vertex-refiner-v1/
```

Versioned downstream directory:

```text
tree-maker-rust/apps/web/public/models/cp-vertex-refiner-v1-<model-id>/
```

The ONNX file in either directory is named:

```text
model.onnx
```

Checkpoint manifests use schema:

```text
create-pattern-detector/vertex-refiner-checkpoint/v1
```

The future current-model pointer path is:

```text
artifacts/checkpoints/current-vertex-refiner-model.json
```

Pointer schema:

```text
create-pattern-detector/current-vertex-refiner-pointer/v1
```

Do not create or update the current pointer until a trained refiner is promoted.

## Phase 0 Baselines

The frozen comparison snapshot is:

```text
artifacts/evaluations/vertex-refiner-v1-phase0-baselines.json
```

The GT-junction ceiling mode lives in `tree-maker-rust`:

```text
compare_exact_solve_benchmark \
  --candidate-source junction-first-v1 \
  --line-evidence-source source-image \
  --junction-evidence-source ground-truth
```

Use that mode to separate refiner errors from downstream candidate graph and
solver errors.
