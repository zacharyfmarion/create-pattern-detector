# Phase 5 Inference CLI

Status: implemented. Phase 5 provides the production inference shell for readable
crease-pattern inputs and page/screenshot images with a visible CP panel border.
It reuses the blessed Phase 3 V1 CPLineNet checkpoint and the Stage 4 graph
assignment, repair, quality-report, and FOLD export stack.

## Setup

Use the shared Python environment:

```bash
scripts/setup_python_env.sh
```

Recover the blessed checkpoint into this worktree if it is missing:

```bash
mkdir -p checkpoints/runpod_phase3_curriculum/stage-balanced
cp /Users/zacharymarion/Documents/code/create-pattern-detector/checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt \
  checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt
shasum -a 256 checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt
stat -f '%z bytes' checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt
```

Expected values from `artifacts/checkpoints/phase3-v1-cpline.json`:

- SHA-256:
  `a2a3d31d2ff80d3cf76952e463d965d03e6c46358a9802c2640a39b57d7732d8`
- Size: `138787408` bytes

Link the synthetic mix only when running checkpoint-backed regression or eval
commands:

```bash
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1
```

## Usage

Phase 5 supports readable CP images via `--rectified`:

```bash
cp-detect --rectified input.png \
  --output output.fold \
  --report output.report.json \
  --debug-dir debug/
```

Batch mode accepts multiple shell-expanded image paths:

```bash
cp-detect --rectified examples/*.png --output-dir outputs/cp-detect
```

For multiple inputs, the command writes one `.fold`, one `.report.json`, one
debug directory per image, and `batch.summary.json`.

Running without `--rectified` exits with a Phase 6 message. In Phase 5, the flag
means the input is a readable CP image or a page/screenshot containing a visible
CP panel; arbitrary real-photo discovery and benchmarked document rectification
remain Phase 6 work.

## CP Panel Isolation

`SquareRectifier` now attempts CP-panel isolation before falling back to the
older resize/pad path:

1. Load with PIL, apply EXIF orientation, normalize RGB/grayscale/RGBA, and
   flatten transparency with the configured matte policy.
2. Detect visible square/quadrilateral CP-panel borders using contour evidence,
   line evidence, square/aspect scoring, interior crease density, and line
   coverage.
3. Perspective-warp the detected panel to the canonical model square, mapping
   the detected CP border to a small inset ring instead of the exact image edge
   so 1px borders survive interpolation. The default inset is 32 px at 1024,
   matching the Phase 3 CPLineNet training renderer's canonical padding.
4. Fill the outside ring with the inferred padding/matte color, keeping nearby
   titles or folded-model art out of the model input while preserving the CP
   border pixels.
5. Record `homography_image_to_square`, source quad, target quad, border margin,
   confidence, and metrics.
6. If no panel is detected confidently, resize/pad the whole input and record the
   lower-confidence fallback transform.

This crop step removes titles, folded-model drawings, photos, and surrounding
page content when the CP has a detectable border. Most clean screenshots,
scanned pages, and mild perspective skews are in scope. Very faint borders,
heavily occluded panels, curved photos, multiple competing CP panels, and CPs
without enough long-line evidence are still Phase 6/benchmark cases.

## Alpha And Dark-Mode Inputs

Opaque RGB or grayscale inputs preserve their pixels. Dark-mode CPs remain dark.

Transparent inputs must be flattened to RGB before CPLineNet inference. The
default policy is:

```bash
--alpha-matte auto
```

`auto` first tries to infer the intended matte from stored transparent pixels,
then from opaque border pixels. If it cannot infer a reliable matte, it falls
back to white and records a report warning. Explicit alternatives are:

```bash
--alpha-matte white
--alpha-matte black
```

Non-square `--rectified` inputs still get a rectification warning. If a CP panel
is detected, that panel is cropped/warped; otherwise the whole image is
letterboxed. Padding color is inferred from the image/matte rather than assumed
white, so dark-mode inputs are not washed out.

## Outputs

Successful, repaired, ambiguous, and out-of-envelope results write a `.fold`
file with `cp_detector` metadata using schema `cp-detector/cp-detect/v1`.

Failed results write the report and debug artifacts but do not write a `.fold`
file. This prevents silent invalid FOLD output.

Debug artifacts are deterministic and lightweight:

```text
input.png
rectified.png
line_prob.png
junction_heatmap.png
graph_overlay.png
assignment_overlay.png
graph.json
report.json
```

## Tests

Fast Phase 5 and Stage 4 tests:

```bash
.venv/bin/python -m pytest \
  tests/test_phase5_inference.py \
  tests/test_stage4_vectorization.py \
  tests/test_stage4_diagnostics.py \
  tests/test_planar_graph_builder.py
```

Checkpoint-backed smoke tests should stay opt-in and use the verified local
checkpoint. Do not retrain or replace the blessed checkpoint as part of Phase 5.
