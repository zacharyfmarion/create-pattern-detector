# Phase 3 V1 Status

Status: Phase 3 V1 is complete for the supported input envelope, and Phase 4 can
start.

## Decision

Treat the CPLineNet Phase 3 architecture as proven end-to-end for readable
1024px crease-pattern inputs. Do not spend another RunPod run chasing the dense
Rabbit Ear tail before starting Phase 4.

The important caveat is that V1 is not claiming robust recovery of extremely
dense or tiny iterative-fold geometry. Those cases should remain in eval as a
separate V2 regression track.

## Supported V1 Envelope

V1 is expected to work on crease patterns that are:

- Rectified or rendered into a square 1024px input.
- Visually readable, with crease spacing large enough to resolve at 1024px.
- Light or dark background, with varied crease colors and line styles.
- Free of dark guide-grid backgrounds and partial occlusions.

Real photos with large borders, skew, paper texture, and uneven lighting are
still part of the longer path, but the photo-like synthetic augmentations are in
place for the Phase 3 model path. Real-photo benchmark collection and
fine-tuning remain Phase 6 work.

## Evidence

The Phase 3 branch now includes:

- CPLineNet dense geometry and assignment heads.
- Raw `raw-manifest.jsonl` synthetic dataset loading via `foldPath`.
- Vector-first render augmentations including light, print, dark, photo-light,
  photo-dark, and exact square symmetries.
- Deprecated old pixel-head training path.
- Local MPS smoke training and 1024px shape/memory proof.
- RunPod 1024px `hrnet_w18` curriculum artifacts.
- Posthoc graph evaluation through `PlanarGraphBuilder`.
- Boundary metrics, per-family graph summaries, and clearer graph visualization.

The best Phase 3 V1 checkpoint is good on the readable clean slice:

| Slice | Samples | Edge recall | Edge precision | Border recall | Assignment accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| TreeMaker clean 1024 | 24 | 95.7% | 97.2% | 92.7% | 99.2% |
| Rabbit Ear clean 1024 | 24 | 86.1% | 89.8% | 79.7% | 97.8% |

Both stratified eval slices had 100% structurally valid predicted FOLD graphs.

The blessed checkpoint is registered in
`artifacts/checkpoints/phase3-v1-cpline.json`. The weight file itself remains an
ignored local artifact at
`checkpoints/runpod_phase3_curriculum/stage-balanced/latest.pt`; see
`docs/checkpoint-management.md` for the organization and replacement process.

## Known V2 Tail

Rabbit Ear is not uniformly bad. Easy Rabbit Ear examples are near-perfect, but
the generated family contains an iterative-fold tail with very short edges and
closely packed vertices. That tail is the main gap.

Local diagnostics on `max_edges <= 300` validation examples showed:

- Rabbit Ear p10 edge length is lower than TreeMaker.
- Rabbit Ear has more edges and vertices packed below 8-12px spacing.
- Edge recall falls as tiny-edge and close-vertex fractions rise.
- On Rabbit Ear, `tiny_edge_frac_lt8` correlated with edge recall at Spearman
  rho `-0.84`.
- On Rabbit Ear, `close_vertex_frac_lt8` correlated with edge recall at
  Spearman rho `-0.87`.

This does not imply the architecture is unusable. It means 1024px V1 should not
pretend to solve all dense tiny-fold cases.

## Phase 4 Handoff

Phase 4 should build on this checkpoint by adding assignment-focused evaluation,
constraint repair, and honest ambiguity/quality reporting. In particular, Phase
4 should expose warnings for inputs that appear outside the V1 envelope, such as
crowded predicted junctions, very short predicted edges, incomplete borders, or
unstable graph validity.

Dense Rabbit Ear and other tiny-geometry examples should stay available as a
separate regression suite, but they should not block Phase 4.

The implemented Stage 4 handoff, setup commands, eval commands, inspector
workflow, and current caveats are tracked in
`docs/phase-4-stage4-handoff.md`.
