# Phase 3 CPLineNet Augmentations

Phase 3 augmentations are CPLineNet-specific and vector-first. When geometry
changes, the FOLD vertices are transformed first, then the input image and all
dense targets are rendered from the transformed graph. Photometric effects are
applied only to the input image.

## Profiles

- `clean`: deterministic white-background render. This is the validation
  baseline and compatibility target.
- `square-symmetry`: exact non-identity square-domain rotations and flips:
  90/180/270-degree rotations, horizontal/vertical flips, transpose, and
  anti-transpose. Identity remains available only as a pinned debug/test value.
- `line-style`: line width, opacity, antialiasing, assignment color jitter, and
  grayscale/monochrome variants.
- `dark-mode`: dark canvas, varied bright/muted M/V colors, and gray/white
  border or unassigned lines. Background grids are out of scope for V1.
- `print-light`: line-style plus paper tone, mild blur/noise, and JPEG
  compression.
- `print-medium`: stronger line/color/background diversity, uneven lighting,
  blur, noise, and compression.
- `photo-light`: mild residual affine/perspective perturbation, lens/defocus
  blur, lighting gradients, and photo-like compression.
- `photo-dark`: dark-mode canvas and crease palette plus mild perspective,
  blur, lighting gradients, noise, and photo-like compression.
- `stage-base`: short geometry/label warmup with only `clean`,
  `square-symmetry`, and `line-style`.
- `stage-balanced`: main training mix with light, print/photo-light, dark, and
  photo-dark samples present together from early in training.
- `mixed`: compatibility alias for the current `stage-balanced` sampler.

`--render-noise` is deprecated for CPLineNet. Use `--augment-profile`; the old
flag remains only as a compatibility alias.

## Target Rules

- Square symmetries and photo geometry transform vertices before rendering
  `line_prob`, `angle`, `junction_heatmap`, `junction_offset`, and assignment
  labels.
- Style-only profiles such as `line-style`, `print-light`, `print-medium`, and
  `dark-mode` do not rotate or flip geometry by default. Curriculum mixes add
  orientation coverage by sampling `square-symmetry` as a separate profile.
- `photo-light` and `photo-dark` apply mild perspective to graph vertices first,
  then render input images and targets from the transformed graph.
- Assignment labels move with the graph. M/V are preserved under rotations and
  flips because the diagram colors are also rendered after the same transform.
- Background grids are intentionally out of scope for V1. Dark-mode augmentation
  varies background darkness and crease-line palettes, but does not render grid
  lines.
- Monochrome line-style renders collapse M/V supervision to `U` for that sample
  so the model does not learn to invent assignments when visual color evidence
  is absent.
- Parsed FOLD records and base pixel geometry may be cached. Random augmented
  tensors should not be cached by dataset index.
- Occlusions are intentionally out of scope for V1. They need a separate
  completion/confidence contract before being used for training.

## Visual QA

Generate contact sheets before increasing image size or training duration:

```bash
scripts/data/link_shared_synthetic_data.sh cp_training_mix_v1

.venv/bin/python scripts/visualize/cpline_augmentations.py \
  --manifest data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
  --profiles all \
  --num-samples 2 \
  --image-size 384 \
  --output-dir visualizations/phase3_augmentations
```

For the square rotation/flip sheet:

```bash
.venv/bin/python scripts/visualize/cpline_augmentations.py \
  --manifest data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl \
  --profiles square-symmetry \
  --num-samples 1 \
  --image-size 384 \
  --output-dir visualizations/phase3_square_symmetry
```

Each profile directory writes `contact_sheet_<size>.png` and
`contact_sheet_<size>.json`. The sidecar records sampled params, render timing,
line/junction pixel counts, clipped vertex count, edge count, profile, selected
profile, and square symmetry.

Visual checks before larger sizes:

- Overlays align with the augmented image.
- Square-symmetry rows show the seven non-identity orientations.
- Dark-mode backgrounds stay free of grid-like guide lines.
- M/V colors remain aligned with assignment overlays.
- Render timing remains acceptable at 256 and 384 before testing 1024.

## Training Sequence

Recommended local-first curriculum:

1. Visual QA for `square-symmetry`, `line-style`, `dark-mode`, `print-light`,
   `print-medium`, `photo-light`, and `photo-dark`.
2. Tiny local smoke at 384px with `print-light`.
3. Tiny local smoke at 384px with `dark-mode`.
4. Short `stage-base` warmup to establish geometry and assignment labels.
5. Main `stage-balanced` training so light, print/photo-light, dark, and
   photo-dark all remain in distribution together.
6. Use targeted continuation only after deterministic eval identifies a
   specific weakness, and keep the other modes in the mix to avoid forgetting.
7. Run a short 1024px local feasibility pass before moving to RunPod.

`stage-balanced` should initialize from the `stage-base` checkpoint. Restarting
each stage from random weights is a stress test, not the intended curriculum.

For local architecture gates on dense mixed data, set `--graph-eval-count` to a
small number such as `4` or `8`. Training and pixel validation still use the
requested sample counts, while PlanarGraphBuilder only vectorizes that many
examples per split. Leave it unset for serious RunPod evaluation.

The RunPod handoff and staged GPU script are documented in
`docs/runpod-phase-3.md`.

CPLine training reads fold-only `raw-manifest.jsonl` datasets directly. The
current default is the linked mixed release at
`data/generated/synthetic/cp_training_mix_v1/raw-manifest.jsonl`. When
`--train-count` or `--val-count` limits a split, CPLine samples a seeded subset
across the whole split so small local runs still include both TreeMaker and
Rabbit Ear rows from the mixed manifest.

Run synthetic M/V-rich examples through the same profiles and evaluate
assignment accuracy separately from geometry.
