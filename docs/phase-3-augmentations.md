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
- `dark-mode`: dark canvas, optional faint grid, bright M/V colors, and
  gray/white border or unassigned lines.
- `print-light`: line-style plus paper tone, mild blur/noise, and JPEG
  compression.
- `print-medium`: stronger line/color/background diversity, uneven lighting,
  blur, noise, and compression.
- `photo-light`: mild residual affine/perspective perturbation, lens/defocus
  blur, lighting gradients, and photo-like compression.
- `mixed`: profile sampler for later staged training. It should only be used
  after the individual profiles pass visual QA and local graph-eval gates.

`--render-noise` is deprecated for CPLineNet. Use `--augment-profile`; the old
flag remains only as a compatibility alias.

## Target Rules

- Square symmetries and photo geometry transform vertices before rendering
  `line_prob`, `angle`, `junction_heatmap`, `junction_offset`, and assignment
  labels.
- Assignment labels move with the graph. M/V are preserved under rotations and
  flips because the diagram colors are also rendered after the same transform.
- Dark-mode grid pixels are background noise. They must never appear in line or
  junction targets.
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
.venv/bin/python scripts/visualize/cpline_augmentations.py \
  --profiles all \
  --num-samples 2 \
  --image-size 384 \
  --output-dir visualizations/phase3_augmentations
```

For the square rotation/flip sheet:

```bash
.venv/bin/python scripts/visualize/cpline_augmentations.py \
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
- Dark-mode grid lines do not appear in line/junction targets.
- M/V colors remain aligned with assignment overlays.
- Render timing remains acceptable at 256 and 384 before testing 1024.

## Training Sequence

Recommended local-first curriculum:

1. Visual QA for `square-symmetry`, `line-style`, `dark-mode`, `print-light`,
   `print-medium`, and `photo-light`.
2. Tiny local smoke at 384px with `print-light`.
3. Tiny local smoke at 384px with `dark-mode`.
4. Robustness gate with staged profile mixes, starting with
   `clean + square-symmetry + line-style + print-light`.
5. Add `print-medium` and `photo-light`.
6. Add `dark-mode` without grid.
7. Add `dark-mode` with grid at low probability.
8. Use full `mixed` only after the staged gates are stable.
9. Run a short 1024px local feasibility pass before moving to RunPod.

When synthetic M/V-rich examples land, run them through the same profiles and
evaluate assignment accuracy separately from geometry.
