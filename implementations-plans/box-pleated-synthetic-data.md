# Box-Pleated Synthetic Data Plan

This plan covers synthetic uniaxial box-pleated data generation. The goal is to
move from a sampled tree to a BP Studio-valid layout packing, then later to
crease assignment and `.fold` export.

## Current Scope

The current implementation is a preview-only packing stage:

1. Sample a small uniaxial tree with terminal flaps.
2. Use the BP Studio optimizer as the placement oracle for flap AABBs.
3. Run BP Studio's TypeScript core headlessly on the optimized tree/flap layout.
4. Store and render the same layout primitives BP Studio uses:
   - flap rounded-rect/circle outlines;
   - ridge creases;
   - stretch-device axis-parallel creases.
5. Reject samples when BP Studio core throws, reports `patternNotFound`, or
   when optimizer tree-distance constraints are violated.
6. Optionally run a strict no-stretches preview mode that rejection-samples
   BP Studio layouts until the core output contains no stretch devices, no
   stretch-device axis-parallel lines, and only horizontal, vertical, or
   45-degree ridge creases.

This stage does not yet assign final mountain/valley labels beyond BP Studio's
layout layers, and it does not emit training `.fold` files.

### Tight mode (`--tight`)

The default and no-stretch paths trade paper efficiency for clean 45/90 grids.
Tight mode instead reproduces the app's "Optimize Layout": it fixes a single
sampled tree and restarts BP Studio's optimizer (basin-hopping on) from several
random initial vectors, keeping the tightest realizable layout. Pythagorean
stretch devices are permitted - their creases are well defined - so the packing
fills the sheet far more efficiently. For 8 leaves this lands sheets around
34-39 versus roughly 87 for the stretch-free banded layout (~5-6x less paper by
area). BP Studio's pairwise packing constraint is Euclidean for every grid type
(the C++ `circle`/`rounded` constraints; `diag` only changes the sheet
boundary), so tight Euclidean optima almost always contain off-grid contacts and
the stretch gussets that resolve them. Tight mode accepts that by design; the
no-stretch modes remain available when a clean grid is required.

## Geometry Completion

The next implemented preview layer is an unassigned crease scaffold. It derives
crease geometry from an accepted packing without assigning mountain/valley
labels:

- BP Studio ridge creases are preserved as `bp-ridge` lines.
- BP Studio 45/90 contour segments are imported as `bp-contour` hinge/contour
  candidates.
- Every 45-degree BP ridge contributes the perimeter of its axis-aligned
  bounding box as `computed-axial` candidates. This is a first-pass version of
  the axial contour geometry needed before MV assignment.
- Remaining unit grid cells outside flap exclusion regions receive a single
  45-degree `gap-ridge` candidate so the sheet can be inspected as a fully
  creased scaffold rather than a partial ridge layout.

The `gap-ridge` layer is deliberately marked as inferred geometry. It is useful
for finding and visualizing still-empty paper regions, but it is not yet a proof
of Lang-style molecule decomposition or flat-foldability. The next validation
step is to replace broad unit-cell gap fill with classified BP molecules.

## BP Studio Findings

BP Studio's layout pipeline is not a discrete tiler of colored square and river
cells. The relevant path is:

- `client/plugins/optimizer`: C++/WASM nonlinear optimizer places flap AABBs
  under pairwise tree-distance constraints. Rectangle flaps use rounded-rect
  constraints.
- `core/design/tasks/roughContour.ts`: leaf and river rough contours are built
  recursively.
- `core/design/tasks/junction.ts`: overlapping flap/ridge situations are
  classified.
- `core/design/tasks/stretch.ts` and `core/design/layout/*`: valid GOPS/stretch
  devices are searched and drawn.
- `client/project/components/layout/*` and `client/svg/index.ts`: the layout
  view renders blue hinge contours/flap circles, red ridges, green
  axis-parallel stretch lines, and flap anchor dots.

The generator now calls the optimizer and core path. The default packing preview
renders only actual flap radius outlines, ridge creases, stretch lines, and flap
anchor dots. BP Studio's hinge/river layout contours are preserved in JSON but
are hidden by default because they are internal fold-region contours, not flap
packing exclusion regions; pass `--show-contours` only for debugging BP core
output.

## Validity Contract

A packing preview is accepted only when:

- BP Studio's optimizer returns an integer layout.
- BP Studio's core pipeline completes without throwing.
- `patternNotFound` is false.
- Every terminal flap has BP Studio flap graphics.
- Optimized flap anchors lie on the returned sheet.
- Pairwise rounded-rectangle tree-distance constraints are satisfied.

BP Studio may draw flap radius outlines or AABB portions outside the sheet; the
layout renderer clips them to the sheet border, matching the app behavior.

For `noStretches` previews, all of the above still applies and the generator
also rejects any layout with:

- BP Studio stretch-device objects.
- Axis-parallel stretch lines emitted by those devices.
- Ridge creases whose slope is not horizontal, vertical, or exactly 45 degrees.

This is not implemented as a native BP Studio optimizer option. BP Studio's
optimizer permits Euclidean/Pythagorean separations, so the current approach is
to bias the initial vectors toward orthogonal edge/perimeter arrangements and
then post-filter the completed BP Studio core layout.

The initial-vector seeding is grid-native rather than rejection-driven. The
tree sampler emits leaves as sibling pairs on a shared hub, so every no-stretch
mode (`horizontal`, `vertical`, and `none`) seeds the canonical two-band layout:
each sibling pair straddles a central axis while hubs spread along it. That makes
every neighbour separation axial or 45-degree, which is exactly what BP Studio
can box-pleat without stretch devices. The `none` case picks the band axis once
per sample (not once per flap) and relies on the random per-leaf edge lengths to
break exact mirror symmetry. Earlier builds flipped each flap's axis
independently, scattering anchors off any common lattice; that dropped the
asymmetric 8-leaf no-stretch yield to ~12% and leaned on the 96-attempt rejection
cap. Per-sample banded seeding restores ~100% first-attempt yield across 4-12
leaves.

## Next Stages

1. Replace the temporary random sampler with a tree/symmetry sampler designed
   around target flap counts, branch-depth distributions, and rejection rates.
2. Decide whether to vendor a pinned BP Studio core snapshot, invoke a local
   clone as an external tool, or port the minimum required core modules.
3. Convert BP Studio layout graphics into crease-pattern data:
   - hinge contours as auxiliary or valley candidates;
   - ridge creases as mountain candidates;
   - stretch-device axis-parallel lines as valley candidates;
   - explicit provenance for every generated crease.
4. Add foldability QA and `.fold` export before adding the family to training
   mixes.

## Commands

Generate BP Studio-style preview packings from a local BP Studio clone:

```bash
BP_STUDIO_ROOT=/tmp/bp-studio-source \
bun run --cwd tools/synthetic-generator box-pleated-preview -- \
  --count 8 \
  --out /tmp/box-pleated-packings \
  --target-leaf-count 4
```

Generate pure 45/90 preview packings with no BP Studio stretch devices:

```bash
BP_STUDIO_ROOT=/tmp/bp-studio-source \
bun run --cwd tools/synthetic-generator box-pleated-preview -- \
  --count 8 \
  --out /tmp/box-pleated-no-stretch-packings \
  --target-leaf-count 6 \
  --symmetry horizontal \
  --no-stretches
```

Generate unassigned completed scaffold previews:

```bash
BP_STUDIO_ROOT=/tmp/bp-studio-source \
bun run --cwd tools/synthetic-generator box-pleated-preview -- \
  --count 8 \
  --out /tmp/box-pleated-scaffold-packings \
  --target-leaf-count 6 \
  --symmetry horizontal \
  --no-stretches \
  --scaffold
```

Show BP Studio's internal hinge/river contours in the preview:

```bash
BP_STUDIO_ROOT=/tmp/bp-studio-source \
bun run --cwd tools/synthetic-generator box-pleated-preview -- \
  --count 4 \
  --out /tmp/box-pleated-debug-contours \
  --target-leaf-count 4 \
  --show-contours
```

Run checks:

```bash
bun run --cwd tools/synthetic-generator typecheck
bun test --cwd tools/synthetic-generator
```
