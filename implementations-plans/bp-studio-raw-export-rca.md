# BP Studio Raw Export Strictness RCA

## Status

The old synthetic generator families and strict-completion fallback have been removed from the production synthetic path. `bp-studio-realistic` is now BP-Studio-or-fail: it samples a BP Studio-style tree/layout, runs the headless BP Studio adapter, normalizes the raw export, and only returns raw exports that pass local Kawasaki/Maekawa precheck before the normal strict validator runs.

Current raw-only smoke:

```bash
bun run generate -- --recipe recipes/synthetic/bp_studio_realistic_v1.yaml --count 2 --out /tmp/bp_studio_raw_only --max-attempts 40
```

Result:

- `0/2` accepted after 40 attempts.
- Rejection class: raw BP Studio export failed local Kawasaki/Maekawa.

Expanded export experiment:

- Added an adapter `exportMode: "expanded"` probe that includes inner contours, pattern/trace/rough contours, device contours, and trace ridges in addition to BP Studio's default export surface.
- A 10-attempt smoke still produced `0/1` accepted.
- Average local failures stayed very high (`~514` Kawasaki, `~682` Maekawa on local-failure attempts), and adapter failures increased for hard sampled layouts.
- Conclusion: simply exporting more visible/internal contours is not enough. The missing piece is a real assignment/geometry completion step, not just more lines.

## Root Cause

BP Studio's `LayoutController.getCP(...)` export is a designer/display CP export, not a certified flat-foldable FOLD-label export. The adapter is mostly using BP Studio's own headless test path correctly:

- construct `Tree`;
- install it in `State`;
- run `Processor.$run(heightTask)`;
- complete stretch repositories;
- call `LayoutController.getCP(border, useAuxiliary)`.

The problem is the export contract. BP Studio emits visible/design lines: sheet border, outer node contours, ridges, and selected device lines. It intentionally omits some internal contour/completion geometry. After our arrangement and assignment normalization, many vertices have impossible local patterns such as degree-3 all-mountain or degree-4 all-mountain/all-valley junctions.

This is not only caused by the random sampler. BP Studio's own tiny two-flap fixture also fails local Rabbit Ear checks after export/normalization, which means raw BP Studio CP export cannot be treated as a production training label by itself.

## Evidence

Representative probe results from subagent RCA:

- Two-flap BP Studio fixture with `useAuxiliary=false`: raw `24` vertices, `36` edges, assignments `{B:6, M:15, V:15}`; local failures: Kawasaki bad `8`, Maekawa bad `12`.
- Same fixture with `useAuxiliary=true`: assignments `{B:6, M:15, F:10, V:5}`; still fails: Kawasaki bad `3`, Maekawa bad `10`.
- Normalized fixture with auxiliary forced to valley: `22` vertices, `36` edges; Kawasaki bad `8`, Maekawa bad `13`.
- A sampler-generated small insect sample: `775` exported lines, stretch repo valid/complete with selected pattern/device/gadget, but raw local check found Kawasaki bad `96`, Maekawa bad `151`.

Likely cause ranking:

1. BP Studio export contract mismatch: high.
2. Missing auxiliary/inner contour/completion geometry: high.
3. Assignment mapping bug: low, because the adapter matches BP Studio's own FOLD exporter mapping.
4. Sampler/layout bug: medium-low as a primary cause; the sampler can make things worse, but fixtures fail too.
5. Adapter API misuse: low for export, higher for production-quality layout because we are not yet using BP Studio's optimizer path.

## Required Next Work

Production data requires a true BP Studio completion/repair layer, not a fallback generator.

Ordered next patches:

1. Preserve `F` auxiliary semantics through diagnostics and stop coercing auxiliary lines into valleys for strict acceptance experiments.
2. Add an adapter probe/export mode that includes more BP Studio internal geometry: inner contours, device contours, trace/rough contours, raw ridges, add-on contours, and source tags.
3. Compare local failure counts for outer-only export vs expanded export on BP Studio fixtures.
4. Implement a local completion layer guided by BP Studio geometry:
   - arrange/split all lines;
   - add missing contour/completion lines;
   - solve/search M/V assignments under Kawasaki/Maekawa;
   - then run Rabbit Ear global validation.
5. Integrate BP Studio optimizer-backed layout generation. Our current sampler passes terminal coordinates directly into `new Tree(...)`; it does not use BP Studio's optimizer bridge.

## Non-Negotiable Gate

No sample enters production training unless the raw BP Studio-derived graph or a BP Studio-geometry-based repaired graph passes:

- complete border;
- no duplicate/degenerate/unsplit crossing geometry;
- local Kawasaki/Maekawa;
- Rabbit Ear global layer solver;
- finite folded coordinates;
- visual QA against BP-like distribution metrics.
