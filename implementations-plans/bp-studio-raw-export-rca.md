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

Optimizer experiment:

- Added an adapter path that calls BP Studio's WASM optimizer bridge before CP export (`TreeController.getHierarchy(...)` + `Bridge.solve(...)`).
- A 5-attempt smoke still produced `0/1` accepted.
- Successful optimized exports still failed locally with very large counts (`~1150-1350` Kawasaki, `~1577-1848` Maekawa in representative attempts), and several sampled trees still crashed inside BP Studio layout/geometry code.
- Conclusion: optimizer-backed coordinates are necessary for production realism, but they do not turn BP Studio's exported design CP into a strict FOLD label. The hard missing layer remains CP completion/assignment repair.

Final-rendered/source-ancestry experiment:

- Added adapter source tagging for every exported edge: `edges_bpRole` plus `edges_bpStudioSource` with source kind, mandatory/optional status, owner/stretch/device IDs, and clipped segment index.
- Added `exportMode: "final"` using BP Studio final rendered node contours (`node.$graphics.$contours`) plus node ridges, device draw ridges, and device axis-parallel lines. This avoids the expanded-mode mistake of mixing rough/trace intermediate contours into the candidate graph.
- Two-flap final-rendered export matches the outer fixture scale (`24` raw vertices, `36` edges), but now records role counts `{border:6, ridge:15, hinge:10, axis:5}`.
- Diagnostic greedy local repair on the same two-flap fixture can make local checks pass only by unassigning all `30` folded non-border creases, leaving retention `0.0`. This proves that naive edge deletion/unassignment is not a viable production repair policy.
- A 3-attempt production smoke with final-rendered export still produced `0/1` accepted; representative failures are now BP Studio internal layout/export crashes (`undefined is not an object` inside BP Studio geometry) before strict validation. The generator now supports `CP_KEEP_BP_STUDIO_TMP=1` to preserve the exact adapter spec/output temp directory for crash RCA.

## Root Cause

BP Studio's `LayoutController.getCP(...)` export is a designer/display CP export, not a certified flat-foldable FOLD-label export. The adapter is mostly using BP Studio's own headless test path correctly:

- construct `Tree`;
- install it in `State`;
- run `Processor.$run(heightTask)`;
- complete stretch repositories;
- call `LayoutController.getCP(border, useAuxiliary)`.

The problem is the export contract. BP Studio emits visible/design lines: sheet border, outer/final node contours, ridges, and selected device lines. These lines have meaningful BP roles, but they are not a solved strict M/V assignment. After our arrangement and assignment normalization, many vertices have impossible local patterns such as degree-3 all-mountain or degree-4 all-mountain/all-valley junctions.

This is not only caused by the random sampler. BP Studio's own tiny two-flap fixture also fails local Rabbit Ear checks after export/normalization, which means raw BP Studio CP export cannot be treated as a production training label by itself.

## Evidence

Representative probe results from subagent RCA:

- Two-flap BP Studio fixture with `useAuxiliary=false`: raw `24` vertices, `36` edges, assignments `{B:6, M:15, V:15}`; local failures: Kawasaki bad `8`, Maekawa bad `12`.
- Same fixture with `useAuxiliary=true`: assignments `{B:6, M:15, F:10, V:5}`; still fails: Kawasaki bad `3`, Maekawa bad `10`.
- Normalized fixture with auxiliary forced to valley: `22` vertices, `36` edges; Kawasaki bad `8`, Maekawa bad `13`.
- A sampler-generated small insect sample: `775` exported lines, stretch repo valid/complete with selected pattern/device/gadget, but raw local check found Kawasaki bad `96`, Maekawa bad `151`.
- Tagged two-flap fixture with source roles: normalized role counts `{border:6, hinge:10, ridge:15, axis:5}`; local failures remain Kawasaki bad `8`, Maekawa bad `13`.
- Greedy unassignment repair for the tagged two-flap fixture: local failures drop to zero only after all `30` folded M/V creases are relabeled `U`, so the output is locally valid but useless as strict CP supervision.

Likely cause ranking:

1. BP Studio export contract mismatch: high.
2. Missing auxiliary/inner contour/completion geometry: high.
3. Assignment mapping bug: low, because the adapter matches BP Studio's own FOLD exporter mapping.
4. Sampler/layout bug: medium-low as a primary cause; the sampler can make things worse, but fixtures fail too.
5. Adapter API misuse: low for export, higher for production-quality layout because we are not yet using BP Studio's optimizer path.

## Required Next Work

Production data requires a true BP Studio completion/repair layer, not a fallback generator.

Ordered next patches:

1. Use source-tagged BP Studio exports as the candidate geometry source. Keep `F`/auxiliary lines as candidate geometry, not forced strict valleys.
2. Stabilize adapter/spec generation by collecting crash fixtures with `CP_KEEP_BP_STUDIO_TMP=1`, reducing the sampler's invalid layout rate, and adding fixture regressions for each BP Studio crash class.
3. Compare local failure counts for `outer`, `final`, and carefully selected optional geometry on BP Studio fixtures. Do not use rough/trace contours as production labels unless a source-specific test proves they help.
4. Implement a local completion layer guided by BP Studio geometry and source ancestry:
   - arrange/split all lines;
   - keep mandatory role lines unless a source-specific proof says they are optional;
   - solve/search active candidate geometry and M/V assignments under Kawasaki/Maekawa;
   - then run Rabbit Ear global validation.
5. Promote only source-ancestry-preserving, strict-global-passing repairs into the production generator. Greedy unassignment/deletion remains diagnostic-only.

## Non-Negotiable Gate

No sample enters production training unless the raw BP Studio-derived graph or a BP Studio-geometry-based repaired graph passes:

- complete border;
- no duplicate/degenerate/unsplit crossing geometry;
- local Kawasaki/Maekawa;
- Rabbit Ear global layer solver;
- finite folded coordinates;
- visual QA against BP-like distribution metrics.
