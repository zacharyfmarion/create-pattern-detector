# BP Studio Adapter

Headless Bun/TypeScript adapter for using the pinned BP Studio source tree at `third_party/bp-studio` to generate crease-pattern FOLD JSON from simple tree/layout specs.

The adapter follows BP Studio's own headless test flow:

1. Build a `Tree` from JSON tree edges and flap rectangles.
2. Store it in BP Studio `State`.
3. Run `heightTask`, which drives the downstream layout/pattern tasks.
4. Export CP lines with `LayoutController.getCP(sheetBorder, useAuxiliary)`.
5. Convert those lines to FOLD-style `vertices_coords`, `edges_vertices`, `edges_assignment`, and `edges_foldAngle`.

By default `useAuxiliary` is `false`, so BP Studio hinge contours are exported as valley folds instead of auxiliary folds.

## Usage

```bash
cd tools/bp-studio-adapter
bun run generate -- --spec fixtures/two-flap.json --out /tmp/bps.fold --metadata /tmp/bps.meta.json
```

## Tests

```bash
cd tools/bp-studio-adapter
bun test
```

## Spec Shape

```json
{
  "title": "Two-flap example",
  "sheet": { "width": 16, "height": 16 },
  "useAuxiliary": false,
  "tree": {
    "edges": [{ "n1": 0, "n2": 1, "length": 7 }],
    "flaps": [{ "id": 1, "x": 0, "y": 0, "width": 0, "height": 0 }]
  }
}
```

`tree.edges` are BP Studio tree edges. `tree.flaps` are placed leaf rectangles; point flaps use `width: 0` and `height: 0`, matching BP Studio's pattern tests.

For compatibility with future samplers, top-level `edges` and `flaps` are also accepted when `tree` is omitted.

## Metadata

When `--metadata` is provided, the adapter writes counts for assignments, vertices, edges, input tree/flap counts, and BP Studio stretch/repository summaries. `completeRepositories` defaults to `true`, which calls BP Studio repository completion so serialized repository data is available for smoke fixtures. Set `"completeRepositories": false` in large specs if exhaustive repository enumeration becomes too slow.

## Notes

- BP Studio is consumed as source via `tsconfig.json` path aliases; no files are copied from the submodule.
- This adapter is intentionally a geometry/export bridge. It does not perform Rabbit Ear strict validation, crossing splitting, or canonical graph cleanup.
- BP Studio CP export is a starting point for production data; downstream validation should still reject non-strict or unsplit outputs.
