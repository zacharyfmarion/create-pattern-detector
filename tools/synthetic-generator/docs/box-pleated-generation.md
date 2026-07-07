# Box-pleat crease-pattern generation

Generates valid, flat-foldable **box-pleated** origami crease patterns (with
Pythagorean stretches) as FOLD training data. The pipeline is split into two
decoupled stages so the expensive, stable **packings** are generated once and the
cheap, still-evolving **assignment** can be re-run freely.

```
seed ──► [Stage A: pack] ──► packing store ──► [Stage B: assign] ──► FOLD dataset
         generateBoxPleatedPacking            buildPackingCP → FOLD + manifest
         (~700ms/seed, STABLE)                (routeHinges, ITERATING)
```

Why two stages: a packing is the tight-pack layout of a random tree — expensive and
unaffected by assignment work. The M/V assignment (hinge routing, coloring) is what
we keep improving. Storing packings means improving assignment never regenerates
them.

## Quickstart

```bash
cd tools/synthetic-generator

# Stage A — build the packing corpus once (parallel, resumable). Writes to $BP_PACKING_STORE.
bun run box-pleated-pack --from 0 --to 100000

# Stage B — assign + serialize to FOLD (parallel, re-runnable as assignment improves).
bun run box-pleated-generate --out data/output/box-pleated --double-fraction 0.5
```

## Stage A — packings (`box-pleated-pack`)

Deterministically generates a packing per seed and stores it. Idempotent per seed.

```bash
bun run box-pleated-pack --from 0 --to 100000 [--workers N]   # range, parallel worker pool (default = CPU count)
bun run box-pleated-pack --from 0 --count 10000               # incremental "N new", single-threaded
```

- **Resumable** — skips packings already in the store; safe to kill and re-run.
- **Atomic** gzipped writes; **crash-safe** per seed.
- A packing is fully determined by its seed (params are fixed via `configForSeed`):
  `numCreases 300`, `tight`, `tightRestarts 14`, `leafCount = [4,5,6][seed%3]`.

## Stage B — crease patterns (`box-pleated-generate`)

Reads packings from the store, assigns M/V, and writes each **valid** CP as a FOLD.
Runs a Bun worker pool over the stored seeds (each worker owns a stride, writes its
own FOLDs + a manifest shard; the driver merges shards).

```bash
bun run box-pleated-generate --out DIR [--double-fraction 0.5] [--workers N] [--limit K]
```

Two levers live here:

- **Scaling** — either the legacy `--double-fraction` (default 0.5: 2× a random
  fraction of seeds) or `--scale-mix 1:0.25,2:0.35,4:0.4` (deterministic by seed
  hash; takes precedence). Scaling all geometry by k keeps a packing valid
  (integers stay integers) but fills the finer grid with more pleats. **Even
  multipliers only** — 3× preserves gap parity (odd gaps stay odd) and probed
  invalid on every packing. Measured medians at grid ~25: 1× ≈ 290 edges/41px
  pitch, 2× ≈ 705/19px, 4× ≈ 1,400–3,600/10px — 4× lands on the native hard-BP
  band (edges 729/1,376/3,478 p10/50/90). Scaled CPs get `-2x`/`-4x` ids.
  **Pitch guard:** 4× on grids >30 would drop rendered pitch below the ~8px
  resolution floor at 1024, so those fall back to 2× automatically.
- **Rescue-fallback**: a packing **invalid at its chosen scale** is retried at the
  next even multiple (1×→2×, 2×→4×). Doubling turns odd leftover gaps even
  (fillable), so this recovers packings otherwise dropped as *incomplete* —
  strictly higher yield. (Because of this, the emitted scale distribution skews
  denser than the *decision* distribution.)

## The packing store

- **Location:** `$BP_PACKING_STORE`, default
  `/Users/zacharymarion/Documents/datasets/create-pattern-detector/bp-packings`
  (outside the work tree). Both stages and the debug tools read it.
- **Format:** one gzipped JSON per packing (~1.8KB), at
  `<store>/<seed/1000>/bp-<seed>.json.gz`, sharded so no directory is huge.
- **Provenance:** `PACKER_VERSION` fingerprints the packing generator; bump it only if
  the tight-pack search / packing params change (assignment changes never touch it).

## Output

- `folds/<id>.fold` — FOLD crease pattern, coords normalized to the `[0,1]²` square
  domain, `edges_assignment` in `M`/`V`/`B`/`U`. Box-pleat provenance + per-CP quality
  ride under `box_pleated_metadata` (`seed`, `scale`, `quality`).
- `metadata/<id>.json` — per-CP config + quality.
- `manifest.jsonl` — one row per CP: `id, seed, scale, leafCount, split, foldPath,
  grid, vertices, edges, assignments, quality`.
- `summary.json` — run totals: `accepted, doubled, rescued, clean`, rejection
  taxonomy (`incomplete` / `off-grid` / `throw`), conflict histogram, throughput.

`quality = { conflicts, kawasakiFailing, unassigned, clean }` — Maekawa conflicts,
Kawasaki/even-degree failures, non-border creases without an M/V label. `clean` means
all three are zero. **Nothing is silently accepted:** every CP records its quality so
downstream can tier/filter (e.g. keep `clean`, or `conflicts <= k`).

`split` (train/val/test, 85/10/5) is deterministic by seed hash so parallel workers
assign it independently.

## Levers & knobs

| Lever | Where | Effect |
|---|---|---|
| `--double-fraction` | Stage B | fraction of packings emitted doubled (2×; legacy) |
| `--scale-mix` | Stage B | deterministic scale mixture, e.g. `1:0.25,2:0.35,4:0.4` (even scales only; wins over `--double-fraction`) |
| `HINGE_BUDGET` (env) | routeHinges | hinge-search node budget (default 300). Lower = faster, slightly worse assignment |
| `--workers` | both stages | worker-pool size (default = CPU count) |
| `leafCountForSeed` | `box-pleated-store.ts` | tree complexity — currently `[4,5,6][seed%3]`; widen to widen the grid/complexity distribution |
| `BP_PACKING_STORE` (env) | store | packing corpus location |

Grid-size distribution (as of this checkpoint): square, ~symmetric, mean ~25, range
14–36, a mixture of three leaf-count sub-populations (4→~19, 5→~25, 6→~29). Complexity
is **capped by the fixed leaf range**; doubling and/or a wider `leafCountForSeed`
extend it.

## Known limitations (current quality)

- **Residual assignment conflicts.** The hinge router does not fully resolve every
  degree-8 flap center / coupled ridge crossing yet, so many CPs carry a handful of
  Maekawa conflicts (recorded in `quality.conflicts`). Global flat-foldability needs a
  layer-aware assignment step.
- **Yield ~24–30%** at 1× (mostly incomplete or off-grid packings); the rescue-fallback
  raises it. Off-grid rejections are not recoverable by doubling.
- **Not yet a `GeneratorFamily`.** This lives beside `index.ts`'s bulk pipeline rather
  than inside it; folding it in (to inherit shared manifest/splits/validation) is a
  future step.

## Debug & inspection tools

| Command | Purpose |
|---|---|
| `box-pleated-hinge-trace <seed>` | render the hinge router's backtracking search, frame per step |
| `box-pleated-hinge-why <seed>` | text: why the hinge search stalls (per stuck vertex, candidate reasons) |
| `box-pleated-gen-trace <seed>` | stage-by-stage generation trace |
| `box-pleated-packing-compare <seed>` | packing vs. gap-filled-with-creases, two panels |
| `box-pleated-preview` / `box-pleated-debug` | assorted CP / packing renders |

## Module map

- `box-pleated-packing.ts` — tight-pack search → packing (Stage A primitive).
- `box-pleated-store.ts` — packing store IO + `scalePacking` / `shouldDouble` / split.
- `box-pleated-cp.ts` — `buildPackingCP`: packing → geometry → assigned CP.
- `box-pleated-molecule.ts` — ridges, axials, pleats, hinges, planarization.
- `box-pleated-assignment.ts` — M/V: ridge coloring, hinge routing (`routeHinges`).
- `box-pleated-axial-coloring.ts` — axial chain coloring.
- `box-pleated-fold.ts` — CP → FOLD + quality.
- `box-pleated-pack.ts` / `-generate.ts` (+ `-worker.ts`) — Stage A / Stage B drivers.
