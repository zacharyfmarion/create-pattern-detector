// Worker for the parallel Stage B generator. Each worker owns a stride of the
// stored seeds ((index) % workers === workerId), reads those packings, assigns
// M/V, and writes their FOLDs + metadata + a manifest shard. Assignment is
// CPU-bound and independent per packing; each worker writes its own files/shard so
// there's no cross-worker contention.
//
// Rescue-fallback: a "don't double" packing that comes back invalid at 1x is
// retried at 2x (doubling makes odd leftover gaps even -> fillable), recovering
// packings that would otherwise be rejected as incomplete.

import { appendFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { buildPackingCP } from "./box-pleated-cp.ts";
import { boxPleatedCpToFold, boxPleatedQuality } from "./box-pleated-fold.ts";
import { leafCountForSeed, readPacking, scaleForSeed, scalePacking, shouldDouble, splitForSeed, storedSeeds, type ScaleMix } from "./box-pleated-store.ts";
import type { EdgeAssignment } from "./types.ts";

declare const self: Worker;

interface WorkMsg {
  store: string;
  out: string;
  doubleFraction: number;
  /** Optional scale mixture; when present it takes precedence over doubleFraction. */
  scaleMix: ScaleMix | null;
  workers: number;
  workerId: number;
  limit: number | null;
}

interface Stats {
  type: "progress" | "done";
  accepted: number;
  doubled: number;
  rescued: number;
  clean: number;
  incomplete: number;
  offgrid: number;
  thrown: number;
  hist: Record<number, number>;
  scaleCounts: Record<number, number>;
}

self.onmessage = async (event: MessageEvent<WorkMsg>): Promise<void> => {
  const { store, out, doubleFraction, scaleMix, workers, workerId, limit } = event.data;
  const seeds: number[] = [];
  for await (const s of storedSeeds(store)) seeds.push(s);
  seeds.sort((a, b) => a - b);
  const mine = seeds.filter((_, i) => i % workers === workerId);

  const rows: string[] = [];
  // Incremental shard flushing: a multi-hour Stage B must not keep manifest
  // rows only in memory (a restart would orphan every fold written so far).
  const shardPath = join(out, `manifest.w${workerId}.jsonl`);
  await writeFile(shardPath, "");
  const flush = async (): Promise<void> => {
    if (!rows.length) return;
    await appendFile(shardPath, rows.join("\n") + "\n");
    rows.length = 0;
  };
  const st: Stats = { type: "progress", accepted: 0, doubled: 0, rescued: 0, clean: 0, incomplete: 0, offgrid: 0, thrown: 0, hist: {}, scaleCounts: {} };

  for (const seed of mine) {
    if (limit !== null && st.accepted >= limit) break;
    const raw = await readPacking(seed, store);
    if (!raw) continue;

    let scale: number = scaleMix ? scaleForSeed(seed, scaleMix) : shouldDouble(seed, doubleFraction) ? 2 : 1;
    // Pitch guard: 4x on a very large grid drops the rendered pitch below the
    // resolution floor at 1024. Grid 36 * 4 = 144 -> ~6.7px, which matches the
    // native hard tail (per-CP p10 junction spacing reaches 7.5px), so grids
    // up to 36 are admitted; the guard now only protects hypothetical larger
    // grids. (Was >30 -> ~8px; that cut off the 3-4.5k-edge regime entirely.)
    if (scale === 4 && raw.sheet.width > 36) scale = 2;
    let cp;
    try {
      cp = buildPackingCP(scale === 1 ? raw : scalePacking(raw, scale));
    } catch {
      st.thrown++;
      continue;
    }
    // Rescue-fallback: doubling turns odd leftover gaps even, so an invalid
    // packing may become valid at the next even multiple (1x->2x, 2x->4x).
    // 3x is never used: it preserves gap parity (probed: invalid every time).
    if (!cp.valid && scale < 4) {
      const rescueScale = scale * 2;
      try {
        const cp2 = buildPackingCP(scalePacking(raw, rescueScale));
        if (cp2.valid) {
          cp = cp2;
          scale = rescueScale;
          st.rescued++;
        }
      } catch {
        /* fall through to reject */
      }
    }
    if (!cp.valid) {
      if (cp.complete) st.offgrid++;
      else st.incomplete++;
      continue;
    }

    const id = `bp-${seed}${scale > 1 ? `-${scale}x` : ""}`;
    const fold = boxPleatedCpToFold(cp, { id, seed, leafCount: leafCountForSeed(seed), scale });
    const q = boxPleatedQuality(cp);
    const assignments = fold.edges_assignment.reduce<Record<EdgeAssignment, number>>((acc, a) => {
      acc[a] = (acc[a] ?? 0) + 1;
      return acc;
    }, {} as Record<EdgeAssignment, number>);
    const foldPath = join("folds", `${id}.fold`);
    await Bun.write(join(out, foldPath), JSON.stringify(fold));
    await Bun.write(join(out, "metadata", `${id}.json`), JSON.stringify({ id, seed, scale, quality: q }));
    rows.push(JSON.stringify({
      id, seed, scale, leafCount: leafCountForSeed(seed), split: splitForSeed(seed), foldPath,
      grid: Math.round(cp.sheet.width), vertices: fold.vertices_coords.length, edges: fold.edges_vertices.length, assignments, quality: q,
    }));
    st.accepted++;
    if (scale > 1) st.doubled++;
    st.scaleCounts[scale] = (st.scaleCounts[scale] ?? 0) + 1;
    if (q.clean) st.clean++;
    st.hist[q.conflicts] = (st.hist[q.conflicts] ?? 0) + 1;
    if (st.accepted % 25 === 0) {
      await flush();
      self.postMessage(st);
    }
  }

  await flush();
  self.postMessage({ ...st, type: "done" });
};
