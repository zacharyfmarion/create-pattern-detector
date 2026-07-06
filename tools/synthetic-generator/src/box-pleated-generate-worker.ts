// Worker for the parallel Stage B generator. Each worker owns a stride of the
// stored seeds ((index) % workers === workerId), reads those packings, assigns
// M/V, and writes their FOLDs + metadata + a manifest shard. Assignment is
// CPU-bound and independent per packing; each worker writes its own files/shard so
// there's no cross-worker contention.
//
// Rescue-fallback: a "don't double" packing that comes back invalid at 1x is
// retried at 2x (doubling makes odd leftover gaps even -> fillable), recovering
// packings that would otherwise be rejected as incomplete.

import { join } from "node:path";
import { buildPackingCP } from "./box-pleated-cp.ts";
import { boxPleatedCpToFold, boxPleatedQuality } from "./box-pleated-fold.ts";
import { leafCountForSeed, readPacking, scalePacking, shouldDouble, splitForSeed, storedSeeds } from "./box-pleated-store.ts";
import type { EdgeAssignment } from "./types.ts";

declare const self: Worker;

interface WorkMsg {
  store: string;
  out: string;
  doubleFraction: number;
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
}

self.onmessage = async (event: MessageEvent<WorkMsg>): Promise<void> => {
  const { store, out, doubleFraction, workers, workerId, limit } = event.data;
  const seeds: number[] = [];
  for await (const s of storedSeeds(store)) seeds.push(s);
  seeds.sort((a, b) => a - b);
  const mine = seeds.filter((_, i) => i % workers === workerId);

  const rows: string[] = [];
  const st: Stats = { type: "progress", accepted: 0, doubled: 0, rescued: 0, clean: 0, incomplete: 0, offgrid: 0, thrown: 0, hist: {} };

  for (const seed of mine) {
    if (limit !== null && st.accepted >= limit) break;
    const raw = await readPacking(seed, store);
    if (!raw) continue;

    let scale = shouldDouble(seed, doubleFraction) ? 2 : 1;
    let cp;
    try {
      cp = buildPackingCP(scale === 2 ? scalePacking(raw, 2) : raw);
    } catch {
      st.thrown++;
      continue;
    }
    // Rescue-fallback: a 1x-invalid packing may become valid when doubled.
    if (!cp.valid && scale === 1) {
      try {
        const cp2 = buildPackingCP(scalePacking(raw, 2));
        if (cp2.valid) {
          cp = cp2;
          scale = 2;
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

    const id = `bp-${seed}${scale === 2 ? "-2x" : ""}`;
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
    if (scale === 2) st.doubled++;
    if (q.clean) st.clean++;
    st.hist[q.conflicts] = (st.hist[q.conflicts] ?? 0) + 1;
    if (st.accepted % 25 === 0) self.postMessage(st);
  }

  await Bun.write(join(out, `manifest.w${workerId}.jsonl`), rows.join("\n") + (rows.length ? "\n" : ""));
  self.postMessage({ ...st, type: "done" });
};
