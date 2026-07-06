// Worker for the parallel Stage A packer. Each worker owns a stride of the seed
// range ((seed - from) % workers === workerId) and generates + stores those
// packings independently - packing generation is CPU-bound and fully independent
// per seed, and the store's per-seed sharded/atomic writes make concurrent writes
// safe (no two workers touch the same file). Posts cumulative progress to the main
// thread and a final "done" message.

import { hasPacking, packForSeed, writePacking } from "./box-pleated-store.ts";

declare const self: Worker;

interface WorkMsg {
  from: number;
  to: number;
  workers: number;
  workerId: number;
}

self.onmessage = async (event: MessageEvent<WorkMsg>): Promise<void> => {
  const { from, to, workers, workerId } = event.data;
  let generated = 0;
  let skipped = 0;
  let errors = 0;
  let bytes = 0;

  for (let seed = from + workerId; seed < to; seed += workers) {
    if (await hasPacking(seed)) {
      skipped++;
      continue;
    }
    try {
      bytes += await writePacking(seed, await packForSeed(seed));
      generated++;
    } catch {
      errors++;
    }
    if (generated % 25 === 0) self.postMessage({ type: "progress", generated, skipped, errors, bytes });
  }

  self.postMessage({ type: "done", generated, skipped, errors, bytes });
};
