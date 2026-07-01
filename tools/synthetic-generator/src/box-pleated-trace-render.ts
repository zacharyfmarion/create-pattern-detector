// CLI: render the whole generation of a box-pleat CP as a PNG frame sequence.
//   bun run src/box-pleated-trace-render.ts <seed> [pixels]
// Frames land in /tmp/gen-trace-<seed>/step_NNN.png, one per pipeline stage
// (paper boundary -> ridges by source -> axials -> pleats by level -> hinges ->
// M/V assignment). Reuses traceGeneration + renderGenerationFrames, so it always
// shows exactly what the build pipeline produces.

import { generateBoxPleatedPacking } from "./box-pleated-packing.ts";
import { traceGeneration } from "./box-pleated-cp.ts";
import { renderGenerationFrames } from "./box-pleated-generation-trace.ts";

const seed = Number(Bun.argv[2] ?? 60045);
const pixels = Number(Bun.argv[3] ?? 1100);
const leafCount = [4, 5, 6][seed % 3];

const packing = await generateBoxPleatedPacking({
  id: `s${seed}`,
  seed,
  numCreases: 300,
  bucket: "s",
  symmetry: "none",
  targetLeafCount: leafCount,
  tight: true,
  tightRestarts: 14,
});

const trace = traceGeneration(packing);
const frames = renderGenerationFrames(trace, packing, { pixels });

const dir = `/tmp/gen-trace-${seed}`;
await Bun.$`rm -rf ${dir}`.quiet().catch(() => {});
await Bun.$`mkdir -p ${dir}`.quiet();
for (let i = 0; i < frames.length; i++) {
  const n = String(i).padStart(3, "0");
  await Bun.write(`${dir}/step_${n}.svg`, frames[i].svg);
  await Bun.$`qlmanage -t -s ${pixels} -o ${dir} ${dir}/step_${n}.svg`.quiet().catch(() => {});
  await Bun.$`mv ${dir}/step_${n}.svg.png ${dir}/step_${n}.png`.quiet().catch(() => {});
  await Bun.$`rm -f ${dir}/step_${n}.svg`.quiet().catch(() => {});
}

console.log(`wrote ${frames.length} PNG frames to ${dir}`);
for (let i = 0; i < frames.length; i++) console.log(`  step_${String(i).padStart(3, "0")}.png  ${frames[i].label}`);
