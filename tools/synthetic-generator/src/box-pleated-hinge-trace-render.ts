// CLI: render the hinge router's full backtracking search as a PNG frame sequence.
//   bun run src/box-pleated-hinge-trace-render.ts <seed> [maxFrames] [pixels]
// Frames land in /tmp/hinge-trace-<seed>/step_NNN.png, one per search event
// (enter / try / backtrack / solved / exhausted). Uses the real routeHinges via
// its onEvent callback, so the trace is exactly the production search - including
// every backtrack.

import { generateBoxPleatedPacking } from "./box-pleated-packing.ts";
import { buildPackingMolecule } from "./box-pleated-cp.ts";
import { routeHinges, type HingeTraceEvent } from "./box-pleated-assignment.ts";
import { renderHingeTraceFrames } from "./box-pleated-hinge-trace.ts";

const seed = Number(Bun.argv[2] ?? 60115);
const maxFrames = Number(Bun.argv[3] ?? 500);
const pixels = Number(Bun.argv[4] ?? 900);
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

const m = buildPackingMolecule(packing); // throws on a rejected packing (nothing to trace)

const events: HingeTraceEvent[] = [];
const hinges = routeHinges(m, m.sheet, (e) => {
  if (events.length < maxFrames) events.push(e);
});
const solved = events.some((e) => e.kind === "solved");
console.log(`routeHinges: ${events.length}${events.length >= maxFrames ? "+ (capped)" : ""} events, ${hinges.length} hinges placed, solved=${solved}`);

const frames = renderHingeTraceFrames(events, m, { pixels });
const dir = `/tmp/hinge-trace-${seed}`;
await Bun.$`rm -rf ${dir}`.quiet().catch(() => {});
await Bun.$`mkdir -p ${dir}`.quiet();
for (let i = 0; i < frames.length; i++) await Bun.write(`${dir}/step_${String(i).padStart(4, "0")}.svg`, frames[i].svg);
// One batched rasterisation pass (far faster than one qlmanage per frame).
await Bun.$`sh -c ${`qlmanage -t -s ${pixels} -o ${dir} ${dir}/*.svg`}`.quiet().catch(() => {});
await Bun.$`sh -c ${`rm -f ${dir}/*.svg`}`.quiet();
await Bun.$`sh -c ${`for f in ${dir}/*.svg.png; do mv "$f" "\${f%.svg.png}.png"; done`}`.quiet().catch(() => {});
console.log(`wrote ${frames.length} frames to ${dir}`);
