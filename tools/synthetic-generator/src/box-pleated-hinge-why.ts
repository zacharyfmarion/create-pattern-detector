// CLI: explain why the hinge router stops (dead-ends) for a seed.
//   bun run src/box-pleated-hinge-why.ts <seed>
// Runs the real routeHinges, finds every "stuck" node (an ENTER that immediately
// backtracks - no frontier vertex was resolvable), and for each stuck frontier
// vertex prints every candidate hinge direction with the reason it was kept or
// rejected. Text only, so it is fast to run in a tight iteration loop.

import { cachedPacking } from "./box-pleated-packing-cache.ts";
import { buildPackingMolecule } from "./box-pleated-cp.ts";
import { routeHinges, explainHingeVertex, stuckNodes, type HingeTraceEvent } from "./box-pleated-assignment.ts";

const seed = Number(Bun.argv[2] ?? 60016);

const packing = await cachedPacking(seed);
const m = buildPackingMolecule(packing);

const events: HingeTraceEvent[] = [];
const t0 = performance.now();
const rays = routeHinges(m, m.sheet, (e) => events.push(e));
const ms = performance.now() - t0;
const solved = events.some((e) => e.kind === "solved");
console.log(`seed ${seed}: ${ms.toFixed(0)}ms, ${events.length} events, solved=${solved}, ${rays.flatMap((r) => r.path).length} hinge segs\n`);

const stuck = stuckNodes(events);
if (stuck.length === 0) {
  console.log("no stuck nodes (search never dead-ended on an unresolvable frontier).");
} else {
  // Deduplicate by (depth, frontier) so repeated visits to the same dead-end collapse.
  const seen = new Set<string>();
  let shown = 0;
  for (const e of stuck) {
    const key = `${e.frontier.map((p) => `${p.x},${p.y}`).sort().join("|")}`;
    if (seen.has(key)) continue;
    seen.add(key);
    console.log(`── stuck @ depth ${e.depth}, ${e.hinges.length} committed hinges, frontier [${e.frontier.map((p) => `${p.x},${p.y}`).join(" ")}]`);
    for (const v of e.frontier) {
      const d = explainHingeVertex(m, v, e.hinges, m.sheet);
      console.log(`  (${v.x},${v.y}) deg=${d.degree} ${d.parity} freeAxes=${d.freeAxes} :: ${d.summary}`);
      for (const c of d.candidates) {
        const term = c.terminus ? `${c.terminusType}@(${c.terminus.x},${c.terminus.y})` : `${c.terminusType}`;
        console.log(`      dir(${c.dir.x},${c.dir.y}) -> ${term.padEnd(20)} ${c.accepted ? "✓" : "✗"} ${c.reason}`);
      }
    }
    console.log("");
    if (++shown >= 6) { console.log(`(+${stuck.length - shown} more stuck nodes)`); break; }
  }
}
