import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import {
  alternatingSequence,
  port,
  sequenceToString,
  solvePortAssignmentProblem,
  type ConnectorState,
  type PortAssignment,
  type PortSolverProblem,
  type PortSolverResult,
  type RegionState,
} from "./bp-port-assignment-solver.ts";

interface Options {
  out: string;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const phaseProblem = phaseChoiceProblem();
  const phaseResult = solvePortAssignmentProblem(phaseProblem);
  const connectorProblem = connectorFallbackProblem();
  const connectorResult = solvePortAssignmentProblem(connectorProblem);
  await mkdir(dirname(options.out), { recursive: true });
  await writeFile(options.out, debugSvg(phaseResult, connectorResult));
  console.log(JSON.stringify({
    out: options.out,
    phaseAssignments: phaseResult.assignments,
    phaseJoins: phaseResult.joins,
    connectorAssignments: connectorResult.assignments,
    connectorJoins: connectorResult.joins,
  }, null, 2));
}

function parseArgs(args: string[]): Options {
  const options: Options = { out: "/tmp/bp-port-solver-debug/port-solver-debug.svg" };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--out") {
      options.out = requiredValue(args[++index], "--out");
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return options;
}

function phaseChoiceProblem(): PortSolverProblem {
  return {
    id: "phase-choice-debug",
    regions: [
      variable("hub", [state("hub", "hub-phase-0", "V", 0)], 0),
      variable("corridor", [
        state("corridor", "corridor-phase-0", "M", 0),
        state("corridor", "corridor-phase-1", "V", 1),
      ], 1),
    ],
    constraints: [{
      id: "hub-corridor",
      aRegion: "hub",
      aPort: "east",
      bRegion: "corridor",
      bPort: "west",
    }],
  };
}

function connectorFallbackProblem(): PortSolverProblem {
  const connector: ConnectorState = {
    id: "chevron-phase-shift",
    label: "chevron phase shift",
    from: port("from", alternatingSequence("V", 5), { width: 5 }),
    to: port("to", alternatingSequence("M", 5), { width: 5 }),
  };
  return {
    id: "connector-fallback-debug",
    regions: [
      variable("left-corridor", [state("left-corridor", "left-locked", "V", 0)], 0),
      variable("right-corridor", [state("right-corridor", "right-locked", "M", 0)], 1),
    ],
    constraints: [{
      id: "corridor-corridor",
      aRegion: "left-corridor",
      aPort: "east",
      bRegion: "right-corridor",
      bPort: "west",
      connectorStates: [connector],
    }],
  };
}

function debugSvg(phaseResult: PortSolverResult, connectorResult: PortSolverResult): string {
  const width = 1800;
  const height = 840;
  const panelWidth = 540;
  const panelHeight = 610;
  const gap = 28;
  const top = 128;
  const left = 34;
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" shape-rendering="geometricPrecision">`,
    `<rect width="${width}" height="${height}" fill="#f8fafc"/>`,
    `<text x="${left}" y="46" font-family="Inter, Arial, sans-serif" font-size="28" font-weight="800" fill="#0f172a">BP port/phase assignment solver - first slice</text>`,
    `<text x="${left}" y="76" font-family="Inter, Arial, sans-serif" font-size="15" fill="#475569">This is not a full CP yet. It visualizes how the compiler should reconcile M/V phases at molecule ports before emitting final crease geometry.</text>`,
    legend(left, 100),
    panel(left, top, panelWidth, panelHeight, "1. Port phase candidates", [
      `<text x="28" y="76" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">Hub requires VMVMV on its east port.</text>`,
      portBar(38, 112, alternatingSequence("V", 5), "hub east", true),
      `<text x="28" y="210" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">Corridor phase 0 exposes MVMVM: rejected.</text>`,
      portBar(38, 246, alternatingSequence("M", 5), "corridor phase 0", false),
      mismatchMarks(38, 246, alternatingSequence("V", 5), alternatingSequence("M", 5)),
      `<text x="28" y="382" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">Corridor phase 1 exposes VMVMV: compatible.</text>`,
      portBar(38, 418, alternatingSequence("V", 5), "corridor phase 1", true),
    ]),
    panel(left + panelWidth + gap, top, panelWidth, panelHeight, "2. Solver-selected state", [
      `<text x="28" y="76" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">The solver selects a whole corridor state, not edge-by-edge recolors.</text>`,
      stateCard(36, 118, "hub", phaseResult.assignments.hub ?? "missing", alternatingSequence("V", 5), "chosen"),
      joinArrow(220, 198, 278, 198, true, "direct join"),
      stateCard(290, 118, "corridor", phaseResult.assignments.corridor ?? "missing", alternatingSequence("V", 5), "chosen"),
      traceSummary(36, 330, phaseResult),
    ]),
    panel(left + (panelWidth + gap) * 2, top, panelWidth, panelHeight, "3. Connector fallback", [
      `<text x="28" y="76" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">When locked phases disagree, a certified connector can bridge them.</text>`,
      stateCard(28, 128, "left", connectorResult.assignments["left-corridor"] ?? "missing", alternatingSequence("V", 5), "chosen"),
      connectorGlyph(207, 169),
      stateCard(292, 128, "right", connectorResult.assignments["right-corridor"] ?? "missing", alternatingSequence("M", 5), "chosen"),
      `<text x="52" y="356" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">join mode: ${escapeXml(connectorResult.joins[0]?.mode ?? "missing")}</text>`,
      `<text x="52" y="384" font-family="Inter, Arial, sans-serif" font-size="14" fill="#334155">connector: ${escapeXml(connectorResult.joins[0]?.connectorId ?? "none")}</text>`,
      `<text x="52" y="432" font-family="Inter, Arial, sans-serif" font-size="13" fill="#64748b">Next: replace toy states with real BP molecules.</text>`,
      `<text x="52" y="454" font-family="Inter, Arial, sans-serif" font-size="13" fill="#64748b">Their ports will come from BP Studio macro regions.</text>`,
    ]),
    `</svg>`,
  ].join("\n");
}

function panel(x: number, y: number, width: number, height: number, title: string, body: string[]): string {
  return [
    `<g transform="translate(${x},${y})">`,
    `<rect width="${width}" height="${height}" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.3"/>`,
    `<text x="24" y="40" font-family="Inter, Arial, sans-serif" font-size="20" font-weight="800" fill="#0f172a">${escapeXml(title)}</text>`,
    ...body,
    `</g>`,
  ].join("\n");
}

function legend(x: number, y: number): string {
  return [
    `<g transform="translate(${x},${y})">`,
    `<rect x="0" y="-18" width="550" height="36" rx="8" fill="#ffffff" stroke="#e2e8f0"/>`,
    swatch(16, 0, "M"),
    `<text x="42" y="5" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">mountain lane</text>`,
    swatch(158, 0, "V"),
    `<text x="184" y="5" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">valley lane</text>`,
    `<rect x="300" y="-10" width="20" height="20" rx="5" fill="#dcfce7" stroke="#22c55e"/>`,
    `<text x="330" y="5" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">compatible / chosen</text>`,
    `<rect x="456" y="-10" width="20" height="20" rx="5" fill="#fee2e2" stroke="#ef4444"/>`,
    `<text x="486" y="5" font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">rejected</text>`,
    `</g>`,
  ].join("\n");
}

function portBar(x: number, y: number, sequence: PortAssignment[], label: string, accepted: boolean): string {
  const cell = 42;
  return [
    `<g transform="translate(${x},${y})">`,
    `<rect x="-12" y="-24" width="${sequence.length * cell + 24}" height="82" rx="8" fill="${accepted ? "#dcfce7" : "#fee2e2"}" stroke="${accepted ? "#22c55e" : "#ef4444"}" stroke-width="1.4"/>`,
    `<text x="0" y="-6" font-family="Inter, Arial, sans-serif" font-size="12" fill="#334155">${escapeXml(label)} (${sequenceToString(sequence)})</text>`,
    ...sequence.map((assignment, index) => swatch(index * cell, 28, assignment)),
    `</g>`,
  ].join("\n");
}

function stateCard(x: number, y: number, label: string, stateId: string, sequence: PortAssignment[], status: "chosen" | "rejected"): string {
  const stroke = status === "chosen" ? "#22c55e" : "#ef4444";
  const fill = status === "chosen" ? "#f0fdf4" : "#fef2f2";
  return [
    `<g transform="translate(${x},${y})">`,
    `<rect width="160" height="156" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>`,
    `<text x="16" y="26" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="800" fill="#0f172a">${escapeXml(label)}</text>`,
    `<text x="16" y="50" font-family="Inter, Arial, sans-serif" font-size="12" fill="#475569">${escapeXml(stateId)}</text>`,
    ...sequence.map((assignment, index) => swatch(18 + index * 26, 90, assignment, 20)),
    `<text x="16" y="132" font-family="Inter, Arial, sans-serif" font-size="12" fill="#475569">${sequenceToString(sequence)}</text>`,
    `</g>`,
  ].join("\n");
}

function swatch(x: number, y: number, assignment: PortAssignment, size = 20): string {
  const fill = assignment === "M" ? "#ef4444" : "#2563eb";
  return `<rect x="${x}" y="${y - size / 2}" width="${size}" height="${size}" rx="4" fill="${fill}"/><text x="${x + size / 2}" y="${y + 4}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="${Math.round(size * 0.55)}" font-weight="800" fill="#ffffff">${assignment}</text>`;
}

function mismatchMarks(x: number, y: number, expected: PortAssignment[], actual: PortAssignment[]): string {
  const cell = 42;
  return [
    `<g transform="translate(${x},${y})">`,
    ...expected.map((value, index) => value === actual[index]
      ? ""
      : `<line x1="${index * cell - 4}" y1="70" x2="${index * cell + 24}" y2="98" stroke="#ef4444" stroke-width="3"/><line x1="${index * cell + 24}" y1="70" x2="${index * cell - 4}" y2="98" stroke="#ef4444" stroke-width="3"/>`),
    `</g>`,
  ].join("\n");
}

function joinArrow(x1: number, y1: number, x2: number, y2: number, ok: boolean, label: string): string {
  const color = ok ? "#22c55e" : "#ef4444";
  return [
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="4" stroke-linecap="round"/>`,
    `<path d="M ${x2} ${y2} l -12 -7 l 0 14 z" fill="${color}"/>`,
    `<text x="${(x1 + x2) / 2}" y="${y1 - 12}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="700" fill="${color}">${escapeXml(label)}</text>`,
  ].join("\n");
}

function connectorGlyph(x: number, y: number): string {
  return [
    `<g transform="translate(${x},${y})">`,
    `<path d="M 0 28 L 44 0 L 88 28 L 44 56 Z" fill="#f5d0fe" stroke="#a855f7" stroke-width="2"/>`,
    `<line x1="0" y1="28" x2="88" y2="28" stroke="#a855f7" stroke-width="2"/>`,
    `<line x1="44" y1="0" x2="44" y2="56" stroke="#a855f7" stroke-width="2"/>`,
    `<text x="44" y="82" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" font-weight="700" fill="#7e22ce">phase shifter</text>`,
    `</g>`,
  ].join("\n");
}

function traceSummary(x: number, y: number, result: PortSolverResult): string {
  const lines = result.trace
    .filter((event) => event.event === "select-region" || event.event === "accept-state" || event.event === "solution")
    .slice(0, 6)
    .map((event) => `${event.event}${event.regionId ? ` ${event.regionId}` : ""}${event.stateId ? `=${event.stateId}` : ""}`);
  return [
    `<g transform="translate(${x},${y})">`,
    `<rect width="386" height="144" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>`,
    `<text x="16" y="26" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="800" fill="#0f172a">solver trace</text>`,
    ...lines.map((line, index) => `<text x="16" y="${52 + index * 18}" font-family="Menlo, Consolas, monospace" font-size="12" fill="#334155">${escapeXml(line)}</text>`),
    `</g>`,
  ].join("\n");
}

function variable(id: string, states: RegionState[], rank?: number): PortSolverProblem["regions"][number] {
  return { id, rank, states };
}

function state(regionId: string, id: string, start: PortAssignment, phase: number): RegionState {
  return {
    id,
    regionId,
    phase,
    cost: phase,
    ports: [
      port("west", alternatingSequence(start, 5), { side: "left", width: 5 }),
      port("east", alternatingSequence(start, 5), { side: "right", width: 5 }),
    ],
  };
}

function requiredValue(value: string | undefined, flag: string): string {
  if (!value) throw new Error(`${flag} requires a value`);
  return value;
}

function escapeXml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
