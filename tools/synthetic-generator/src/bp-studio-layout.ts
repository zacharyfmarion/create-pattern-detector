import { join } from "node:path";

import type { BoxPleatedBounds } from "./box-pleated-packing.ts";

export interface BpStudioLayoutPoint {
  x: number;
  y: number;
}

export type BpStudioLayoutLine = [BpStudioLayoutPoint, BpStudioLayoutPoint];

export interface BpStudioLayoutContour {
  outer: BpStudioLayoutPoint[];
  inner?: BpStudioLayoutPoint[][];
}

export interface BpStudioLayoutGraphics {
  contours: BpStudioLayoutContour[];
  ridges: BpStudioLayoutLine[];
  axisParallel?: BpStudioLayoutLine[];
}

export interface BpStudioLayoutInput {
  bpStudioRoot?: string;
  edges: Array<{ n1: number; n2: number; length: number }>;
  flaps: Array<{ id: number; x: number; y: number; width: number; height: number }>;
  sheet: BoxPleatedBounds;
}

export interface BpStudioLayoutResult {
  patternNotFound: boolean;
  graphics: Record<string, BpStudioLayoutGraphics>;
}

const DEFAULT_BP_STUDIO_ROOT = "/tmp/bp-studio-source";

const HEADLESS_BP_STUDIO_LAYOUT_SCRIPT = `
const input = await new Response(Bun.stdin.stream()).json();
const { Tree } = await import("core/design/context/tree");
const { heightTask } = await import("core/design/tasks/height");
const { Processor } = await import("core/service/processor");
const { State, fullReset } = await import("core/service/state");
const { UpdateResult } = await import("core/service/updateResult");

fullReset();
State.m.$tree = new Tree(input.edges, input.flaps);
Processor.$run(heightTask);
const result = UpdateResult.$flush();
console.log(JSON.stringify({
  patternNotFound: result.patternNotFound,
  graphics: result.graphics,
}));
`;

export async function runBpStudioLayout(input: BpStudioLayoutInput): Promise<BpStudioLayoutResult> {
  const root = input.bpStudioRoot ?? process.env.BP_STUDIO_ROOT ?? DEFAULT_BP_STUDIO_ROOT;
  const proc = Bun.spawn({
    cmd: ["bun", "--eval", HEADLESS_BP_STUDIO_LAYOUT_SCRIPT],
    cwd: root,
    env: {
      ...process.env,
      NODE_PATH: join(root, "src"),
    },
    stdin: "pipe",
    stdout: "pipe",
    stderr: "pipe",
  });

  await proc.stdin.write(JSON.stringify({
    edges: input.edges,
    flaps: input.flaps,
    sheet: input.sheet,
  }));
  proc.stdin.end();

  const [exitCode, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ]);
  if (exitCode !== 0) {
    throw new Error(`BP Studio layout runner failed with exit ${exitCode}: ${stderr.trim() || stdout.trim()}`);
  }

  return JSON.parse(stdout) as BpStudioLayoutResult;
}
