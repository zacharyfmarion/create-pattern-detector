import { expect, test } from "bun:test";
import {
  alternatingSequence,
  compatiblePorts,
  port,
  sequenceToString,
  solvePortAssignmentProblem,
  type PortSolverProblem,
  type RegionState,
} from "../src/bp-port-assignment-solver.ts";

test("port compatibility checks width parity orientation and exposed M/V sequence", () => {
  const base = port("p", alternatingSequence("V", 5), { width: 5, parity: "integer", orientation: "horizontal" });
  expect(compatiblePorts(base, port("q", alternatingSequence("V", 5), { width: 5, parity: "integer", orientation: "horizontal" })).ok).toBe(true);
  expect(compatiblePorts(base, port("q", alternatingSequence("M", 5), { width: 5, parity: "integer", orientation: "horizontal" })).reason).toStartWith("sequence");
  expect(compatiblePorts(base, port("q", alternatingSequence("V", 5), { width: 4, parity: "integer", orientation: "horizontal" })).reason).toStartWith("width");
  expect(compatiblePorts(base, port("q", alternatingSequence("V", 5), { width: 5, parity: "half", orientation: "horizontal" })).reason).toStartWith("parity");
  expect(compatiblePorts(base, port("q", alternatingSequence("V", 5), { width: 5, parity: "integer", orientation: "vertical" })).reason).toStartWith("orientation");
});

test("solver chooses a compatible corridor phase instead of recoloring individual creases", () => {
  const problem: PortSolverProblem = {
    id: "phase-flip-fixture",
    regions: [
      variable("hub", [
        state("hub", "hub-phase-0", "V", 0),
      ], 0),
      variable("corridor", [
        state("corridor", "corridor-phase-0", "M", 0),
        state("corridor", "corridor-phase-1", "V", 1),
      ], 1),
    ],
    constraints: [
      {
        id: "hub-corridor",
        aRegion: "hub",
        aPort: "east",
        bRegion: "corridor",
        bPort: "west",
      },
    ],
  };

  const result = solvePortAssignmentProblem(problem);
  expect(result.ok).toBe(true);
  expect(result.assignments).toEqual({
    hub: "hub-phase-0",
    corridor: "corridor-phase-1",
  });
  expect(result.joins).toEqual([{
    constraintId: "hub-corridor",
    aRegion: "hub",
    aState: "hub-phase-0",
    aPort: "east",
    bRegion: "corridor",
    bState: "corridor-phase-1",
    bPort: "west",
    mode: "direct",
  }]);
});

test("solver inserts a connector when both locked phases are incompatible", () => {
  const problem: PortSolverProblem = {
    id: "connector-fixture",
    regions: [
      variable("left-corridor", [state("left-corridor", "left-locked", "V", 0)], 0),
      variable("right-corridor", [state("right-corridor", "right-locked", "M", 0)], 1),
    ],
    constraints: [
      {
        id: "corridor-corridor",
        aRegion: "left-corridor",
        aPort: "east",
        bRegion: "right-corridor",
        bPort: "west",
        connectorStates: [{
          id: "chevron-phase-shift",
          label: "chevron phase shift",
          from: port("from", alternatingSequence("V", 5), { width: 5 }),
          to: port("to", alternatingSequence("M", 5), { width: 5 }),
        }],
      },
    ],
  };

  const result = solvePortAssignmentProblem(problem);
  expect(result.ok).toBe(true);
  expect(result.joins).toHaveLength(1);
  expect(result.joins[0].mode).toBe("connector");
  expect(result.joins[0].connectorId).toBe("chevron-phase-shift");
});

test("solver rejects layouts whose domains cannot satisfy port constraints", () => {
  const problem: PortSolverProblem = {
    id: "unsat-width-fixture",
    regions: [
      variable("hub", [state("hub", "hub-wide", "V", 0, 5)], 0),
      variable("corridor", [state("corridor", "corridor-narrow", "V", 0, 3)], 1),
    ],
    constraints: [
      {
        id: "hub-corridor",
        aRegion: "hub",
        aPort: "east",
        bRegion: "corridor",
        bPort: "west",
      },
    ],
  };

  const result = solvePortAssignmentProblem(problem);
  expect(result.ok).toBe(false);
  expect(result.errors).toContain("port-solver-unsat");
});

test("solver uses minimum remaining values before broad domains", () => {
  const problem: PortSolverProblem = {
    id: "mrv-fixture",
    regions: [
      variable("broad-a", [
        state("broad-a", "a0", "V", 0),
        state("broad-a", "a1", "M", 1),
      ], 10),
      variable("forced-hub", [
        state("forced-hub", "hub0", "V", 0),
      ], 10),
      variable("broad-b", [
        state("broad-b", "b0", "V", 0),
        state("broad-b", "b1", "M", 1),
      ], 10),
    ],
    constraints: [],
  };

  const result = solvePortAssignmentProblem(problem);
  const firstSelection = result.trace.find((event) => event.event === "select-region");
  expect(firstSelection?.regionId).toBe("forced-hub");
});

test("sequence helper makes alternating pleat lanes explicit", () => {
  expect(sequenceToString(alternatingSequence("V", 5))).toBe("VMVMV");
  expect(sequenceToString(alternatingSequence("M", 6))).toBe("MVMVMV");
});

function variable(id: string, states: RegionState[], rank?: number): PortSolverProblem["regions"][number] {
  return { id, rank, states };
}

function state(regionId: string, id: string, start: "M" | "V", phase: number, width = 5): RegionState {
  return {
    id,
    regionId,
    phase,
    cost: phase,
    ports: [
      port("west", alternatingSequence(start, width), { side: "left", width }),
      port("east", alternatingSequence(start, width), { side: "right", width }),
    ],
  };
}
