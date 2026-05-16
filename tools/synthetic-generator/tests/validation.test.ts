import { expect, test } from "bun:test";
import { arrangeSegments } from "../src/line-arrangement.ts";
import type { FOLDFormat, ValidationConfig } from "../src/types.ts";
import { validateFold } from "../src/validate.ts";

const validation: ValidationConfig = {
  strictGlobal: false,
  globalBackend: "rabbit-ear-solver",
  minVertexDistance: 1e-5,
  maxVertices: 16,
  maxEdges: 32,
};

const bpValidation: ValidationConfig = {
  ...validation,
  maxVertices: 64,
  maxEdges: 128,
  requireBoxPleat: true,
};

const denseValidation: ValidationConfig = {
  strictGlobal: false,
  globalBackend: "rabbit-ear-solver",
  minVertexDistance: 1e-5,
  maxVertices: 256,
  maxEdges: 512,
  requireDense: true,
};

test("validation rejects missing borders", async () => {
  const result = await validateFold({ ...square(), edges_assignment: ["U", "U", "U", "U"] }, validation);
  expect(result.failed).toContain("complete-border");
});

test("validation rejects degenerate edges", async () => {
  const fold = square();
  fold.edges_vertices[0] = [0, 0];
  const result = await validateFold(fold, validation);
  expect(result.failed).toContain("edge-geometry");
});

test("validation rejects crossing edges that do not meet at vertices", async () => {
  const result = await validateFold(
    {
      ...square(),
      vertices_coords: [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [0, 0.5],
        [1, 0.5],
        [0.5, 0],
        [0.5, 1],
      ],
      edges_vertices: [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [6, 7],
      ],
      edges_assignment: ["B", "B", "B", "B", "M", "V"],
    },
    validation,
  );
  expect(result.failed).toContain("no-self-intersections");
});

test("validation rejects close vertex collapse", async () => {
  const result = await validateFold(
    {
      ...square(),
      vertices_coords: [
        [0, 0],
        [0.000001, 0],
        [1, 1],
        [0, 1],
      ],
    },
    validation,
  );
  expect(result.failed).toContain("complexity-bounds");
});

test("validation rejects over-complex graphs", async () => {
  const fold = square();
  const result = await validateFold(
    {
      ...fold,
      vertices_coords: [
        ...fold.vertices_coords,
        [0.25, 0.25],
        [0.75, 0.75],
      ],
    },
    { ...validation, maxVertices: 4 },
  );
  expect(result.failed).toContain("complexity-bounds");
});

test("box-pleat validation rejects missing ridges", async () => {
  const fold = bpFixture();
  fold.edges_bpRole = fold.edges_bpRole?.map((role) => (role === "ridge" ? "hinge" : role));
  fold.edges_assignment = fold.edges_assignment.map((assignment) => (assignment === "M" ? "V" : assignment));
  const result = await validateFold(fold, bpValidation);
  expect(result.failed).toContain("box-pleat-structure");
});

test("box-pleat validation rejects missing hinges", async () => {
  const fold = bpFixture();
  fold.edges_bpRole = fold.edges_bpRole?.map((role) => (role === "hinge" ? "axis" : role));
  const result = await validateFold(fold, bpValidation);
  expect(result.failed).toContain("box-pleat-structure");
});

test("box-pleat validation rejects duplicate segments", async () => {
  const fold = bpFixture();
  fold.edges_vertices.push(fold.edges_vertices[0]);
  fold.edges_assignment.push(fold.edges_assignment[0]);
  fold.edges_bpRole?.push(fold.edges_bpRole[0]);
  const result = await validateFold(fold, bpValidation);
  expect(result.failed).toContain("edge-geometry");
});

test("box-pleat validation rejects non-grid coordinates", async () => {
  const fold = bpFixture();
  fold.vertices_coords[0] = [0.13, 0];
  const result = await validateFold(fold, bpValidation);
  expect(result.failed).toContain("box-pleat-structure");
});

test("dense validation rejects too-simple graphs", async () => {
  const result = await validateFold(
    {
      ...square(),
      density_metadata: {
        densityBucket: "small",
        gridSize: 8,
        targetEdgeRange: [80, 350],
        subfamily: "recursive-axiom",
        symmetry: "test",
        generatorSteps: ["test"],
        moleculeCounts: {},
      },
    },
    denseValidation,
  );
  expect(result.failed).toContain("dense-structure");
});

test("Rabbit Ear fold-program validation rejects missing metadata", async () => {
  const result = await validateFold(square(), {
    ...validation,
    requireRabbitEarFoldProgram: true,
  });
  expect(result.failed).toContain("rabbit-ear-fold-program-structure");
});

test("Rabbit Ear fold-program validation rejects sparse outputs", async () => {
  const fold: FOLDFormat = {
    ...square(),
    rabbit_ear_metadata: {
      generator: "rabbit-ear-fold-program",
      rabbitEarApi: "ear.graph.flatFold",
      appliedFoldCount: 0,
      attemptedFoldCount: 0,
      axiomUsage: {},
      activeCreaseCount: 0,
      targetActiveCreases: 40,
      targetActiveCreaseRange: [40, 90],
      requestedBucket: "small",
    },
    label_policy: {
      labelSource: "rabbit-ear-fold-program",
      geometrySource: "rabbit-ear-fold-program",
      assignmentSource: "rabbit-ear-fold-program",
      trainingEligible: true,
      notes: [],
    },
  };
  const result = await validateFold(fold, {
    ...validation,
    requireRabbitEarFoldProgram: true,
  });
  expect(result.failed).toContain("rabbit-ear-fold-program-structure");
});

test("realistic BP validation rejects missing design metadata", async () => {
  const fold = bpFixture();
  fold.density_metadata = {
    densityBucket: "small",
    gridSize: 4,
    targetEdgeRange: [4, 128],
    subfamily: "realistic-tree-base",
    symmetry: "test",
    generatorSteps: ["test"],
    moleculeCounts: {},
  };
  const result = await validateFold(fold, {
    ...bpValidation,
    requireDense: true,
    requireRealistic: true,
    boxPleatMode: "dense",
  });
  expect(result.failed).toContain("realistic-structure");
});

test("dense BP validation rejects missing diagonal structure", async () => {
  const fold = bpFixture();
  fold.density_metadata = {
    densityBucket: "small",
    gridSize: 4,
    targetEdgeRange: [4, 128],
    subfamily: "dense-molecule-tessellation",
    symmetry: "test",
    generatorSteps: ["test"],
    moleculeCounts: {},
  };
  fold.edges_bpRole = fold.edges_bpRole?.map((role) => (role === "ridge" ? "hinge" : role));
  fold.edges_assignment = fold.edges_assignment.map((assignment) => (assignment === "M" ? "V" : assignment));
  const result = await validateFold(fold, {
    ...denseValidation,
    requireBoxPleat: true,
    boxPleatMode: "dense",
  });
  expect(result.failed).toContain("box-pleat-structure");
});

function square(): FOLDFormat {
  return {
    file_spec: 1.1,
    file_creator: "test",
    file_classes: ["singleModel"],
    frame_classes: ["creasePattern"],
    vertices_coords: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ],
    edges_vertices: [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 0],
    ],
    edges_assignment: ["B", "B", "B", "B"],
  };
}

function bpFixture(): FOLDFormat {
  const fold = arrangeSegments(
    [
      { p1: [0, 0], p2: [1, 0], assignment: "B", role: "border" },
      { p1: [1, 0], p2: [1, 1], assignment: "B", role: "border" },
      { p1: [1, 1], p2: [0, 1], assignment: "B", role: "border" },
      { p1: [0, 1], p2: [0, 0], assignment: "B", role: "border" },
      { p1: [0.25, 0], p2: [0.25, 1], assignment: "V", role: "hinge" },
      { p1: [0, 0.5], p2: [1, 0.5], assignment: "V", role: "axis" },
      { p1: [0, 0], p2: [1, 1], assignment: "M", role: "ridge" },
    ],
    "test/bp-fixture",
    {
      gridSize: 4,
      bpSubfamily: "two-flap-stretch",
      flapCount: 2,
      gadgetCount: 1,
      ridgeCount: 1,
      hingeCount: 1,
      axisCount: 1,
    },
  );
  return fold;
}
