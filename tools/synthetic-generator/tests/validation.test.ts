import { expect, test } from "bun:test";
import type { FOLDFormat, ValidationConfig } from "../src/types.ts";
import { validateFold } from "../src/validate.ts";

const validation: ValidationConfig = {
  strictGlobal: false,
  globalBackend: "rabbit-ear-solver",
  minVertexDistance: 1e-5,
  maxVertices: 16,
  maxEdges: 32,
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

test("dense validation rejects too-simple graphs", async () => {
  const result = await validateFold(
    {
      ...square(),
      density_metadata: {
        densityBucket: "small",
        gridSize: 0,
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
