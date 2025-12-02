/**
 * Rabbit Ear Generator (Correct API)
 *
 * Uses Rabbit Ear's actual working APIs
 */

import ear from 'rabbit-ear';
import type { FOLDFormat, GenerationConfig } from '../types/crease-pattern.ts';

const rand = (a: number = 0, b: number = 1) => a + Math.random() * (b - a);
const choice = <T>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const randPoint = (pad: number = 0.05): [number, number] => [rand(pad, 1 - pad), rand(pad, 1 - pad)];

/**
 * Check if two line segments intersect (CCW test method)
 */
function segmentsIntersect(
  a: [number, number],
  b: [number, number],
  c: [number, number],
  d: [number, number]
): boolean {
  function ccw(p1: [number, number], p2: [number, number], p3: [number, number]): boolean {
    return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0]);
  }
  return ccw(a, c, d) !== ccw(b, c, d) && ccw(a, b, c) !== ccw(a, b, d);
}

/**
 * Check if two edges share a vertex
 */
function sharesVertex(edge1: [number, number], edge2: [number, number]): boolean {
  return edge1[0] === edge2[0] || edge1[0] === edge2[1] ||
         edge1[1] === edge2[0] || edge1[1] === edge2[1];
}

/**
 * Check if a graph has self-intersections
 */
function hasSelfIntersections(graph: any): boolean {
  if (!graph.edges_vertices || graph.edges_vertices.length === 0 || !graph.vertices_coords) {
    return false;
  }

  // Check all edge pairs for intersection
  for (let i = 0; i < graph.edges_vertices.length; i++) {
    for (let j = i + 1; j < graph.edges_vertices.length; j++) {
      const edge1 = graph.edges_vertices[i];
      const edge2 = graph.edges_vertices[j];

      // Skip if edges share a vertex
      if (sharesVertex(edge1, edge2)) {
        continue;
      }

      // Check for intersection
      const v1 = graph.vertices_coords[edge1[0]];
      const v2 = graph.vertices_coords[edge1[1]];
      const v3 = graph.vertices_coords[edge2[0]];
      const v4 = graph.vertices_coords[edge2[1]];

      if (segmentsIntersect(v1, v2, v3, v4)) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Method A: Constructive folds via axioms with validation
 */
export function genByAxioms(config: GenerationConfig): FOLDFormat {
  const { numCreases } = config;

  // Dynamic fold target based on desired complexity
  // More creases = more folds, but with diminishing returns
  const targetFolds = Math.min(Math.ceil(numCreases / 10), 50);

  // Adaptive retry attempts - more for complex patterns
  const maxAttemptsPerFold = numCreases > 100 ? 20 : 10;

  const paper = ear.origami();
  let successfulFolds = 0;
  let consecutiveFailures = 0;

  for (let i = 0; i < targetFolds * 5 && successfulFolds < targetFolds; i++) {
    let foldApplied = false;

    for (let attempt = 0; attempt < maxAttemptsPerFold; attempt++) {
      // Save current state
      const previousGraph = JSON.parse(JSON.stringify(paper));

      // Use simpler axioms more frequently (1,2,4 are simpler than 5,6,7)
      const r = Math.random();
      const axiomNum = r < 0.6 ? (Math.random() < 0.5 ? 1 : 2) : Math.floor(rand(1, 8));

      let line;
      try {
        switch (axiomNum) {
          case 1: // Fold through two points
            line = ear.axiom.axiom1(randPoint(), randPoint());
            break;
          case 2: // Fold point onto point
            line = ear.axiom.axiom2(randPoint(), randPoint());
            break;
          case 3: {
            // Fold line onto line - need two lines
            const l1 = { origin: randPoint(), vector: [rand(-1, 1), rand(-1, 1)] };
            const l2 = { origin: randPoint(), vector: [rand(-1, 1), rand(-1, 1)] };
            const result = ear.axiom.axiom3(l1, l2);
            line = Array.isArray(result) ? result[0] : result;
            break;
          }
          case 4: {
            // Point onto line
            const l = { origin: randPoint(), vector: [rand(-1, 1), rand(-1, 1)] };
            line = ear.axiom.axiom4(l, randPoint());
            break;
          }
          case 5: {
            // Point onto line through point
            const l = { origin: randPoint(), vector: [rand(-1, 1), rand(-1, 1)] };
            const result = ear.axiom.axiom5(l, randPoint(), randPoint());
            line = Array.isArray(result) ? result[0] : result;
            break;
          }
          case 6: {
            // Point onto point onto line
            const l1 = { origin: randPoint(), vector: [rand(-1, 1), rand(-1, 1)] };
            const l2 = { origin: randPoint(), vector: [rand(-1, 1), rand(-1, 1)] };
            const result = ear.axiom.axiom6(l1, l2, randPoint(), randPoint());
            line = Array.isArray(result) ? result[0] : result;
            break;
          }
          case 7: {
            // Point onto line perpendicular to line
            const l1 = { origin: randPoint(), vector: [rand(-1, 1), rand(-1, 1)] };
            const l2 = { origin: randPoint(), vector: [rand(-1, 1), rand(-1, 1)] };
            const result = ear.axiom.axiom7(l1, l2, randPoint());
            line = Array.isArray(result) ? result[0] : result;
            break;
          }
          default:
            line = ear.axiom.axiom1(randPoint(), randPoint());
        }

        if (line) {
          paper.flatFold(line);

          // Check for self-intersections
          if (!hasSelfIntersections(paper)) {
            // Fold is valid, keep it
            successfulFolds++;
            foldApplied = true;
            consecutiveFailures = 0;
            break;
          } else {
            // Fold created intersections, restore previous state
            Object.assign(paper, previousGraph);
          }
        }
      } catch (e) {
        // Axiom failed, restore and try again
        Object.assign(paper, previousGraph);
        continue;
      }
    }

    // If we couldn't apply a fold after max attempts, continue to next
    if (!foldApplied) {
      consecutiveFailures++;

      // If we've failed too many times in a row, we're probably stuck
      // Accept what we have rather than spinning forever
      if (consecutiveFailures >= 20) {
        break;
      }
      continue;
    }
  }

  // Scale to requested dimensions
  return scaleGraph(paper, config.width, config.height);
}

/**
 * Method B: CP canvas with segments
 */
export function genWithCPCanvas(config: GenerationConfig): FOLDFormat {
  const { numCreases } = config;

  const cp = ear.cp();

  // Add random segments
  for (let i = 0; i < numCreases; i++) {
    const p1 = randPoint();
    const p2 = randPoint();

    // Ensure some minimum length
    const dist = Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
    if (dist < 0.1) continue;

    const seg = cp.segment(p1[0], p1[1], p2[0], p2[1]);

    // Random M/V
    if (Math.random() < 0.5) {
      seg.mountain();
    } else {
      seg.valley();
    }
  }

  // Scale to requested dimensions
  return scaleGraph(cp, config.width, config.height);
}

/**
 * Scale graph from unit square to requested dimensions
 */
function scaleGraph(graph: any, width: number, height: number): FOLDFormat {
  const scaled: any = {
    ...graph,
    vertices_coords: graph.vertices_coords.map((v: number[]) => [
      v[0] * width,
      v[1] * height
    ])
  };

  scaled.file_spec = 1.1;
  scaled.file_creator = scaled.file_creator || 'rabbit-ear-generator';

  return scaled as FOLDFormat;
}

/**
 * Main generator
 */
export function generateRabbitEarCP(config: GenerationConfig): FOLDFormat {
  // Only use axioms - cp canvas creates disconnected line segments
  return genByAxioms(config);
}
