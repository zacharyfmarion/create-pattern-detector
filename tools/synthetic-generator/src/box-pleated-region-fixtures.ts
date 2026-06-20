// Fixtures for the gap-filling packing problem.
//
// An empty region left by BP Studio is a rectangle whose four sides are each
// either the PAPER BOUNDARY (the flap molecule may sit cropped against it, with
// its center on the edge) or PAPER INTERIOR (bordered by another flap/river, so
// the flap must be fully self-contained). We must tile the region exactly with
// valid flaps. These fixtures cover the distinct configurations so solutions can
// be inspected visually.

export interface RegionEdges {
  /** True when this side coincides with the paper boundary. */
  left: boolean;
  right: boolean;
  top: boolean;
  bottom: boolean;
}

export interface RegionFixture {
  name: string;
  width: number;
  height: number;
  edges: RegionEdges;
  /** What we expect: solvable, or known-impossible. */
  note: string;
}

const interior = (): RegionEdges => ({ left: false, right: false, top: false, bottom: false });

export const regionFixtures: RegionFixture[] = [
  // --- fully interior (all sides border other flaps/rivers) ---
  { name: "interior-8x6-even", width: 8, height: 6, edges: interior(), note: "even x even - single flap" },
  { name: "interior-7x10-odd-even", width: 7, height: 10, edges: interior(), note: "odd x even - sliceable" },
  { name: "interior-5x5-odd", width: 5, height: 5, edges: interior(), note: "odd x odd interior - hard" },
  { name: "interior-3x4", width: 3, height: 4, edges: interior(), note: "odd x even - sliceable" },
  { name: "interior-7x1-thin", width: 7, height: 1, edges: interior(), note: "1-wide interior - likely impossible" },

  // --- one side on the paper boundary (edge-crop) ---
  { name: "edge-5x3-short-on-right", width: 5, height: 3, edges: { ...interior(), right: true }, note: "5 wide perpendicular to right edge" },
  { name: "edge-1x3-long-on-right", width: 1, height: 3, edges: { ...interior(), right: true }, note: "1 wide perpendicular to right edge" },
  { name: "edge-5x3-long-on-top", width: 5, height: 3, edges: { ...interior(), top: true }, note: "5 wide parallel to top edge" },
  { name: "edge-8x2-on-left", width: 8, height: 2, edges: { ...interior(), left: true }, note: "even short side - simple" },

  // --- corner (two adjacent sides on the paper boundary) ---
  { name: "corner-5x5", width: 5, height: 5, edges: { ...interior(), left: true, top: true }, note: "5x5 in a paper corner" },
  { name: "corner-5x3", width: 5, height: 3, edges: { ...interior(), right: true, top: true }, note: "5x3 in a paper corner" },
  { name: "corner-3x3", width: 3, height: 3, edges: { ...interior(), left: true, bottom: true }, note: "small odd corner" },
];
