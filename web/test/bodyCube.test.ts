import { describe, expect, it } from "vitest";
import { CUBE_GRID_CELLS, SANDBOX_INTERIOR_MM, containsBox, cubeBounds, cubeEdges, cubeFaces, cubeFitDistance, spanFitDistance } from "../src/bodyCube";

describe("glass cube geometry", () => {
  it("is a few dozen mm wide: big next to a ~3 mm fly, small enough to see it whole", () => {
    expect(SANDBOX_INTERIOR_MM).toBeGreaterThanOrEqual(40);
    expect(SANDBOX_INTERIOR_MM).toBeLessThanOrEqual(60);
  });
  it("centres the floor on the start and stands on the floor height", () => {
    const b = cubeBounds(-0.3, 50);
    expect(b.min).toEqual([-25, -0.3, -25]);
    expect(b.max).toEqual([25, 49.7, 25]);
  });
  it("defaults to the sandbox size", () => {
    expect(cubeBounds(0).max[0]).toBe(SANDBOX_INTERIOR_MM / 2);
    expect(cubeBounds(0).max[1]).toBe(SANDBOX_INTERIOR_MM);
  });
  it("has 12 edges, each as long as the side and axis-aligned", () => {
    const e = cubeEdges(cubeBounds(0, 40));
    expect(e.length).toBe(12 * 2 * 3);
    for (let i = 0; i < 12; i++) {
      const a = [e[i * 6], e[i * 6 + 1], e[i * 6 + 2]], b = [e[i * 6 + 3], e[i * 6 + 4], e[i * 6 + 5]];
      const d = a.map((v, k) => Math.abs(v - b[k]));
      expect(d.filter((v) => v > 1e-9)).toEqual([40]);
    }
  });
  it("every edge endpoint is a corner of the cube", () => {
    const b = cubeBounds(1, 20);
    const e = cubeEdges(b);
    for (let i = 0; i < e.length; i += 3) {
      for (let k = 0; k < 3; k++) expect([b.min[k], b.max[k]]).toContain(e[i + k]);
    }
  });
  it("has six quad faces as indexed triangles, all vertices on the cube's surface", () => {
    const b = cubeBounds(0, 30);
    const { positions, indices } = cubeFaces(b);
    expect(positions.length).toBe(24 * 3);
    expect(indices.length).toBe(6 * 2 * 3);
    expect(Math.max(...indices)).toBe(23);
    for (let i = 0; i < positions.length; i += 3) {
      const onSurface = [0, 1, 2].some((k) => positions[i + k] === b.min[k] || positions[i + k] === b.max[k]);
      expect(onSurface).toBe(true);
    }
  });
  it("faces cover the surface: triangle areas add up to 6 side^2", () => {
    const { positions: p, indices: ix } = cubeFaces(cubeBounds(0, 10));
    let area = 0;
    for (let t = 0; t < ix.length; t += 3) {
      const v = (i: number) => [p[ix[i] * 3], p[ix[i] * 3 + 1], p[ix[i] * 3 + 2]];
      const [a, b, c] = [v(t), v(t + 1), v(t + 2)];
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      area += 0.5 * Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]);
    }
    expect(area).toBeCloseTo(600, 6);
  });
  it("contains a box only when it lies fully inside", () => {
    const b = cubeBounds(0, 50);
    expect(containsBox(b, [0, 1, 0], 2)).toBe(true);
    expect(containsBox(b, [24.5, 1, 0], 2)).toBe(false); // would reach x = 25.5
    expect(containsBox(b, [24, 1, 0], 2)).toBe(true); // touching the wall is inside;
    expect(containsBox(b, [0, 0.5, 0], 2)).toBe(false); // under the floor
    expect(containsBox(b, [0, 49.5, 0], 2)).toBe(false); // through the ceiling
  });
  it("floor grid divides the interior into whole cells", () => {
    expect(SANDBOX_INTERIOR_MM / CUBE_GRID_CELLS).toBe(Math.round(SANDBOX_INTERIOR_MM / CUBE_GRID_CELLS));
  });
});

describe("camera distances", () => {
  it("fits a span in the limiting dimension", () => {
    const d = spanFitDistance(20, 40, 2); // vertical limits at aspect 2
    expect(d).toBeCloseTo(10 / Math.tan((20 * Math.PI) / 180), 6);
    const narrow = spanFitDistance(20, 40, 0.5); // horizontal limits
    expect(narrow).toBeGreaterThan(d);
  });
  it("fits the whole cube at a distance that grows with the cube and shrinks with a wider view", () => {
    const small = cubeFitDistance(cubeBounds(0, 20), 40, 1.5);
    const big = cubeFitDistance(cubeBounds(0, 50), 40, 1.5);
    expect(big).toBeGreaterThan(small * 2);
    expect(cubeFitDistance(cubeBounds(0, 50), 60, 1.5)).toBeLessThan(big);
  });
  it("puts the whole cube's bounding sphere inside the view", () => {
    const b = cubeBounds(0, 50);
    const r = Math.hypot(50, 50, 50) / 2;
    expect(cubeFitDistance(b, 40, 1.6)).toBeGreaterThanOrEqual(spanFitDistance(2 * r, 40, 1.6) - 1e-9);
  });
});
