import { describe, expect, it } from "vitest";
import { buildHops, hopLabel, hopWidth, parsePathEvent, pathLegend, pointAlong, routeVertices } from "../src/path";

const ok = { selection_id: "sel_a1", rows: [2, 0, 1, 3], synapses: [19, 5, 40], bottleneck: 5 };

describe("parsePathEvent", () => {
  it("accepts a well-formed event", () => {
    expect(parsePathEvent(ok, 10)).toEqual({ selectionId: "sel_a1", rows: [2, 0, 1, 3], synapses: [19, 5, 40], bottleneck: 5 });
  });
  it("needs one synapse count per hop", () => {
    expect(parsePathEvent({ ...ok, synapses: [19, 5] }, 10)).toMatch(/synapses/);
    expect(parsePathEvent({ ...ok, synapses: [19, 5, 40, 1] }, 10)).toMatch(/synapses/);
  });
  it("needs at least two rows, all integers inside the point count", () => {
    expect(parsePathEvent({ ...ok, rows: [1], synapses: [] }, 10)).toMatch(/rows/);
    expect(parsePathEvent({ ...ok, rows: [2, 0, 1, 10] }, 10)).toMatch(/row/);
    expect(parsePathEvent({ ...ok, rows: [2, 0, -1, 3] }, 10)).toMatch(/row/);
    expect(parsePathEvent({ ...ok, rows: [2, 0, 1.5, 3] }, 10)).toMatch(/row/);
    expect(typeof parsePathEvent({ ...ok, rows: [2, 0, 1, 10] }, null)).toBe("object"); // count not known yet
  });
  it("rejects non-numeric or negative synapse counts and a bad bottleneck", () => {
    expect(parsePathEvent({ ...ok, synapses: [19, "x", 40] }, 10)).toMatch(/synapses/);
    expect(parsePathEvent({ ...ok, synapses: [19, -5, 40] }, 10)).toMatch(/synapses/);
    expect(parsePathEvent({ ...ok, bottleneck: "5" }, 10)).toMatch(/bottleneck/);
  });
  it("rejects non-objects", () => {
    expect(parsePathEvent(null, 10)).toMatch(/object/);
    expect(parsePathEvent([], 10)).toMatch(/object/);
  });
});

describe("hopWidth", () => {
  it("grows with the log of the synapses and is clamped", () => {
    expect(hopWidth(5)).toBeLessThan(hopWidth(50));
    expect(hopWidth(50)).toBeLessThan(hopWidth(500));
    expect(hopWidth(0)).toBe(hopWidth(1));
    expect(hopWidth(1e9)).toBe(hopWidth(1e12));
    expect(hopWidth(1)).toBeGreaterThanOrEqual(1);
    expect(hopWidth(1)).toBeLessThanOrEqual(1.2); // fine threads, not ropes
    expect(hopWidth(1e12)).toBeLessThanOrEqual(2.5);
    expect(hopWidth(19)).toBeGreaterThan(hopWidth(5) + 0.2); // visibly thicker, not a rounding difference
  });
});

describe("labels", () => {
  it("formats a hop and the legend", () => {
    expect(hopLabel(19)).toBe("19 syn");
    expect(hopLabel(1234)).toBe("1,234 syn");
    expect(pathLegend(parsePathEvent(ok, 10) as never)).toBe("path · 3 hops · bottleneck 5 syn");
    expect(pathLegend({ selectionId: "", rows: [1, 2], synapses: [7], bottleneck: 7 })).toBe("path · 1 hop · bottleneck 7 syn");
  });
});

describe("buildHops / routeVertices", () => {
  // 4 points on the x axis: row r sits at x = 10 * r
  const positions = Float32Array.from([0, 0, 0, 10, 0, 0, 20, 0, 0, 30, 0, 0]);
  const p = { selectionId: "", rows: [2, 0, 1, 3], synapses: [19, 5, 40], bottleneck: 5 };
  it("makes one hop per consecutive pair, with endpoints, width, label and midpoint", () => {
    const hops = buildHops(p, positions);
    expect(hops).toHaveLength(3);
    expect(hops[0].from).toEqual([20, 0, 0]);
    expect(hops[0].to).toEqual([0, 0, 0]);
    expect(hops[0].mid).toEqual([10, 0, 0]);
    expect(hops[0].label).toBe("19 syn");
    expect(hops[0].width).toBe(hopWidth(19));
    expect(hops[2].width).toBeGreaterThan(hops[1].width);
  });
  it("lists the route's vertices in order", () => {
    expect(routeVertices(p, positions)).toEqual([[20, 0, 0], [0, 0, 0], [10, 0, 0], [30, 0, 0]]);
  });
});

describe("pointAlong", () => {
  const v: [number, number, number][] = [[0, 0, 0], [10, 0, 0], [10, 20, 0]]; // 10 then 20 long
  it("walks the polyline by fraction of its total length", () => {
    expect(pointAlong(v, 0)).toEqual([0, 0, 0]);
    expect(pointAlong(v, 1)).toEqual([10, 20, 0]);
    const a = pointAlong(v, 0.5); // 15 along: 5 up the second leg
    expect(a[0]).toBeCloseTo(10);
    expect(a[1]).toBeCloseTo(5);
  });
  it("clamps and wraps nothing", () => {
    expect(pointAlong(v, -3)).toEqual([0, 0, 0]);
    expect(pointAlong(v, 7)).toEqual([10, 20, 0]);
  });
  it("copes with a degenerate polyline", () => {
    expect(pointAlong([[1, 2, 3]], 0.5)).toEqual([1, 2, 3]);
    expect(pointAlong([[1, 2, 3], [1, 2, 3]], 0.5)).toEqual([1, 2, 3]);
  });
});
