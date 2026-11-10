import { describe, expect, it } from "vitest";
import { homeCamera, labelSpecs, placeLabels, screenBounds, type Box, type Landmarks } from "../src/anatomy";

const LM: Landmarks = {
  landmarks: [
    { name: "central brain", x: 500, y: 190, z: 140, count: 10 },
    { name: "optic lobe L", x: 280, y: 270, z: 200, count: 10 },
    { name: "lamina L", x: 200, y: 290, z: 170, count: 10 },
    { name: "optic lobe R", x: 760, y: 280, z: 165, count: 10 },
    { name: "lamina R", x: 830, y: 290, z: 126, count: 10 },
  ],
  sides: {
    left: { lobe_center: [280, 270, 200], radius: 100, lamina_direction: [-1, 0, 0] },
    right: { lobe_center: [760, 280, 165], radius: 100, lamina_direction: [0.8, 0.0, -0.6] },
  },
};
const norm = (a: number[]) => Math.hypot(a[0], a[1], a[2]);

describe("labelSpecs", () => {
  const specs = labelSpecs(LM);
  it("one label per landmark, with friendly text", () => {
    expect(specs.map((s) => s.text)).toEqual([
      "central brain", "optic lobe", "lamina · eye input", "optic lobe", "lamina · eye input",
    ]);
  });
  it("the central brain is labeled above the cloud, the rest to the sides", () => {
    expect(specs.map((s) => s.kind)).toEqual(["top", "side", "side", "side", "side"]);
  });
  it("positions come straight from the landmarks", () => {
    expect(specs[1].position).toEqual([280, 270, 200]);
  });
  it("missing groups just produce fewer labels", () => {
    expect(labelSpecs({ landmarks: [LM.landmarks[0]], sides: {} }).map((s) => s.text)).toEqual(["central brain"]);
  });
});

describe("homeCamera", () => {
  it("front view: straight on the -z side (where the eyes face), level", () => {
    const [x, y, z] = homeCamera(100, 0, 0);
    expect(x).toBeCloseTo(0);
    expect(y).toBeCloseTo(0);
    expect(z).toBeCloseTo(-100);
  });
  it("keeps the distance while turning and rising", () => {
    const p = homeCamera(140, 28, 14);
    expect(norm(p)).toBeCloseTo(140, 6);
    expect(p[1]).toBeGreaterThan(0);
    expect(p[0]).not.toBeCloseTo(0);
  });
});

describe("screenBounds", () => {
  it("is the min/max of the flat xy list", () => {
    expect(screenBounds([10, 20, 30, 5, 22, 80])).toEqual({ minX: 10, maxX: 30, minY: 5, maxY: 80 });
  });
  it("is null without points", () => {
    expect(screenBounds([])).toBeNull();
  });
});

describe("placeLabels", () => {
  const box: Box = { minX: 300, maxX: 700, minY: 200, maxY: 500 };
  const opts = { margin: 24, spacing: 18, width: 1000, height: 800 };
  const cx = 500;
  const outside = (x: number, y: number) => x < box.minX - opts.margin + 1e-9 || x > box.maxX + opts.margin - 1e-9 || y < box.minY - opts.margin + 1e-9 || y > box.maxY + opts.margin - 1e-9;

  it("left-screen anchors go left and right-screen anchors go right, whatever the data side says", () => {
    const [a, b] = placeLabels([{ ax: 400, ay: 350, kind: "side" }, { ax: 600, ay: 350, kind: "side" }], box, cx, opts);
    expect(a.side).toBe("left");
    expect(b.side).toBe("right");
  });
  it("every text end sits outside the cloud's projected bounds plus the margin", () => {
    const items = [
      { ax: 400, ay: 350, kind: "side" as const }, { ax: 350, ay: 360, kind: "side" as const },
      { ax: 650, ay: 300, kind: "side" as const }, { ax: 500, ay: 250, kind: "top" as const },
    ];
    for (const [i, p] of placeLabels(items, box, cx, opts).entries()) {
      expect(outside(items[i].ax + p.dx, items[i].ay + p.dy)).toBe(true);
    }
  });
  it("top labels rise above the cloud and keep the anchor's x", () => {
    const [p] = placeLabels([{ ax: 480, ay: 260, kind: "top" }], box, cx, opts);
    expect(p.side).toBe("top");
    expect(p.dx).toBeCloseTo(0);
    expect(260 + p.dy).toBeLessThanOrEqual(box.minY - opts.margin);
  });
  it("labels on the same side never stack closer than the spacing", () => {
    const items = [{ ax: 400, ay: 350, kind: "side" as const }, { ax: 380, ay: 352, kind: "side" as const }, { ax: 360, ay: 351, kind: "side" as const }];
    const ys = placeLabels(items, box, cx, opts).map((p, i) => items[i].ay + p.dy).sort((x, y) => x - y);
    expect(ys[1] - ys[0]).toBeGreaterThanOrEqual(opts.spacing - 1e-9);
    expect(ys[2] - ys[1]).toBeGreaterThanOrEqual(opts.spacing - 1e-9);
  });
  it("an anchor already outside the margin still leaves the leader pointing outward", () => {
    const [p] = placeLabels([{ ax: 200, ay: 350, kind: "side" }], box, cx, opts);
    expect(p.side).toBe("left");
    expect(p.dx).toBeLessThan(0);
  });
  it("keeps labels inside the viewport vertically", () => {
    const [p] = placeLabels([{ ax: 400, ay: 5, kind: "side" }], box, cx, opts);
    expect(5 + p.dy).toBeGreaterThanOrEqual(opts.spacing / 2);
  });
  it("leader length and angle follow the offset", () => {
    const [p] = placeLabels([{ ax: 600, ay: 350, kind: "side" }], box, cx, opts);
    expect(p.length).toBeCloseTo(Math.hypot(p.dx, p.dy));
    expect(p.angle).toBeCloseTo((Math.atan2(p.dy, p.dx) * 180) / Math.PI);
  });
});
