import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { frameDistance, loadBody, parseBody, type BodyMeta } from "../src/bodyData";

// Synthetic asset: bounds x [0,100], y [-50,50], z [10,30] (um); two segments in two parts.
const meta: BodyMeta = {
  bounds: [[0, -50, 10], [100, 50, 30]],
  head_center_um: [100, 0, 30],
  parts: [
    { name: "c_head", group: "head", side: "c", segment_start: 0, segment_count: 1 },
    { name: "l_wing", group: "wing", side: "l", segment_start: 1, segment_count: 1 },
  ],
};
function bin(values: number[]): ArrayBuffer {
  const b = new ArrayBuffer(values.length * 2);
  const v = new DataView(b);
  values.forEach((q, i) => v.setUint16(i * 2, q, true));
  return b;
}
// seg0: min corner -> max corner ; seg1: both endpoints at the midpoint
const good = () => bin([0, 0, 0, 65535, 65535, 65535, 32768, 32768, 32768, 32768, 32768, 32768]);

describe("parseBody", () => {
  it("maps the quantized corners to the bounds exactly, in millimetres and Three axes", () => {
    const b = parseBody(good(), meta);
    expect(b.segmentCount).toBe(2);
    expect(b.positions.length).toBe(2 * 2 * 3);
    // three = (y_fly, z_fly, x_fly) in mm
    expect(Array.from(b.positions.slice(0, 3))).toEqual([-0.05, 0.01, 0].map((v) => expect.closeTo(v, 6)));
    expect(Array.from(b.positions.slice(3, 6))).toEqual([0.05, 0.03, 0.1].map((v) => expect.closeTo(v, 6)));
  });
  it("decodes the midpoint within one quantization step", () => {
    const b = parseBody(good(), meta);
    const step = 100 / 65535 / 1000;
    expect(Math.abs(b.positions[6] - 0)).toBeLessThanOrEqual(0.1 / 65535 + 1e-6); // y: -50..50 mid ~0
    expect(Math.abs(b.positions[8] - 0.05)).toBeLessThanOrEqual(step + 1e-6); // x_fly -> Z
  });
  it("puts fly +x on Three +Z, +y on +X, +z on +Y, keeping right-handedness", () => {
    const b = parseBody(bin([0, 0, 0, 65535, 0, 0, 0, 0, 0, 0, 65535, 0]), meta);
    const o = b.positions;
    expect(o[5] - o[2]).toBeGreaterThan(0); // +x_fly -> +Z
    expect(o[9] - o[6]).toBeGreaterThan(0); // +y_fly -> +X
    expect(o[9] - o[6]).toBeGreaterThan(0);
    const ex = [1, 0, 0]; // fly x -> three (0,0,1); y -> (1,0,0); z -> (0,1,0)
    const tx = [0, 0, 1], ty = [1, 0, 0], tz = [0, 1, 0];
    const cross = [tx[1] * ty[2] - tx[2] * ty[1], tx[2] * ty[0] - tx[0] * ty[2], tx[0] * ty[1] - tx[1] * ty[0]];
    expect(cross).toEqual(tz); // x cross y = z is preserved
    expect(ex).toBeDefined();
  });
  it("maps +z_fly to Three +Y", () => {
    const b = parseBody(bin([0, 0, 0, 0, 0, 65535, 0, 0, 0, 0, 0, 0]), meta);
    expect(b.positions[4] - b.positions[1]).toBeGreaterThan(0);
  });
  it("rejects a wrong byte length with a clear error", () => {
    expect(() => parseBody(new ArrayBuffer(10), meta)).toThrow(/10 bytes.*24/);
  });
  it("rejects parts that do not cover the segments", () => {
    const bad = { ...meta, parts: [meta.parts[0]] };
    expect(() => parseBody(good(), bad)).toThrow(/parts/);
  });
  it("keeps contiguous parts that cover every segment", () => {
    const b = parseBody(good(), meta);
    expect(b.parts).toEqual([
      { name: "c_head", group: "head", side: "c", start: 0, count: 1 },
      { name: "l_wing", group: "wing", side: "l", start: 1, count: 1 },
    ]);
  });
  it("maps the head centre and reports bounds and centre in Three axes (mm)", () => {
    const b = parseBody(good(), meta);
    expect(b.headCenter).toEqual([0, 0.03, 0.1].map((v) => expect.closeTo(v, 9)));
    expect(b.bounds.min).toEqual([-0.05, 0.01, 0].map((v) => expect.closeTo(v, 9)));
    expect(b.bounds.max).toEqual([0.05, 0.03, 0.1].map((v) => expect.closeTo(v, 9)));
    expect(b.center).toEqual([0, 0.02, 0.05].map((v) => expect.closeTo(v, 9)));
  });
});

describe("frameDistance", () => {
  const box = (s: number) => ({ min: [-s, -s, -s] as [number, number, number], max: [s, s, s] as [number, number, number] });
  it("grows with the body size", () => {
    expect(frameDistance(box(2), 40, 1.5)).toBeGreaterThan(frameDistance(box(1), 40, 1.5));
  });
  it("grows when the viewport gets narrower and when the fov gets smaller", () => {
    expect(frameDistance(box(1), 40, 0.6)).toBeGreaterThan(frameDistance(box(1), 40, 1.6));
    expect(frameDistance(box(1), 20, 1.5)).toBeGreaterThan(frameDistance(box(1), 60, 1.5));
  });
  it("leaves roughly 10% margin: the bounding sphere fits the vertical view", () => {
    const b = box(1);
    const d = frameDistance(b, 40, 10);
    const r = Math.sqrt(3);
    expect(d * Math.tan((20 * Math.PI) / 180)).toBeGreaterThanOrEqual(r);
  });
});

describe("loadBody", () => {
  it("fetches the static json and bin (not under /api) and parses them", async () => {
    const urls: string[] = [];
    const f = (async (u: string) => {
      urls.push(u);
      return u.endsWith(".json")
        ? new Response(JSON.stringify(meta))
        : new Response(good());
    }) as unknown as typeof fetch;
    const b = await loadBody(f);
    expect(urls.sort()).toEqual(["/fly-body.bin", "/fly-body.json"]);
    expect(b.segmentCount).toBe(2);
  });
  it("fails clearly on an HTTP error", async () => {
    const f = (async () => new Response("no", { status: 404 })) as unknown as typeof fetch;
    await expect(loadBody(f)).rejects.toThrow(/404/);
  });
});

describe("the committed asset", () => {
  const m = JSON.parse(readFileSync("public/fly-body.json", "utf8")) as BodyMeta;
  const raw = readFileSync("public/fly-body.bin");
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  it("parses to 32262 segments inside the json bounds, no NaN", () => {
    const b = parseBody(buf, m);
    expect(b.segmentCount).toBe(32262);
    const eps = 1e-9;
    let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < b.positions.length; i++) {
      const v = b.positions[i];
      expect(Number.isFinite(v)).toBe(true);
      const a = i % 3;
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
    for (let a = 0; a < 3; a++) {
      expect(min[a]).toBeGreaterThanOrEqual(b.bounds.min[a] - eps);
      expect(max[a]).toBeLessThanOrEqual(b.bounds.max[a] + eps);
    }
    // json bounds (um) -> mm, Three axes
    expect(b.bounds.min[0]).toBeCloseTo(m.bounds[0][1] / 1000, 6);
    expect(b.bounds.max[2]).toBeCloseTo(m.bounds[1][0] / 1000, 6);
  });
});
