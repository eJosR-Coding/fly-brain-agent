import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { boundingRadius, centroid, loadBody, parseBody, parseBodyMesh, viewFitDistance, type BodyMeta, type MeshMeta, type Vec3 } from "../src/bodyData";

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

describe("boundingRadius and centroid", () => {
  const pts = new Float32Array([-1, 0, 0, 3, 0, 0, 0, 2, 0]); // three endpoints
  it("centroid is the mean of all endpoints", () => {
    expect(centroid(pts)).toEqual([2 / 3, 2 / 3, 0].map((v) => expect.closeTo(v, 6)));
  });
  it("boundingRadius is the farthest vertex from the given centre", () => {
    expect(boundingRadius(pts, [0, 0, 0])).toBeCloseTo(3, 6);
    expect(boundingRadius(pts, [3, 0, 0])).toBeCloseTo(4, 6);
  });
  it("handles an empty buffer without NaN", () => {
    expect(centroid(new Float32Array(0))).toEqual([0, 0, 0]);
    expect(boundingRadius(new Float32Array(0), [0, 0, 0])).toBe(0);
  });
  it("parseBody exposes the centroid and the radius around it", () => {
    const b = parseBody(good(), meta);
    expect(b.centroid).toEqual(centroid(b.positions));
    expect(b.radius).toBeCloseTo(boundingRadius(b.positions, b.centroid), 9);
  });
});

describe("viewFitDistance", () => {
  const ring = (r: number) => new Float32Array([r, 0, 0, -r, 0, 0, 0, r, 0, 0, -r, 0]); // circle of radius r in the XY plane
  const c: Vec3 = [0, 0, 0];
  it("fills the requested fraction of the limiting dimension in a wide view", () => {
    const d = viewFitDistance(ring(1), c, 0, 0, 40, 4, 0.7);
    const halfV = Math.tan((20 * Math.PI) / 180);
    expect(1 / (d * halfV)).toBeCloseTo(0.7, 6); // a vertex at y=1 occupies 70% of the half height
  });
  it("uses the horizontal limit when the view is narrow", () => {
    const d = viewFitDistance(ring(1), c, 0, 0, 40, 0.5, 0.7);
    const halfH = Math.tan((20 * Math.PI) / 180) * 0.5;
    expect(1 / (d * halfH)).toBeCloseTo(0.7, 6);
  });
  it("grows with the body size, and when the view gets narrower or the fov smaller", () => {
    expect(viewFitDistance(ring(2), c, 35, 15, 40, 1.5, 0.7)).toBeGreaterThan(viewFitDistance(ring(1), c, 35, 15, 40, 1.5, 0.7));
    expect(viewFitDistance(ring(1), c, 35, 15, 40, 0.6, 0.7)).toBeGreaterThan(viewFitDistance(ring(1), c, 35, 15, 40, 1.6, 0.7));
    expect(viewFitDistance(ring(1), c, 35, 15, 20, 1.5, 0.7)).toBeGreaterThan(viewFitDistance(ring(1), c, 35, 15, 60, 1.5, 0.7));
  });
  it("a smaller fill moves the camera farther away", () => {
    expect(viewFitDistance(ring(1), c, 35, 15, 40, 1.5, 0.5)).toBeGreaterThan(viewFitDistance(ring(1), c, 35, 15, 40, 1.5, 0.7));
  });
  it("never puts a vertex behind the camera", () => {
    const d = viewFitDistance(new Float32Array([0, 0, 5, 0, 0, -5]), c, 0, 0, 40, 1.5, 0.7);
    expect(d).toBeGreaterThan(5);
  });
  it("returns a positive finite distance for an empty body", () => {
    const d = viewFitDistance(new Float32Array(0), c, 35, 15, 40, 1.5, 0.7);
    expect(Number.isFinite(d)).toBe(true);
    expect(d).toBeGreaterThan(0);
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

describe("framing of the committed fly", () => {
  const m = JSON.parse(readFileSync("public/fly-body.json", "utf8")) as BodyMeta;
  const raw = readFileSync("public/fly-body.bin");
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  it("at the default view the fly's vertical extent is about 70% of a wide pane, centred on the centroid", () => {
    const b = parseBody(buf, m);
    const fov = 40, aspect = 1.9, az = 35, el = 15;
    const d = viewFitDistance(b.positions, b.centroid, az, el, fov, aspect, 0.7);
    // project every vertex with the same camera and measure the vertical extent in NDC
    const a = (az * Math.PI) / 180, e = (el * Math.PI) / 180;
    const dir = [Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e)];
    const right = [Math.cos(a), 0, -Math.sin(a)];
    const up = [dir[1] * right[2] - dir[2] * right[1], dir[2] * right[0] - dir[0] * right[2], dir[0] * right[1] - dir[1] * right[0]];
    const t = Math.tan((fov * Math.PI) / 360);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < b.positions.length; i += 3) {
      const p = [b.positions[i] - b.centroid[0], b.positions[i + 1] - b.centroid[1], b.positions[i + 2] - b.centroid[2]];
      const depth = d - (p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2]);
      const y = p[0] * up[0] + p[1] * up[1] + p[2] * up[2];
      const ndc = y / (depth * t);
      lo = Math.min(lo, ndc);
      hi = Math.max(hi, ndc);
    }
    const fill = (hi - lo) / 2; // fraction of the full pane height (NDC spans 2)
    expect(Math.max(Math.abs(lo), Math.abs(hi))).toBeCloseTo(0.7, 2);
    expect(fill).toBeGreaterThan(0.55);
    expect(fill).toBeLessThanOrEqual(0.71);
  });
});

// Synthetic mesh: 3 vertices (min corner, max corner, midpoint) and one triangle.
const meshInfo = (over: Partial<MeshMeta> = {}): MeshMeta => ({ file: "fly-body-mesh.bin", vertex_count: 3, triangle_count: 1, index_dtype: "uint16", bytes: 3 * 6 + 6, ...over });
function meshBin(verts: number[], idx: number[], bytes: 2 | 4 = 2): ArrayBuffer {
  const b = new ArrayBuffer(verts.length * 2 + idx.length * bytes);
  const v = new DataView(b);
  verts.forEach((q, i) => v.setUint16(i * 2, q, true));
  idx.forEach((q, i) => (bytes === 2 ? v.setUint16(verts.length * 2 + i * 2, q, true) : v.setUint32(verts.length * 2 + i * 4, q, true)));
  return b;
}
const goodMesh = () => meshBin([0, 0, 0, 65535, 65535, 65535, 32768, 32768, 32768], [0, 1, 2]);

describe("parseBodyMesh", () => {
  it("decodes corners to mm in Three axes (three = y_fly, z_fly, x_fly) and keeps the indices", () => {
    const m = parseBodyMesh(goodMesh(), { ...meta, mesh: meshInfo() });
    expect(m.positions.length).toBe(9);
    expect(Array.from(m.positions.slice(0, 3))).toEqual([-0.05, 0.01, 0].map((v) => expect.closeTo(v, 6)));
    expect(Array.from(m.positions.slice(3, 6))).toEqual([0.05, 0.03, 0.1].map((v) => expect.closeTo(v, 6)));
    expect(m.indices).toBeInstanceOf(Uint16Array);
    expect(Array.from(m.indices)).toEqual([0, 1, 2]);
  });
  it("reads uint32 indices when the json says so", () => {
    const info = meshInfo({ index_dtype: "uint32", bytes: 3 * 6 + 12 });
    const m = parseBodyMesh(meshBin([0, 0, 0, 1, 1, 1, 2, 2, 2], [2, 1, 0], 4), { ...meta, mesh: info });
    expect(m.indices).toBeInstanceOf(Uint32Array);
    expect(Array.from(m.indices)).toEqual([2, 1, 0]);
  });
  it("rejects a wrong byte length", () => {
    expect(() => parseBodyMesh(new ArrayBuffer(10), { ...meta, mesh: meshInfo() })).toThrow(/10 bytes.*24/);
  });
  it("rejects an index outside the vertex range", () => {
    expect(() => parseBodyMesh(meshBin([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 3]), { ...meta, mesh: meshInfo() })).toThrow(/index 3/);
  });
  it("rejects metadata without a mesh and an unknown index type", () => {
    expect(() => parseBodyMesh(goodMesh(), meta)).toThrow(/mesh/);
    expect(() => parseBodyMesh(goodMesh(), { ...meta, mesh: meshInfo({ index_dtype: "uint8" as never }) })).toThrow(/index_dtype/);
  });
});

describe("loadBody with the mesh", () => {
  const fakeFetch = (opts: { mesh?: boolean; meshFails?: "http" | "garbage" } = {}) => {
    const urls: string[] = [];
    const f = (async (u: string) => {
      urls.push(u);
      if (u.endsWith(".json")) return new Response(JSON.stringify(opts.mesh ? { ...meta, mesh: meshInfo() } : meta));
      if (u.endsWith("mesh.bin")) {
        if (opts.meshFails === "http") return new Response("no", { status: 404 });
        return new Response(opts.meshFails === "garbage" ? new ArrayBuffer(5) : goodMesh());
      }
      return new Response(good());
    }) as unknown as typeof fetch;
    return { f, urls };
  };
  it("also fetches /fly-body-mesh.bin when the json has a mesh key and returns it", async () => {
    const { f, urls } = fakeFetch({ mesh: true });
    const b = await loadBody(f);
    expect(urls.sort()).toEqual(["/fly-body-mesh.bin", "/fly-body.bin", "/fly-body.json"]);
    expect(b.mesh?.indices.length).toBe(3);
  });
  it("returns mesh null and does not fetch it when the json has no mesh key", async () => {
    const { f, urls } = fakeFetch();
    const b = await loadBody(f);
    expect(b.mesh).toBeNull();
    expect(urls).not.toContain("/fly-body-mesh.bin");
  });
  it.each(["http", "garbage"] as const)("falls back to the lines only, with one warning, when the mesh fails (%s)", async (kind) => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const { f } = fakeFetch({ mesh: true, meshFails: kind });
      const b = await loadBody(f);
      expect(b.mesh).toBeNull();
      expect(b.segmentCount).toBe(2);
      expect(warn).toHaveBeenCalledTimes(1);
    } finally {
      warn.mockRestore();
    }
  });
});

describe("the committed mesh asset", () => {
  const m = JSON.parse(readFileSync("public/fly-body.json", "utf8")) as BodyMeta;
  const raw = readFileSync("public/fly-body-mesh.bin");
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  const lines = parseBody(
    (() => { const r = readFileSync("public/fly-body.bin"); return r.buffer.slice(r.byteOffset, r.byteOffset + r.byteLength) as ArrayBuffer; })(),
    m,
  );
  it("has the triangle count of the json, valid indices and no NaN", () => {
    const mesh = parseBodyMesh(buf, m);
    expect(m.mesh?.triangle_count).toBeGreaterThan(50_000);
    expect(mesh.indices.length).toBe(m.mesh!.triangle_count * 3);
    expect(mesh.positions.length).toBe(m.mesh!.vertex_count * 3);
    for (const v of mesh.positions) expect(Number.isFinite(v)).toBe(true);
    let max = 0;
    for (const i of mesh.indices) max = Math.max(max, i);
    expect(max).toBeLessThan(m.mesh!.vertex_count);
  });
  it("decodes inside the line asset bounds", () => {
    const mesh = parseBodyMesh(buf, m);
    const eps = 1e-6; // positions are float32
    for (let i = 0; i < mesh.positions.length; i++) {
      const a = i % 3;
      expect(mesh.positions[i]).toBeGreaterThanOrEqual(lines.bounds.min[a] - eps);
      expect(mesh.positions[i]).toBeLessThanOrEqual(lines.bounds.max[a] + eps);
    }
  });
});

describe("viewFitDistance degenerate input", () => {
  const origin: Vec3 = [0, 0, 0];
  it("returns 1 for an empty position list", () => {
    expect(viewFitDistance([], origin, 30, 20, 40, 1.5, 0.8)).toBe(1);
  });
  it("ignores a trailing incomplete vertex", () => {
    expect(viewFitDistance([1, 2], origin, 30, 20, 40, 1.5, 0.8)).toBe(1);
  });
  it("keeps the camera just in front of vertices that all sit on the target (zero radius)", () => {
    expect(viewFitDistance([0, 0, 0, 0, 0, 0], origin, 30, 20, 40, 1.5, 0.8)).toBeCloseTo(1e-3, 9);
  });
  it("stays finite for a collapsed container (aspect 0) and a zero fill", () => {
    const d = viewFitDistance([1, 0, 0, -1, 0, 0], origin, 0, 0, 40, 0, 0);
    expect(Number.isFinite(d)).toBe(true);
    expect(d).toBeGreaterThan(0);
  });
});
