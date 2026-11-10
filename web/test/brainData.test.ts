import { describe, expect, it } from "vitest";
import { computeBounds, decodeIndices, loadBrain } from "../src/brainData";

const meta = { count: 2, bytes: 24, dtype: "float32", endianness: "little", layout: "xyz-interleaved", order: "atlas rows", extent_um: [10, 20, 30] };

function fakeFetch(points: ArrayBuffer, m = meta) {
  return (async (url: string) => {
    if (url === "/api/brain/meta") return new Response(JSON.stringify(m));
    if (url === "/api/brain/points") return new Response(points);
    return new Response("nope", { status: 404 });
  }) as unknown as typeof fetch;
}

describe("loadBrain", () => {
  it("returns Float32Array positions, meta and the bounding-box center", async () => {
    const pts = new Float32Array([0, 0, 0, 10, 20, 30]);
    const brain = await loadBrain(fakeFetch(pts.buffer));
    expect(Array.from(brain.positions)).toEqual([0, 0, 0, 10, 20, 30]);
    expect(brain.meta.count).toBe(2);
    expect(brain.center).toEqual([5, 10, 15]);
    expect(brain.size).toEqual([10, 20, 30]);
  });

  it("rejects when the byte length does not match meta.bytes", async () => {
    const short = new Float32Array([0, 0, 0]);
    await expect(loadBrain(fakeFetch(short.buffer))).rejects.toThrow(/12 bytes.*24/);
  });

  it("rejects when meta.count does not match the byte length", async () => {
    const pts = new Float32Array(6); // 24 bytes = 2 points, but meta claims 3
    await expect(loadBrain(fakeFetch(pts.buffer, { ...meta, count: 3, bytes: 24 }))).rejects.toThrow(/3 points.*36/);
  });

  it("rejects a byte length that is not a multiple of 12 with a clear message", async () => {
    const m = { ...meta, count: 2, bytes: 10 };
    await expect(loadBrain(fakeFetch(new ArrayBuffer(10), m))).rejects.toThrow(/multiple of 12/);
  });

  it("rejects on an http error", async () => {
    const f = (async () => new Response("x", { status: 500 })) as unknown as typeof fetch;
    await expect(loadBrain(f)).rejects.toThrow(/500/);
  });
});

describe("helpers", () => {
  it("computeBounds scans min/max per axis", () => {
    const b = computeBounds(new Float32Array([1, 5, 9, -1, 7, 3]));
    expect(b.min).toEqual([-1, 5, 3]);
    expect(b.max).toEqual([1, 7, 9]);
  });

  it("decodeIndices reads little-endian uint32", () => {
    const buf = new Uint8Array([1, 0, 0, 0, 0, 1, 0, 0]).buffer;
    expect(Array.from(decodeIndices(buf))).toEqual([1, 256]);
  });

  it("decodeIndices rejects a buffer that is not a multiple of 4", () => {
    expect(() => decodeIndices(new ArrayBuffer(5))).toThrow(/multiple of 4/);
  });
});
