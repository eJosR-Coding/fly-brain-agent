// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { hexToRgb, mountPreview, parsePreview, previewToImageData } from "../src/brainPreview";

function buf(w: number, h: number, cells: number[], extra = 0): ArrayBuffer {
  const out = new Uint8Array(4 + w * h + extra);
  new DataView(out.buffer).setUint16(0, w, true);
  new DataView(out.buffer).setUint16(2, h, true);
  out.set(cells, 4);
  return out.buffer;
}

describe("parsePreview", () => {
  it("reads the little-endian header and the brightness grid", () => {
    const p = parsePreview(buf(3, 2, [0, 10, 20, 30, 40, 255]));
    expect(p.width).toBe(3);
    expect(p.height).toBe(2);
    expect(Array.from(p.data)).toEqual([0, 10, 20, 30, 40, 255]);
  });
  it("rejects a buffer shorter than the header", () => {
    expect(() => parsePreview(new ArrayBuffer(3))).toThrow(/header/);
  });
  it("rejects a length that does not match the header", () => {
    expect(() => parsePreview(buf(3, 2, [], 1))).toThrow(/11 bytes.*3x2/);
    expect(() => parsePreview(buf(3, 2, []).slice(0, 8))).toThrow(/3x2/);
  });
  it("rejects an empty grid", () => {
    expect(() => parsePreview(buf(0, 0, []))).toThrow(/empty|zero/i);
  });
});

describe("hexToRgb", () => {
  it("parses #rrggbb and #rgb, falls back to white", () => {
    expect(hexToRgb("#b4b4b4")).toEqual([180, 180, 180]);
    expect(hexToRgb("#fff")).toEqual([255, 255, 255]);
    expect(hexToRgb("nonsense")).toEqual([255, 255, 255]);
  });
});

describe("previewToImageData", () => {
  it("uses the theme color with alpha = brightness", () => {
    const px = previewToImageData({ width: 2, height: 1, data: new Uint8Array([0, 200]) }, [10, 20, 30]);
    expect(Array.from(px)).toEqual([10, 20, 30, 0, 10, 20, 30, 200]);
  });
});

describe("mountPreview", () => {
  afterEach(() => vi.restoreAllMocks());

  const theme = { highlightHex: "#ffffff" };
  const ok = (b: ArrayBuffer) => (async (url: string) => {
    expect(url).toBe("/api/brain/preview");
    return new Response(b);
  }) as unknown as typeof fetch;

  function stubCtx() {
    const put = vi.fn();
    const ctx = { putImageData: put, createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }) };
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(ctx as unknown as CanvasRenderingContext2D);
    return put;
  }

  it("draws a canvas into the host and removes it", async () => {
    const put = stubCtx();
    const host = document.createElement("div");
    const m = mountPreview(host, { fetchFn: ok(buf(2, 2, [0, 1, 2, 3])), theme });
    await m.ready;
    const canvas = host.querySelector("canvas") as HTMLCanvasElement;
    expect(canvas).not.toBeNull();
    expect([canvas.width, canvas.height]).toEqual([2, 2]);
    expect(put).toHaveBeenCalledTimes(1);
    expect(host.querySelector(".brain-preview")).not.toBeNull();
    m.remove();
    expect(host.querySelector(".brain-preview")).toBeNull();
  });

  it("shows nothing and does not throw on a failed fetch or a bad body", async () => {
    stubCtx();
    const host = document.createElement("div");
    const bad = (async () => new Response("x", { status: 500 })) as unknown as typeof fetch;
    const a = mountPreview(host, { fetchFn: bad, theme });
    await a.ready;
    const b = mountPreview(host, { fetchFn: ok(new ArrayBuffer(2)), theme });
    await b.ready;
    const c = mountPreview(host, { fetchFn: (async () => { throw new Error("offline"); }) as unknown as typeof fetch, theme });
    await c.ready;
    expect(host.querySelector("canvas")).toBeNull();
    a.remove(); b.remove(); c.remove();
  });

  it("does not draw if removed before the fetch resolves", async () => {
    const put = stubCtx();
    const host = document.createElement("div");
    const m = mountPreview(host, { fetchFn: ok(buf(1, 1, [9])), theme });
    m.remove();
    await m.ready;
    expect(host.querySelector("canvas")).toBeNull();
    expect(put).not.toHaveBeenCalled();
  });
});
