/** Brain preview: a tiny static picture from GET /brain/preview, drawn while the real scene loads.
 *  Wire format: [width u16 LE, height u16 LE] + width*height uint8 brightness, top row first, 0 = empty. */

export interface Preview {
  width: number;
  height: number;
  data: Uint8Array;
}

export function parsePreview(buffer: ArrayBuffer): Preview {
  if (buffer.byteLength < 4) throw new Error(`brain preview is ${buffer.byteLength} bytes, shorter than its 4-byte header`);
  const view = new DataView(buffer);
  const width = view.getUint16(0, true);
  const height = view.getUint16(2, true);
  if (width === 0 || height === 0) throw new Error("brain preview has an empty (zero-size) grid");
  if (buffer.byteLength !== 4 + width * height) {
    throw new Error(`brain preview is ${buffer.byteLength} bytes, its header says ${width}x${height} (needs ${4 + width * height})`);
  }
  return { width, height, data: new Uint8Array(buffer, 4) };
}

export function hexToRgb(hex: string): [number, number, number] {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return [255, 255, 255];
  const h = m[1].length === 3 ? m[1].replace(/./g, (c) => c + c) : m[1];
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** RGBA bytes: the theme color everywhere, alpha = brightness (so it sits on any background). */
export function previewToImageData(p: Preview, rgb: [number, number, number]): Uint8ClampedArray {
  const out = new Uint8ClampedArray(p.width * p.height * 4);
  for (let i = 0; i < p.data.length; i++) {
    out[i * 4] = rgb[0];
    out[i * 4 + 1] = rgb[1];
    out[i * 4 + 2] = rgb[2];
    out[i * 4 + 3] = p.data[i];
  }
  return out;
}

export interface MountedPreview {
  /** resolves when the fetch/draw attempt is over (never rejects) */
  ready: Promise<void>;
  remove(): void;
}

export function mountPreview(
  host: HTMLElement,
  opts: { fetchFn?: typeof fetch; theme: { highlightHex: string }; base?: string },
): MountedPreview {
  const fetchFn = opts.fetchFn ?? fetch;
  let wrap: HTMLElement | null = null;
  let removed = false;
  const ready = (async () => {
    try {
      const res = await fetchFn(`${opts.base ?? "/api"}/brain/preview`);
      if (!res.ok) return;
      const preview = parsePreview(await res.arrayBuffer());
      if (removed) return;
      const canvas = document.createElement("canvas");
      canvas.width = preview.width;
      canvas.height = preview.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const img = ctx.createImageData(preview.width, preview.height);
      img.data.set(previewToImageData(preview, hexToRgb(opts.theme.highlightHex)));
      ctx.putImageData(img, 0, 0);
      wrap = document.createElement("div");
      wrap.className = "brain-preview";
      wrap.setAttribute("aria-hidden", "true");
      wrap.appendChild(canvas);
      host.appendChild(wrap);
      const calm = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
      if (calm) wrap.classList.add("on");
      else requestAnimationFrame(() => wrap?.classList.add("on"));
    } catch {
      // a placeholder must never block or break boot
    }
  })();
  return {
    ready,
    remove() {
      removed = true;
      wrap?.remove();
      wrap = null;
    },
  };
}
