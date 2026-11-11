/** DOM glue for the charts panel: a spike raster and the readout trace, plain SVG. The shaping lives in charts.ts (tested). */
import type { Activity } from "./activity";
import {
  groupLabelLayout, linearScale, parseRowTypes, rasterModel, rasterTooltip, readoutSeries, smooth, smoothWindow, traceModel, tracePath,
  traceTooltip, type RasterModel, type TraceModel,
} from "./charts";

export interface ChartsPanel {
  /** Show a recording (and its baseline, for silenced runs). Fetches the cell types of the raster rows. */
  show(a: Activity, baseline: Activity | null): void;
  clear(): void;
  /** Move the playhead (ms), or hide it with null. */
  setTime(ms: number | null): void;
  /** Open or close the panel (it only draws while a recording is loaded). */
  setOpen(open: boolean): void;
  readonly open: boolean;
}

const NS = "http://www.w3.org/2000/svg";
const W = 420;
const LEFT = 84; // room for the cell-type labels
const RIGHT = 62; // room for the direct line labels
const ROW_PX = 5;
const RASTER_RIGHT = RIGHT; // same plot width as the trace so the time axes and the playhead line up
const TOP_N = 30;

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, text?: string): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  if (text !== undefined) e.textContent = text;
  return e;
}

export function createChartsPanel(root: HTMLElement): ChartsPanel {
  let isOpen = false;
  let current: { a: Activity; baseline: Activity | null } | null = null;
  let token = 0;
  let playheads: { line: SVGLineElement; x: (ms: number) => number }[] = [];

  const rasterBox = document.createElement("div");
  const traceBox = document.createElement("div");
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.hidden = true;
  const title = (text: string) => {
    const h = document.createElement("h2");
    h.textContent = text;
    return h;
  };
  const traceTitle = title("readout · mean rate");
  root.replaceChildren(title("raster · busiest neurons"), rasterBox, traceTitle, traceBox, tip);

  const showTip = (text: string, ev: PointerEvent) => {
    tip.textContent = text;
    tip.hidden = false;
    const box = root.getBoundingClientRect();
    const x = Math.min(ev.clientX - box.left + 12, box.width - tip.offsetWidth - 8);
    tip.style.transform = `translate(${Math.max(8, x)}px, ${ev.clientY - box.top - 34}px)`;
  };
  const hideTip = () => {
    tip.hidden = true;
  };

  const xAxis = (g: SVGElement, x: (ms: number) => number, durationMs: number, y: number) => {
    for (const [i, frac] of [0, 0.5, 1].entries()) {
      const ms = Math.round(frac * durationMs);
      g.appendChild(svg("line", { x1: x(ms), x2: x(ms), y1: y, y2: y + 4, class: "ch-axis" }));
      g.appendChild(svg("text", { x: x(ms), y: y + 16, "text-anchor": i === 0 ? "start" : i === 2 ? "end" : "middle", class: "ch-text" }, i === 2 ? `${ms} ms` : String(ms)));
    }
  };

  const drawRaster = (m: RasterModel, a: Activity) => {
    const plotH = m.rows.length * ROW_PX;
    const H = plotH + 22;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
      "aria-label": `Spike raster of the ${m.rows.length} most active neurons over ${Math.round(m.durationMs)} ms` });
    const x = linearScale(0, m.durationMs, LEFT, W - RASTER_RIGHT);
    const binW = Math.max(1, ((W - RASTER_RIGHT - LEFT) / m.nBins) * 0.55); // thin ticks keep the spike pattern readable
    const frame = svg("g");
    frame.appendChild(svg("line", { x1: LEFT, x2: W - RASTER_RIGHT, y1: plotH + 0.5, y2: plotH + 0.5, class: "ch-axis" }));
    // group brackets and labels
    for (const g of m.groups) {
      const y0 = g.start * ROW_PX + 0.5;
      const y1 = g.end * ROW_PX - 0.5;
      frame.appendChild(svg("line", { x1: LEFT - 6, x2: LEFT - 6, y1: y0, y2: y1, class: g.emphasized ? "ch-axis strong" : "ch-axis" }));
    }
    for (const l of groupLabelLayout(m.groups, ROW_PX, 14)) {
      frame.appendChild(svg("text", { x: LEFT - 12, y: l.y + 4, "text-anchor": "end", class: l.emphasized ? "ch-text strong" : "ch-text" }, l.label));
    }
    const marks = svg("g");
    m.rows.forEach((r, i) => {
      for (const sp of r.spikes) {
        const level = Math.sqrt(sp.count / a.peak);
        marks.appendChild(svg("rect", {
          x: x(sp.bin * m.binMs), y: i * ROW_PX, width: binW, height: ROW_PX - 1,
          fill: "#fff", "fill-opacity": (r.emphasized ? 1 : 0.3 + 0.7 * level).toFixed(2),
        }));
      }
    });
    frame.appendChild(marks);
    xAxis(frame, x, m.durationMs, plotH + 2);
    const head = svg("line", { x1: LEFT, x2: LEFT, y1: 0, y2: plotH, class: "ch-head", visibility: "hidden" });
    const hover = svg("rect", { height: ROW_PX - 1, width: binW + 2, class: "ch-hover", visibility: "hidden" });
    s.append(frame, hover, head);
    playheads.push({ line: head, x });
    const over = svg("rect", { x: LEFT, y: 0, width: W - RASTER_RIGHT - LEFT, height: plotH, fill: "transparent" });
    over.addEventListener("pointermove", (ev) => {
      const box = s.getBoundingClientRect();
      const px = ((ev.clientX - box.left) / box.width) * W;
      const py = ((ev.clientY - box.top) / box.height) * H;
      const bin = Math.min(m.nBins - 1, Math.max(0, Math.floor(((px - LEFT) / (W - RASTER_RIGHT - LEFT)) * m.nBins)));
      const row = Math.min(m.rows.length - 1, Math.max(0, Math.floor(py / ROW_PX)));
      hover.setAttribute("x", String(x(bin * m.binMs) - 1));
      hover.setAttribute("y", String(row * ROW_PX));
      hover.setAttribute("visibility", "visible");
      showTip(rasterTooltip(m.rows[row], bin, m.binMs), ev);
    });
    over.addEventListener("pointerleave", () => {
      hover.setAttribute("visibility", "hidden");
      hideTip();
    });
    s.appendChild(over);
    rasterBox.replaceChildren(s);
  };

  const drawTrace = (m: TraceModel | null, label = "", windowMs = 0) => {
    if (!m) {
      traceTitle.textContent = "readout";
      const p = document.createElement("p");
      p.className = "ch-note";
      p.textContent = "no readout group in this run (give sugar reads MN9)";
      traceBox.replaceChildren(p);
      return;
    }
    traceTitle.textContent = `readout ${label} · Hz · ${windowMs} ms average`;
    const H = 126;
    const top = 6;
    const bottom = H - 24;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
      "aria-label": `Readout rate over time: ${m.series.map((x) => x.label).join(" and ")}, up to ${m.axis.max} Hz` });
    const x = linearScale(0, m.durationMs, LEFT, W - RIGHT);
    const y = linearScale(0, m.axis.max, bottom, top);
    for (const t of m.axis.ticks) {
      s.appendChild(svg("line", { x1: LEFT, x2: W - RIGHT, y1: y(t), y2: y(t), class: "ch-grid" }));
      s.appendChild(svg("text", { x: LEFT - 8, y: y(t) + 4, "text-anchor": "end", class: "ch-text" }, String(t)));
    }
    xAxis(s, x, m.durationMs, bottom + 2);
    for (const ser of m.series) {
      s.appendChild(svg("path", { d: tracePath(ser, x, y), class: ser.dashed ? "ch-line dashed" : "ch-line", fill: "none" }));
    }
    // direct labels at the right end, nudged apart when the lines finish close together
    let lastY = -Infinity;
    for (const ser of [...m.series].sort((p, q) => y(p.points.at(-1)?.hz ?? 0) - y(q.points.at(-1)?.hz ?? 0))) {
      const ly = Math.max(y(ser.points.at(-1)?.hz ?? 0), lastY + 14);
      lastY = ly;
      s.appendChild(svg("text", { x: W - RIGHT + 6, y: ly + 4, class: "ch-text strong" }, ser.label));
    }
    const head = svg("line", { x1: LEFT, x2: LEFT, y1: top, y2: bottom, class: "ch-head", visibility: "hidden" });
    const cross = svg("line", { y1: top, y2: bottom, class: "ch-cross", visibility: "hidden" });
    const dots = m.series.map(() => svg("circle", { r: 4, class: "ch-dot", visibility: "hidden" }));
    s.append(cross, ...dots, head);
    playheads.push({ line: head, x });
    const over = svg("rect", { x: LEFT, y: top, width: W - RIGHT - LEFT, height: bottom - top, fill: "transparent" });
    over.addEventListener("pointermove", (ev) => {
      const box = s.getBoundingClientRect();
      const px = ((ev.clientX - box.left) / box.width) * W;
      const nBins = m.series[0].points.length;
      const bin = Math.min(nBins - 1, Math.max(0, Math.floor(((px - LEFT) / (W - RIGHT - LEFT)) * nBins)));
      const cx = x(m.series[0].points[bin].t);
      cross.setAttribute("x1", String(cx));
      cross.setAttribute("x2", String(cx));
      cross.setAttribute("visibility", "visible");
      m.series.forEach((ser, i) => {
        dots[i].setAttribute("cx", String(cx));
        dots[i].setAttribute("cy", String(y(ser.points[bin].hz)));
        dots[i].setAttribute("visibility", "visible");
        dots[i].classList.toggle("dashed", ser.dashed);
      });
      showTip(traceTooltip(m, bin), ev);
    });
    over.addEventListener("pointerleave", () => {
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      hideTip();
    });
    s.appendChild(over);
    // legend for two or more series: a short stroke per series, solid or dashed (never colour)
    const nodes: Node[] = [s];
    if (m.series.length > 1) {
      const legend = document.createElement("div");
      legend.className = "ch-legend";
      for (const ser of m.series) {
        const item = document.createElement("span");
        const key = svg("svg", { width: 22, height: 8, viewBox: "0 0 22 8", "aria-hidden": "true" });
        key.appendChild(svg("line", { x1: 0, x2: 22, y1: 4, y2: 4, class: ser.dashed ? "ch-line dashed" : "ch-line" }));
        item.append(key, ser.label);
        legend.appendChild(item);
      }
      nodes.unshift(legend);
    }
    // the values as a table for screen readers (the chart is an img)
    const table = document.createElement("table");
    table.className = "sr-only";
    const head2 = table.createTHead().insertRow();
    for (const h of ["t (ms)", ...m.series.map((q) => `${q.label} (Hz)`)]) head2.insertCell().textContent = h;
    const body = table.createTBody();
    m.series[0].points.forEach((p, i) => {
      const r = body.insertRow();
      r.insertCell().textContent = String(Math.round(p.t));
      for (const ser of m.series) r.insertCell().textContent = ser.points[i].hz.toFixed(1);
    });
    nodes.push(table);
    traceBox.replaceChildren(...nodes);
  };

  const render = async () => {
    if (!current || !isOpen) return;
    const { a, baseline } = current;
    const mine = ++token;
    playheads = [];
    let types = new Map<number, string | null>();
    if (a.id) {
      try {
        const res = await fetch(`/api/simulations/${a.id}/rows`);
        if (res.ok) types = parseRowTypes(await res.json());
      } catch {
        /* labels fall back to "unknown" */
      }
    }
    if (mine !== token) return; // a newer recording replaced this one while the labels loaded
    playheads = [];
    drawRaster(rasterModel(a, (r) => types.get(r) ?? null, { top: TOP_N, emphasize: a.readout ? { rows: a.readout.rows, label: a.readout.label } : null }), a);
    if (a.readout) {
      const w = smoothWindow(a.binMs);
      const list = [];
      if (baseline?.readout) list.push(readoutSeries("baseline", smooth(baseline.readout.hz, w), a.binMs, false));
      list.push(readoutSeries(baseline?.readout ? "silenced" : a.readout.label, smooth(a.readout.hz, w), a.binMs, baseline?.readout ? true : false));
      drawTrace(traceModel(list, a.binMs, a.durationMs), a.readout.label, Math.round(w * a.binMs));
    } else {
      drawTrace(null);
    }
  };

  const sync = () => {
    root.hidden = !(isOpen && current);
    document.body.classList.toggle("has-charts", isOpen && current !== null);
    if (!root.hidden) void render();
  };

  return {
    show(a, baseline) {
      current = { a, baseline };
      sync();
    },
    clear() {
      current = null;
      token++;
      playheads = [];
      sync();
    },
    setTime(ms) {
      for (const p of playheads) {
        if (ms === null) p.line.setAttribute("visibility", "hidden");
        else {
          const px = String(p.x(ms));
          p.line.setAttribute("x1", px);
          p.line.setAttribute("x2", px);
          p.line.setAttribute("visibility", "visible");
        }
      }
    },
    setOpen(open) {
      isOpen = open;
      sync();
    },
    get open() {
      return isOpen;
    },
  };
}
