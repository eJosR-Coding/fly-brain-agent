/** DOM glue for the charts panel: a spike raster and the readout trace, plain SVG. The shaping lives in charts.ts (tested). */
import type { Activity } from "./activity";
import { doseCaption, doseModel, doseTooltip, screenCaption, screenModel, type DoseChart, type ExperimentChart, type ScreenChart } from "./experiments";
import {
  groupLabelLayout, linearScale, parseRowTypes, rasterCaption, rasterModel, traceCaption, rasterTooltip, readoutSeries, smooth, smoothWindow, traceModel, tracePath,
  traceTooltip, type RasterModel, type TraceModel,
} from "./charts";

export interface ChartsPanel {
  /** Show a recording (and its baseline, for silenced runs). Fetches the cell types of the raster rows. */
  show(a: Activity, baseline: Activity | null): void;
  /** Show an experiment chart (dose-response, screen); `onPick` receives the simulation id of a clicked screen row. */
  showExperiment(c: ExperimentChart, onPick: (simulationId: string) => void): void;
  /** Whether an experiment chart is held (the recording view offers a way back to it). */
  readonly hasExperiment: boolean;
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
  let experiment: { chart: ExperimentChart; onPick: (id: string) => void } | null = null;
  let view: "recording" | "experiment" = "recording";

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
  const rasterTitle = title("raster · busiest neurons");
  const expTitle = title("experiment");
  const expBox = document.createElement("div");
  const back = document.createElement("button");
  back.type = "button";
  back.className = "ch-back";
  back.hidden = true;
  back.addEventListener("click", () => {
    view = "experiment";
    sync();
  });
  root.replaceChildren(back, expTitle, expBox, rasterTitle, rasterBox, traceTitle, traceBox, tip);

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
    rasterBox.replaceChildren(plain(rasterCaption(m.rows.length, m.durationMs)), s);
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
    nodes.unshift(plain(traceCaption(label, m)));
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

  const plain = (text: string) => {
    const p = document.createElement("p");
    p.className = "ch-plain";
    p.textContent = text;
    return p;
  };
  const note = (text: string) => {
    const p = document.createElement("p");
    p.className = "ch-note";
    p.textContent = text;
    return p;
  };
  const table = (head: string[], rows: string[][]) => {
    const t = document.createElement("table");
    t.className = "sr-only";
    const h = t.createTHead().insertRow();
    for (const c of head) h.insertCell().textContent = c;
    const body = t.createTBody();
    for (const r of rows) {
      const tr = body.insertRow();
      for (const c of r) tr.insertCell().textContent = c;
    }
    return t;
  };

  const drawDose = (c: DoseChart) => {
    const H = 196;
    const box = { left: 50, right: W - 70, top: 12, bottom: H - 40 };
    const m = doseModel(c, box);
    expTitle.textContent = `dose-response · ${c.readoutLabel} Hz vs ${c.stimulusLabel} rate`;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
      "aria-label": `Dose-response: ${c.readoutLabel} mean rate at ${c.points.map((p) => `${p.rate} Hz`).join(", ")} stimulation` });
    for (const t of m.yAxis.ticks) {
      s.appendChild(svg("line", { x1: box.left, x2: box.right, y1: m.y(t), y2: m.y(t), class: "ch-grid" }));
      s.appendChild(svg("text", { x: box.left - 8, y: m.y(t) + 4, "text-anchor": "end", class: "ch-text" }, String(t)));
    }
    let lastX = -Infinity;
    for (const t of m.xTicks) {
      if (t.x - lastX < 30) continue; // drop a label that would touch the previous one
      lastX = t.x;
      s.appendChild(svg("line", { x1: t.x, x2: t.x, y1: box.bottom, y2: box.bottom + 4, class: "ch-axis" }));
      s.appendChild(svg("text", { x: t.x, y: box.bottom + 17, "text-anchor": "middle", class: "ch-text" }, String(t.rate)));
    }
    s.appendChild(svg("line", { x1: box.left, x2: box.right, y1: box.bottom, y2: box.bottom, class: "ch-axis" }));
    s.appendChild(svg("text", { x: (box.left + box.right) / 2, y: H - 4, "text-anchor": "middle", class: "ch-text" }, "stimulus rate (Hz, log scale)"));
    if (m.band) s.appendChild(svg("path", { d: m.band, class: "ch-band" }));
    if (m.ec50) {
      s.appendChild(svg("line", { x1: m.ec50.x, x2: m.ec50.x, y1: box.top, y2: box.bottom, class: "ch-ec50" }));
      const right = m.ec50.x > box.right - 80;
      // low on the plot, where a rising curve is not: the label never sits on a point or a whisker
      s.appendChild(svg("text", { x: m.ec50.x + (right ? -5 : 5), y: box.bottom - 6, "text-anchor": right ? "end" : "start", class: "ch-text strong" }, m.ec50.label));
    }
    s.appendChild(svg("path", { d: m.points.map((p, i) => `${i ? "L" : "M"}${p.x},${p.y}`).join(" "), class: "ch-line", fill: "none" }));
    for (const p of m.points) {
      if (p.low !== null && p.high !== null) {
        s.appendChild(svg("line", { x1: p.x, x2: p.x, y1: p.yLow, y2: p.yHigh, class: "ch-whisker" }));
        for (const yy of [p.yLow, p.yHigh]) s.appendChild(svg("line", { x1: p.x - 3, x2: p.x + 3, y1: yy, y2: yy, class: "ch-whisker" }));
      }
      s.appendChild(svg("circle", { cx: p.x, cy: p.y, r: 4, class: "ch-dot" }));
    }
    const ring = svg("circle", { r: 7, class: "ch-hover", visibility: "hidden" });
    s.appendChild(ring);
    const over = svg("rect", { x: box.left - 8, y: box.top, width: box.right - box.left + 16, height: box.bottom - box.top, fill: "transparent" });
    over.addEventListener("pointermove", (ev) => {
      const r = s.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const i = m.points.reduce((best, p, k) => (Math.abs(p.x - px) < Math.abs(m.points[best].x - px) ? k : best), 0);
      ring.setAttribute("cx", String(m.points[i].x));
      ring.setAttribute("cy", String(m.points[i].y));
      ring.setAttribute("visibility", "visible");
      showTip(doseTooltip(c.points[i], c.readoutLabel), ev);
    });
    over.addEventListener("pointerleave", () => {
      ring.setAttribute("visibility", "hidden");
      hideTip();
    });
    s.appendChild(over);
    expBox.replaceChildren(
      plain(doseCaption(c)), s,
      note(`${c.trials} trial${c.trials === 1 ? "" : "s"} per rate · mean with 95% CI · model prediction, not a measurement`),
      table(["rate (Hz)", `${c.readoutLabel} mean (Hz)`, "CI low", "CI high"], c.points.map((p) => [String(p.rate), String(p.mean), String(p.low ?? ""), String(p.high ?? "")])),
    );
  };

  const drawScreen = (c: ScreenChart, onPick: (id: string) => void) => {
    expTitle.textContent = `silencing screen · effect on ${c.readoutLabel} (Hz)`;
    if (!c.rows.length) {
      expBox.replaceChildren(note("no candidate neuron fired in the baseline"));
      return;
    }
    const ROW = 18;
    const m = screenModel(c, { left: 92, right: W - 24, top: 6, rowPx: ROW });
    const s = svg("svg", { viewBox: `0 0 ${W} ${m.height}`, width: W, height: m.height, role: "group",
      "aria-label": `Silencing screen: change in ${c.readoutLabel} rate when each of ${c.rows.length} neurons is silenced` });
    for (const t of m.xTicks) {
      s.appendChild(svg("line", { x1: t.x, x2: t.x, y1: 6, y2: m.axisY, class: "ch-grid" }));
      s.appendChild(svg("text", { x: t.x, y: m.axisY + 16, "text-anchor": "middle", class: "ch-text" }, (t.value > 0 ? "+" : "") + t.value));
    }
    s.appendChild(svg("line", { x1: m.zeroX, x2: m.zeroX, y1: 6, y2: m.axisY, class: "ch-axis strong" }));
    s.appendChild(svg("text", { x: 92 + (W - 24 - 92) / 2, y: m.height - 4, "text-anchor": "middle", class: "ch-text" }, "Δ readout Hz (silenced − baseline)"));
    for (const r of m.rows) {
      const g = svg("g", { class: "ch-row", tabindex: 0, role: "button", "aria-label": r.tooltip });
      g.appendChild(svg("rect", { x: 0, y: r.y - ROW / 2, width: W, height: ROW, class: "ch-rowhit" }));
      g.appendChild(svg("text", { x: 84, y: r.y + 4, "text-anchor": "end", class: "ch-text strong" }, r.label.length > 11 ? `${r.label.slice(0, 10)}…` : r.label));
      g.appendChild(svg("line", { x1: r.xLow, x2: r.xHigh, y1: r.y, y2: r.y, class: "ch-whisker" }));
      for (const xx of [r.xLow, r.xHigh]) g.appendChild(svg("line", { x1: xx, x2: xx, y1: r.y - 3, y2: r.y + 3, class: "ch-whisker" }));
      g.appendChild(svg("circle", { cx: r.x, cy: r.y, r: 5, class: r.filled ? "ch-mark filled" : "ch-mark hollow" }));
      g.addEventListener("pointermove", (ev) => showTip(r.tooltip, ev));
      g.addEventListener("pointerleave", hideTip);
      g.addEventListener("click", () => onPick(r.simulationId));
      g.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          onPick(r.simulationId);
        }
      });
      s.appendChild(g);
    }
    const legend = document.createElement("div");
    legend.className = "ch-legend";
    for (const [cls, text] of [["filled", "significant (Holm-corrected)"], ["hollow", "not significant"]] as const) {
      const item = document.createElement("span");
      const key = svg("svg", { width: 12, height: 12, viewBox: "0 0 12 12", "aria-hidden": "true" });
      key.appendChild(svg("circle", { cx: 6, cy: 6, r: 4.5, class: `ch-mark ${cls}` }));
      item.append(key, text);
      legend.appendChild(item);
    }
    expBox.replaceChildren(
      plain(screenCaption(c)), legend, s,
      note(`${c.trials} trial${c.trials === 1 ? "" : "s"} per condition · whiskers are 95% CI · click a row to open its difference map · model prediction`),
      table(["neuron", "row", "delta (Hz)", "CI low", "CI high", "p", "significant"],
        c.rows.map((r) => [r.label, String(r.row), String(r.delta), String(r.low ?? ""), String(r.high ?? ""), String(r.p ?? ""), r.significant ? "yes" : "no"])),
    );
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
    const has = view === "experiment" ? experiment !== null : current !== null;
    root.hidden = !(isOpen && has);
    document.body.classList.toggle("has-charts", isOpen && has);
    const showExp = view === "experiment" && experiment !== null;
    for (const el of [expTitle, expBox]) el.hidden = !showExp;
    for (const el of [rasterTitle, rasterBox, traceTitle, traceBox]) el.hidden = showExp;
    back.hidden = showExp || experiment === null;
    back.textContent = experiment?.chart.kind === "screen" ? "← back to the screen chart" : "← back to the dose-response chart";
    if (root.hidden) return;
    if (showExp && experiment) {
      playheads = [];
      if (experiment.chart.kind === "screen") drawScreen(experiment.chart, experiment.onPick);
      else drawDose(experiment.chart);
    } else void render();
  };

  return {
    show(a, baseline) {
      current = { a, baseline };
      view = "recording";
      sync();
    },
    showExperiment(c, onPick) {
      experiment = { chart: c, onPick };
      view = "experiment";
      sync();
    },
    get hasExperiment() {
      return experiment !== null;
    },
    clear() {
      current = null;
      experiment = null;
      view = "recording";
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
