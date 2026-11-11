/** Click-to-inspect glue: hover ring, click-vs-drag, the neuron card (textContent only) and its follow loop. */
import type { BrainScene } from "./scene";
import { LABEL_TIPS, PARTNER_TIPS, cardView, isClick, parseNeuronCard, placeCard, type CardView, type NeuronCard, type Pt } from "./inspect";

export interface InspectorOptions {
  /** where the card lives (the stage: positioned, same box as the canvas) */
  stage: HTMLElement;
  scene: BrainScene;
  pointCount: number | null;
  /** "show partners": light up the neuron and its listed partners */
  onShowPartners?: (rows: number[], label: string) => void;
  fetchFn?: typeof fetch;
}

export interface Inspector {
  open(row: number): Promise<void>;
  close(): void;
  readonly isOpen: boolean;
  dispose(): void;
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, txt?: string) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt !== undefined) e.textContent = txt;
  return e;
};

function partnerList(title: string, items: CardView["inputs"]): HTMLElement {
  const sec = el("section", "nc-partners");
  const h = el("h3", undefined, title);
  if (PARTNER_TIPS[title]) h.title = PARTNER_TIPS[title];
  sec.append(h);
  const ul = el("ul");
  for (const p of items) {
    const li = el("li");
    li.append(el("span", "nc-pname", p.text), el("b", undefined, p.syn));
    ul.append(li);
  }
  if (!items.length) ul.append(el("li", "nc-none", "none at this threshold"));
  sec.append(ul);
  return sec;
}

export function mountInspector(o: InspectorOptions): Inspector {
  const { stage, scene, onShowPartners } = o;
  const fetchFn = o.fetchFn ?? fetch;
  const card = el("section", "neuron-card");
  card.id = "neuron-card";
  card.tabIndex = -1;
  card.setAttribute("role", "dialog");
  card.setAttribute("aria-label", "Neuron details");
  card.hidden = true;
  stage.appendChild(card);

  let current: number | null = null; // row of the open card
  let token = 0; // newest request wins
  let offFrame: (() => void) | null = null; // follows the camera only when the scene draws a frame

  const follow = () => {
    if (current === null) return;
    const p = scene.project(current);
    card.style.visibility = p ? "visible" : "hidden";
    if (p) {
      const pos = placeCard(p, { w: card.offsetWidth, h: card.offsetHeight }, { w: stage.clientWidth, h: stage.clientHeight }, { gap: 22, margin: 12 });
      card.style.transform = `translate(${Math.round(pos.left)}px, ${Math.round(pos.top)}px)`;
    }
  };

  const close = () => {
    token++;
    if (current === null) return;
    const hadFocus = card.contains(document.activeElement);
    current = null;
    card.hidden = true;
    card.replaceChildren();
    scene.setSelected(null);
    offFrame?.();
    offFrame = null;
    if (hadFocus) scene.element.focus?.();
  };

  const render = (c: NeuronCard) => {
    const v = cardView(c);
    const head = el("header");
    head.append(el("h2", undefined, v.title));
    const x = el("button", "nc-close", "×");
    x.type = "button";
    x.setAttribute("aria-label", "Close neuron details");
    x.addEventListener("click", close);
    head.append(x);
    const dl = el("dl");
    for (const r of v.rows) {
      const row = el("div");
      const dt = el("dt", undefined, r.label);
      if (LABEL_TIPS[r.label]) dt.title = LABEL_TIPS[r.label];
      row.append(dt, el("dd", undefined, r.value));
      dl.append(row);
    }
    card.replaceChildren(head);
    if (v.plain) card.append(el("p", "nc-plain", v.plain)); // the sentence for non-specialists, right under the title
    card.append(dl);
    if (v.partnersNote) {
      card.append(el("p", "nc-note", v.partnersNote));
    } else {
      card.append(partnerList("top inputs", v.inputs), partnerList("top outputs", v.outputs));
      const rows = [c.row, ...v.inputs.map((p) => p.row), ...v.outputs.map((p) => p.row)];
      if (rows.length > 1 && onShowPartners) {
        const b = el("button", "chip nc-show", "show partners");
        b.type = "button";
        b.addEventListener("click", () => onShowPartners(rows, `${v.title} · with top partners`));
        card.append(b);
      }
    }
    follow(); // new content, new size: re-place now (the next camera move would do it anyway)
  };

  const message = (text: string) => {
    const head = el("header");
    const x = el("button", "nc-close", "×");
    x.type = "button";
    x.setAttribute("aria-label", "Close neuron details");
    x.addEventListener("click", close);
    head.append(el("h2", undefined, `neuron ${current ?? ""}`), x);
    card.replaceChildren(head, el("p", "nc-note", text));
    follow();
  };

  async function open(row: number): Promise<void> {
    const mine = ++token;
    current = row;
    scene.setSelected(row);
    card.hidden = false;
    message("loading…");
    offFrame ??= scene.onFrame(follow);
    follow();
    scene.requestRender();
    try {
      const res = await fetchFn(`/api/neurons/${row}`);
      const body: unknown = await res.json().catch(() => null);
      if (mine !== token) return;
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const parsed = parseNeuronCard(body, o.pointCount);
      if (typeof parsed === "string") throw new Error(parsed);
      render(parsed);
      card.focus({ preventScroll: true });
    } catch (e) {
      if (mine === token) message(`could not load this neuron: ${(e as Error).message}`);
    }
  }

  // pointer handling on the canvas
  const canvas = scene.element;
  let down: Pt | null = null;
  let hoverFrame = 0;
  let last: Pt | null = null;
  const onDown = (e: PointerEvent) => {
    if (e.button === 0) down = { x: e.clientX, y: e.clientY };
  };
  const onUp = (e: PointerEvent) => {
    const start = down;
    down = null;
    if (e.button !== 0 || !start || !isClick(start, { x: e.clientX, y: e.clientY })) return;
    const row = scene.pick(e.clientX, e.clientY);
    if (row === null) close();
    else void open(row);
  };
  const onMove = (e: PointerEvent) => {
    if (e.buttons) return; // dragging: orbit owns the pointer
    last = { x: e.clientX, y: e.clientY };
    if (hoverFrame) return;
    hoverFrame = requestAnimationFrame(() => {
      hoverFrame = 0;
      if (!last) return;
      const row = scene.pick(last.x, last.y);
      scene.setHover(row);
      canvas.style.cursor = row === null ? "" : "pointer";
    });
  };
  const onLeave = () => {
    last = null;
    scene.setHover(null);
    canvas.style.cursor = "";
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape" && current !== null) close();
  };
  const onOutside = (e: PointerEvent) => {
    if (current === null) return;
    const t = e.target as Node;
    if (!card.contains(t) && !canvas.contains(t)) close();
  };
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  document.addEventListener("keydown", onKey);
  document.addEventListener("pointerdown", onOutside);

  return {
    open,
    close,
    get isOpen() {
      return current !== null;
    },
    dispose() {
      close();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onOutside);
      if (hoverFrame) cancelAnimationFrame(hoverFrame);
      card.remove();
    },
  };
}
