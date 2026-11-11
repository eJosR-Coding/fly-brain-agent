/** Small DOM helpers for the bottom-left controls: labelled groups and the color key. */

/** A labelled role=group (aria-label = visible label); returns the element the chips go into. */
export function addGroup(host: HTMLElement, name: string, label: string): HTMLElement {
  const g = document.createElement("div");
  g.className = "ctl-group";
  g.dataset.group = name;
  g.setAttribute("role", "group");
  g.setAttribute("aria-label", label);
  const l = document.createElement("span");
  l.className = "ctl-label";
  l.setAttribute("aria-hidden", "true"); // the group already carries the name
  l.textContent = label;
  const items = document.createElement("div");
  items.className = "ctl-items";
  g.append(l, items);
  host.appendChild(g);
  return items;
}

/** A tight cluster inside a group (e.g. the camera presets); returns its chip container. */
export function addSubgroup(items: HTMLElement, label: string): HTMLElement {
  const s = document.createElement("div");
  s.className = "ctl-sub";
  s.setAttribute("role", "group");
  s.setAttribute("aria-label", label);
  items.appendChild(s);
  return s;
}

/** Color key: a label row, then one row per category with its real neuron count. */
export function renderKey(el: HTMLElement, title: string, rows: { name: string; count: number; hex: string }[]): void {
  const head = document.createElement("div");
  head.className = "ctl-label";
  head.textContent = title;
  el.replaceChildren(
    head,
    ...rows.map((r) => {
      const row = document.createElement("div");
      row.className = "key-row";
      const dot = document.createElement("i");
      dot.style.background = r.hex;
      const name = document.createElement("span");
      name.textContent = r.name;
      const n = document.createElement("b");
      n.textContent = r.count.toLocaleString("en-US");
      row.append(dot, name, n);
      return row;
    }),
  );
}
