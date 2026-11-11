/**
 * A small Markdown renderer for chat answers with a strict safety model.
 * Model output is untrusted: the parser yields a plain AST and the renderer builds DOM nodes with
 * createElement + text nodes only. Nothing is ever parsed as HTML, links are never anchors, images are
 * never rendered, and styling comes from fixed class names, never from input. The single exception is
 * math, whose KaTeX output is sanitized in ./math (see there).
 */
import { renderMath } from "./math";

export type Inline =
  | { t: "text"; v: string }
  | { t: "b"; c: Inline[] }
  | { t: "i"; c: Inline[] }
  | { t: "del"; c: Inline[] }
  | { t: "code"; v: string }
  | { t: "math"; v: string }
  | { t: "br" };

export interface ListItem {
  c: Inline[];
  checked?: boolean;
  sub?: ListNode;
}
export interface ListNode {
  t: "ul" | "ol";
  items: ListItem[];
}
export type Align = "left" | "center" | "right" | null;

export type Block =
  | { t: "p"; c: Inline[] }
  | { t: "h"; level: 3 | 4; c: Inline[] }
  | { t: "code"; lang: string; v: string }
  | { t: "quote"; c: Block[] }
  | { t: "hr" }
  | { t: "math"; v: string }
  | { t: "table"; align: Align[]; head: Inline[][]; rows: Inline[][][]; rowHeaders: boolean }
  | ListNode;

export const MAX_MARKDOWN_CHARS = 20_000;
export const MAX_TABLE_ROWS = 200;
export const MAX_TABLE_COLS = 20;
export const MAX_MATH_CHARS = 2_000;
export const MAX_MATH_NODES = 150;
const MAX_LIST_DEPTH = 3;
const MAX_QUOTE_DEPTH = 4;

const LIST_RE = /^(\s*)([*-]|\d+\.)[ \t]+(.*)$/;
const HEADING_RE = /^(#{1,6})[ \t]+(.*)$/;
const TASK_RE = /^\[([ xX])\](?:[ \t]+(.*))?$/;
const LANG_RE = /^[A-Za-z0-9_+#.-]{1,20}$/;
const DELIM_CELL_RE = /^:?-+:?$/;

const isWs = (ch: string | undefined) => ch !== undefined && /\s/.test(ch);
const isDigit = (ch: string | undefined) => ch !== undefined && ch >= "0" && ch <= "9";

/**
 * Inline math close search starting after the opener. Heuristic (pandoc-like, plus one rule):
 * the opening `$` is not followed by whitespace, the closing `$` is not preceded by whitespace and not
 * followed by a digit, and the FIRST `$` after the opener decides (so "$5 and $10 and $x$" is text + math).
 * Bounded to MAX_MATH_CHARS so hostile input cannot make it quadratic.
 */
function findDollarClose(s: string, from: number): number {
  const end = Math.min(s.length, from + MAX_MATH_CHARS + 1);
  for (let k = from; k < end; k++) {
    const ch = s[k];
    if (ch === "\\") { k++; continue; }
    if (ch === "$") return k > from && !isWs(s[k - 1]) && !isDigit(s[k + 1]) ? k : -1;
  }
  return -1;
}

/** Closer search within a bounded window (used for `$$` and `\(`). */
function findBounded(s: string, d: string, from: number): number {
  const k = s.slice(from, from + MAX_MATH_CHARS + d.length).indexOf(d);
  return k < 0 ? -1 : from + k;
}

/** Inline spans: `code`, **bold**, *italic*, ~~strike~~, math. Anything unclosed stays literal. */
function parseInline(s: string): Inline[] {
  const out: Inline[] = [];
  let buf = "";
  const flush = () => {
    if (buf) out.push({ t: "text", v: buf });
    buf = "";
  };
  // a delimiter that is missing after position p is missing after every later position too: no rescans
  const misses = new Map<string, number>();
  const find = (d: string, from: number): number => {
    const m = misses.get(d);
    if (m !== undefined && from >= m) return -1;
    const j = s.indexOf(d, from);
    if (j < 0) misses.set(d, from);
    return j;
  };
  const pushMath = (v: string): boolean => {
    const tex = v.trim();
    if (!tex || tex.length > MAX_MATH_CHARS) return false;
    flush();
    out.push({ t: "math", v: tex });
    return true;
  };
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === "`") {
      const j = find("`", i + 1);
      if (j > i + 1) {
        flush();
        out.push({ t: "code", v: s.slice(i + 1, j) });
        i = j + 1;
        continue;
      }
    } else if (ch === "\\" && s[i + 1] === "$") {
      buf += "$";
      i += 2;
      continue;
    } else if (ch === "\\" && s[i + 1] === "(") {
      const j = findBounded(s, "\\)", i + 2);
      if (j > i + 2 && pushMath(s.slice(i + 2, j))) {
        i = j + 2;
        continue;
      }
    } else if (ch === "$" && s[i + 1] === "$") {
      const j = findBounded(s, "$$", i + 2);
      if (j > i + 2 && pushMath(s.slice(i + 2, j))) {
        i = j + 2;
        continue;
      }
      buf += "$$";
      i += 2;
      continue;
    } else if (ch === "$" && s[i + 1] !== undefined && !isWs(s[i + 1])) {
      const j = findDollarClose(s, i + 1);
      if (j > i + 1 && pushMath(s.slice(i + 1, j))) {
        i = j + 1;
        continue;
      }
    } else if (ch === "~" && s[i + 1] === "~") {
      const j = find("~~", i + 2);
      if (j > i + 2) {
        flush();
        out.push({ t: "del", c: parseInline(s.slice(i + 2, j)) });
        i = j + 2;
        continue;
      }
      buf += "~~";
      i += 2;
      continue;
    } else if (ch === "*" && s[i + 1] === "*") {
      const j = find("**", i + 2);
      if (j > i + 2) {
        flush();
        out.push({ t: "b", c: parseInline(s.slice(i + 2, j)) });
        i = j + 2;
        continue;
      }
      buf += "**";
      i += 2;
      continue;
    } else if (ch === "*" && s[i + 1] !== undefined && !isWs(s[i + 1])) {
      const j = find("*", i + 1);
      if (j > i + 1 && !isWs(s[j - 1])) {
        flush();
        out.push({ t: "i", c: parseInline(s.slice(i + 1, j)) });
        i = j + 1;
        continue;
      }
    }
    buf += ch;
    i++;
  }
  flush();
  return out;
}

/** Plain text lines (used for constructs that never closed, e.g. a streaming partial fence). */
function literal(lines: string[]): Block {
  const c: Inline[] = [];
  lines.forEach((l, k) => {
    if (k) c.push({ t: "br" });
    if (l) c.push({ t: "text", v: l });
  });
  return { t: "p", c };
}

function indentOf(line: string): number {
  let n = 0;
  for (const ch of line) {
    if (ch === " ") n++;
    else if (ch === "\t") n += 4;
    else break;
  }
  return n;
}

function isHr(line: string): boolean {
  if (indentOf(line) > 3) return false;
  const t = line.trim();
  if (t.length < 3 || (t[0] !== "-" && t[0] !== "*" && t[0] !== "_")) return false;
  let n = 0;
  for (const ch of t) {
    if (ch === t[0]) n++;
    else if (ch !== " " && ch !== "\t") return false;
  }
  return n >= 3;
}

function fenceOf(line: string): { ch: string; len: number; lang: string } | null {
  if (indentOf(line) > 3) return null;
  const t = line.trim();
  const ch = t[0];
  if (ch !== "`" && ch !== "~") return null;
  let len = 0;
  while (t[len] === ch) len++;
  if (len < 3) return null;
  const info = t.slice(len).trim();
  if (ch === "`" && info.includes("`")) return null;
  return { ch, len, lang: LANG_RE.test(info) ? info : "" };
}

function isFenceClose(line: string, ch: string, len: number): boolean {
  const t = line.trim();
  if (t.length < len) return false;
  for (const c of t) if (c !== ch) return false;
  return true;
}

const isQuoteLine = (line: string) => indentOf(line) <= 3 && line.trimStart().startsWith(">");
const stripQuote = (line: string) => {
  const t = line.trimStart().slice(1);
  return t.startsWith(" ") ? t.slice(1) : t;
};

function splitRow(line: string): string[] {
  let t = line.trim();
  if (t.startsWith("|")) t = t.slice(1);
  if (t.endsWith("|") && !t.endsWith("\\|")) t = t.slice(0, -1);
  const cells: string[] = [];
  let buf = "";
  for (let k = 0; k < t.length; k++) {
    if (t[k] === "\\" && t[k + 1] === "|") {
      buf += "|";
      k++;
    } else if (t[k] === "|") {
      cells.push(buf.trim());
      buf = "";
    } else buf += t[k];
  }
  cells.push(buf.trim());
  return cells;
}

function alignOf(cell: string): Align {
  const l = cell.startsWith(":");
  const r = cell.endsWith(":");
  return l && r ? "center" : r ? "right" : l ? "left" : null;
}

function startsBlock(line: string): boolean {
  return fenceOf(line) !== null || isHr(line) || HEADING_RE.test(line.trim()) || isQuoteLine(line);
}

function tableAt(lines: string[], i: number): { block: Block; next: number } | null {
  const head = lines[i];
  const delim = lines[i + 1];
  if (delim === undefined || !head.includes("|") || !delim.includes("-")) return null;
  const hc = splitRow(head);
  const dc = splitRow(delim);
  if (hc.length !== dc.length || !dc.every((c) => DELIM_CELL_RE.test(c))) return null;
  const cols = Math.min(hc.length, MAX_TABLE_COLS);
  const rows: Inline[][][] = [];
  let j = i + 2;
  for (; j < lines.length && lines[j].trim() && !startsBlock(lines[j]); j++) {
    if (rows.length >= MAX_TABLE_ROWS) continue; // swallow the excess rows, do not render them
    const cells = splitRow(lines[j]);
    rows.push(Array.from({ length: cols }, (_, k) => (cells[k] ? parseInline(cells[k]) : [])));
  }
  return {
    block: {
      t: "table",
      align: dc.slice(0, cols).map(alignOf),
      head: hc.slice(0, cols).map((c) => (c ? parseInline(c) : [])),
      rows,
      rowHeaders: hc[0] === "",
    },
    next: j,
  };
}

/** Display math starting at line i: `$$ x $$`, `$$\n x \n$$`, `\[ x \]`, `\[\n x \n\]`. */
function mathAt(lines: string[], i: number): { block: Block; next: number } | null {
  const t = lines[i].trim();
  const [open, close] = t.startsWith("$$") ? ["$$", "$$"] : t.startsWith("\\[") ? ["\\[", "\\]"] : [];
  if (!open || !close || indentOf(lines[i]) > 3) return null;
  const parts: string[] = [];
  let rest = t.slice(2);
  let j = i;
  for (;;) {
    if (rest.trimEnd().endsWith(close) && (j > i || rest.trimEnd().length >= close.length)) {
      parts.push(rest.trimEnd().slice(0, -close.length));
      const v = parts.join("\n").trim();
      if (!v) return null;
      return { block: v.length > MAX_MATH_CHARS ? { t: "code", lang: "", v } : { t: "math", v }, next: j + 1 };
    }
    parts.push(rest);
    j++;
    if (j >= lines.length) return { block: literal(lines.slice(i)), next: lines.length }; // unclosed: plain text
    rest = lines[j];
  }
}

function parseBlocks(lines: string[], depth: number): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let levels: { indent: number; list: ListNode }[] = [];

  const endPara = () => {
    if (!para.length) return;
    const c: Inline[] = [];
    para.forEach((line, k) => {
      if (k) c.push({ t: "br" });
      c.push(...parseInline(line));
    });
    blocks.push({ t: "p", c });
    para = [];
  };
  const endList = () => {
    levels = [];
  };
  const addItem = (indent: number, kind: "ul" | "ol", text: string) => {
    const task = TASK_RE.exec(text);
    const item: ListItem = task
      ? { c: task[2] ? parseInline(task[2]) : [], checked: task[1] !== " " }
      : { c: parseInline(text) };
    if (!levels.length) {
      const list: ListNode = { t: kind, items: [] };
      blocks.push(list);
      levels = [{ indent, list }];
    } else {
      let top = levels[levels.length - 1];
      if (indent >= top.indent + 2 && top.list.items.length) {
        if (levels.length < MAX_LIST_DEPTH) {
          const parent = top.list.items[top.list.items.length - 1];
          parent.sub = { t: kind, items: [] };
          top = { indent, list: parent.sub };
          levels.push(top);
        } // at the depth cap the item is flattened into the current list
      } else {
        while (levels.length > 1 && indent < top.indent) {
          levels.pop();
          top = levels[levels.length - 1];
        }
        if (levels.length === 1 && top.list.t !== kind) {
          const list: ListNode = { t: kind, items: [] };
          blocks.push(list);
          levels = [{ indent, list }];
          top = levels[0];
        }
      }
      top.list.items.push(item);
      return;
    }
    levels[0].list.items.push(item);
  };

  for (let i = 0; i < lines.length; ) {
    const raw = lines[i];
    if (!raw.trim()) {
      endPara();
      endList();
      i++;
      continue;
    }
    const fence = fenceOf(raw);
    if (fence) {
      endPara();
      endList();
      let j = i + 1;
      while (j < lines.length && !isFenceClose(lines[j], fence.ch, fence.len)) j++;
      if (j >= lines.length) {
        blocks.push(literal(lines.slice(i))); // unclosed (e.g. still streaming): plain text
        break;
      }
      blocks.push({ t: "code", lang: fence.lang, v: lines.slice(i + 1, j).join("\n") });
      i = j + 1;
      continue;
    }
    const math = mathAt(lines, i);
    if (math) {
      endPara();
      endList();
      blocks.push(math.block);
      i = math.next;
      continue;
    }
    if (isHr(raw)) {
      endPara();
      endList();
      blocks.push({ t: "hr" });
      i++;
      continue;
    }
    if (depth < MAX_QUOTE_DEPTH && isQuoteLine(raw)) {
      endPara();
      endList();
      const inner: string[] = [];
      for (; i < lines.length && isQuoteLine(lines[i]); i++) inner.push(stripQuote(lines[i]));
      blocks.push({ t: "quote", c: parseBlocks(inner, depth + 1) });
      continue;
    }
    const table = tableAt(lines, i);
    if (table) {
      endPara();
      endList();
      blocks.push(table.block);
      i = table.next;
      continue;
    }
    const li = LIST_RE.exec(raw);
    if (li && (levels.length || indentOf(raw) < 4)) {
      endPara();
      addItem(indentOf(li[1]), /\d/.test(li[2]) ? "ol" : "ul", li[3]);
      i++;
      continue;
    }
    if (!para.length && indentOf(raw) >= 4) {
      endList();
      let j = i;
      while (j < lines.length && (!lines[j].trim() || indentOf(lines[j]) >= 4)) j++;
      const chunk = lines.slice(i, j);
      while (chunk.length && !chunk[chunk.length - 1].trim()) chunk.pop();
      blocks.push({ t: "code", lang: "", v: chunk.map((l) => (l.startsWith("\t") ? l.slice(1) : l.slice(4))).join("\n") });
      i = j;
      continue;
    }
    endList();
    const h = HEADING_RE.exec(raw);
    if (h) {
      endPara();
      blocks.push({ t: "h", level: h[1].length >= 4 ? 4 : 3, c: parseInline(h[2]) });
      i++;
      continue;
    }
    para.push(raw.trim());
    i++;
  }
  endPara();
  return blocks;
}

export function parseMarkdown(input: string): Block[] {
  const src = input.length > MAX_MARKDOWN_CHARS ? input.slice(0, MAX_MARKDOWN_CHARS) : input;
  return parseBlocks(src.split(/\r?\n/), 0);
}

type Doc = Pick<Document, "createElement" | "createTextNode">;
interface Ctx {
  doc: Doc;
  math: number; // formulas rendered so far in this document (bounded)
}

const ALIGN_CLASS = { left: "al-left", center: "al-center", right: "al-right" } as const;

function codeFallback(text: string, ctx: Ctx): HTMLElement {
  const el = ctx.doc.createElement("code");
  el.append(ctx.doc.createTextNode(text));
  return el;
}

function mathNode(tex: string, display: boolean, ctx: Ctx): Node {
  const wrap = ctx.doc.createElement(display ? "div" : "span");
  const frag = ctx.math < MAX_MATH_NODES ? renderMath(tex, display) : null;
  ctx.math++;
  if (!frag) {
    wrap.className = "md-math-src";
    wrap.append(ctx.doc.createTextNode(display ? `$$${tex}$$` : `$${tex}$`));
    return wrap;
  }
  wrap.className = display ? "md-math md-math-display" : "md-math";
  wrap.append(frag);
  return wrap;
}

function renderInline(nodes: Inline[], ctx: Ctx): Node[] {
  const { doc } = ctx;
  return nodes.map((n) => {
    switch (n.t) {
      case "text": return doc.createTextNode(n.v);
      case "br": return doc.createElement("br");
      case "code": return codeFallback(n.v, ctx);
      case "math": return mathNode(n.v, false, ctx);
      default: {
        const el = doc.createElement(n.t === "b" ? "strong" : n.t === "i" ? "em" : "del");
        el.append(...renderInline(n.c, ctx));
        return el;
      }
    }
  });
}

function renderList(l: ListNode, ctx: Ctx): HTMLElement {
  const { doc } = ctx;
  const el = doc.createElement(l.t);
  for (const it of l.items) {
    const li = doc.createElement("li");
    if (it.checked !== undefined) {
      li.className = "task";
      const box = doc.createElement("input");
      box.type = "checkbox";
      box.disabled = true;
      box.checked = it.checked;
      li.append(box);
    }
    li.append(...renderInline(it.c, ctx));
    if (it.sub) li.append(renderList(it.sub, ctx));
    el.append(li);
  }
  return el;
}

function renderTable(b: Extract<Block, { t: "table" }>, ctx: Ctx): HTMLElement {
  const { doc } = ctx;
  const wrap = doc.createElement("div");
  wrap.className = "md-table";
  const table = doc.createElement("table");
  const cell = (tag: "th" | "td", inl: Inline[], k: number, scope?: "col" | "row") => {
    const c = doc.createElement(tag);
    const a = b.align[k];
    if (a) c.className = ALIGN_CLASS[a];
    if (scope) c.scope = scope;
    c.append(...renderInline(inl, ctx));
    return c;
  };
  const thead = doc.createElement("thead");
  const hr = doc.createElement("tr");
  b.head.forEach((h, k) => hr.append(cell("th", h, k, "col")));
  thead.append(hr);
  const tbody = doc.createElement("tbody");
  for (const row of b.rows) {
    const tr = doc.createElement("tr");
    row.forEach((c, k) => tr.append(b.rowHeaders && k === 0 ? cell("th", c, k, "row") : cell("td", c, k)));
    tbody.append(tr);
  }
  table.append(thead, tbody);
  wrap.append(table);
  return wrap;
}

function renderBlocks(blocks: Block[], ctx: Ctx): Node[] {
  const { doc } = ctx;
  return blocks.map((b): Node => {
    switch (b.t) {
      case "ul":
      case "ol": return renderList(b, ctx);
      case "hr": return doc.createElement("hr");
      case "math": return mathNode(b.v, true, ctx);
      case "table": return renderTable(b, ctx);
      case "quote": {
        const el = doc.createElement("blockquote");
        el.append(...renderBlocks(b.c, ctx));
        return el;
      }
      case "code": {
        const wrap = doc.createElement("div");
        wrap.className = "md-code";
        if (b.lang) {
          const label = doc.createElement("span");
          label.className = "md-code-lang";
          label.append(doc.createTextNode(b.lang));
          wrap.append(label);
        }
        const pre = doc.createElement("pre");
        pre.append(codeFallback(b.v, ctx));
        wrap.append(pre);
        return wrap;
      }
      default: {
        const el = doc.createElement(b.t === "p" ? "p" : `h${b.level}`);
        el.append(...renderInline(b.c, ctx));
        return el;
      }
    }
  });
}

/** Render Markdown text into a `div.md` of DOM nodes (safe for untrusted text). */
export function renderMarkdown(input: string, doc: Doc = document): HTMLElement {
  const root = doc.createElement("div");
  root.className = "md";
  root.append(...renderBlocks(parseMarkdown(input), { doc, math: 0 }));
  return root;
}
