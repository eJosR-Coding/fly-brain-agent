/**
 * A tiny, dependency-free Markdown renderer for chat answers.
 * Model output is untrusted: the parser yields a plain AST and the renderer builds DOM nodes with
 * createElement + text nodes only. Nothing is ever parsed as HTML, links are never anchors.
 */

export type Inline =
  | { t: "text"; v: string }
  | { t: "b"; c: Inline[] }
  | { t: "i"; c: Inline[] }
  | { t: "code"; v: string }
  | { t: "br" };

export interface ListNode {
  t: "ul" | "ol";
  items: { c: Inline[]; sub?: ListNode }[];
}

export type Block = { t: "p"; c: Inline[] } | { t: "h"; level: 3 | 4; c: Inline[] } | ListNode;

export const MAX_MARKDOWN_CHARS = 20_000;

const LIST_RE = /^(\s*)([*-]|\d+\.)[ \t]+(.*)$/;
const HEADING_RE = /^(#{1,6})[ \t]+(.*)$/;

/** Inline spans: `code`, **bold**, *italic*. Anything unclosed stays literal. */
function parseInline(s: string): Inline[] {
  const out: Inline[] = [];
  let buf = "";
  const flush = () => {
    if (buf) out.push({ t: "text", v: buf });
    buf = "";
  };
  let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (ch === "`") {
      const j = s.indexOf("`", i + 1);
      if (j > i + 1) {
        flush();
        out.push({ t: "code", v: s.slice(i + 1, j) });
        i = j + 1;
        continue;
      }
    } else if (ch === "*" && s[i + 1] === "*") {
      const j = s.indexOf("**", i + 2);
      if (j > i + 2) {
        flush();
        out.push({ t: "b", c: parseInline(s.slice(i + 2, j)) });
        i = j + 2;
        continue;
      }
      buf += "**";
      i += 2;
      continue;
    } else if (ch === "*" && s[i + 1] !== undefined && !/\s/.test(s[i + 1])) {
      const j = s.indexOf("*", i + 1);
      if (j > i + 1 && !/\s/.test(s[j - 1])) {
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

export function parseMarkdown(input: string): Block[] {
  const src = input.length > MAX_MARKDOWN_CHARS ? input.slice(0, MAX_MARKDOWN_CHARS) : input;
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: ListNode | null = null;
  let item: ListNode["items"][number] | null = null;

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
    list = null;
    item = null;
  };

  for (const raw of src.split(/\r?\n/)) {
    if (!raw.trim()) {
      endPara();
      endList();
      continue;
    }
    const li = LIST_RE.exec(raw);
    if (li) {
      endPara();
      const kind = /\d/.test(li[2]) ? "ol" : "ul";
      const content = parseInline(li[3]);
      if (li[1].length >= 2 && list && item) {
        item.sub ??= { t: kind, items: [] };
        item.sub.items.push({ c: content });
      } else {
        if (!list || list.t !== kind) {
          list = { t: kind, items: [] };
          blocks.push(list);
        }
        item = { c: content };
        list.items.push(item);
      }
      continue;
    }
    endList();
    const h = HEADING_RE.exec(raw);
    if (h) {
      endPara();
      blocks.push({ t: "h", level: h[1].length >= 4 ? 4 : 3, c: parseInline(h[2]) });
      continue;
    }
    para.push(raw.trim());
  }
  endPara();
  return blocks;
}

type Doc = Pick<Document, "createElement" | "createTextNode">;

function renderInline(nodes: Inline[], doc: Doc): Node[] {
  return nodes.map((n) => {
    switch (n.t) {
      case "text": return doc.createTextNode(n.v);
      case "br": return doc.createElement("br");
      case "code": {
        const el = doc.createElement("code");
        el.append(doc.createTextNode(n.v));
        return el;
      }
      default: {
        const el = doc.createElement(n.t === "b" ? "strong" : "em");
        el.append(...renderInline(n.c, doc));
        return el;
      }
    }
  });
}

function renderList(l: ListNode, doc: Doc): HTMLElement {
  const el = doc.createElement(l.t);
  for (const it of l.items) {
    const li = doc.createElement("li");
    li.append(...renderInline(it.c, doc));
    if (it.sub) li.append(renderList(it.sub, doc));
    el.append(li);
  }
  return el;
}

/** Render Markdown text into a `div.md` of DOM nodes (safe for untrusted text). */
export function renderMarkdown(input: string, doc: Doc = document): HTMLElement {
  const root = doc.createElement("div");
  root.className = "md";
  for (const b of parseMarkdown(input)) {
    if (b.t === "ul" || b.t === "ol") {
      root.append(renderList(b, doc));
    } else if (b.t === "p" || b.t === "h") {
      const el = doc.createElement(b.t === "p" ? "p" : `h${b.level}`);
      el.append(...renderInline(b.c, doc));
      root.append(el);
    }
  }
  return root;
}
