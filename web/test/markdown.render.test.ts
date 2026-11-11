// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import DOMPurify from "dompurify";
import katex from "katex";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MAX_MATH_CHARS, MAX_MATH_NODES, renderMarkdown } from "../src/markdown";

const source = (f: string) => readFileSync(resolve(process.cwd(), "src", f), "utf8");

const FORBIDDEN_TAGS = ["script", "iframe", "object", "embed", "svg", "img", "a", "style", "link", "meta", "form", "base"];

/** The safety contract for any rendered answer. */
function assertSafe(root: HTMLElement) {
  for (const tag of FORBIDDEN_TAGS) expect(root.querySelector(tag), `<${tag}> present`).toBeNull();
  for (const el of [root, ...root.querySelectorAll("*")]) {
    for (const attr of Array.from(el.attributes)) {
      expect(attr.name.toLowerCase().startsWith("on"), `attr ${attr.name}`).toBe(false);
      expect(attr.name.toLowerCase(), `attr ${attr.name}`).not.toBe("style");
      expect(attr.value, `value of ${attr.name}`).not.toMatch(/javascript:|data:/i);
    }
  }
}

const hostile: { name: string; md: string; visible?: string }[] = [
  { name: "script tag", md: "<script>alert(1)</script>", visible: "<script>alert(1)</script>" },
  { name: "img onerror", md: "<img src=x onerror=alert(1)>", visible: "<img src=x onerror=alert(1)>" },
  { name: "javascript link", md: "[x](javascript:alert(1))", visible: "[x](javascript:alert(1))" },
  { name: "remote image", md: "![a](https://evil/x.png)", visible: "![a](https://evil/x.png)" },
  { name: "style tag", md: "<style>body{display:none}</style>", visible: "<style>body{display:none}</style>" },
  { name: "iframe", md: '<iframe src="https://evil"></iframe>', visible: "<iframe" },
  { name: "svg onload", md: "<svg onload=alert(1)><circle/></svg>", visible: "<svg onload=alert(1)>" },
  { name: "data uri", md: "[x](data:text/html;base64,PHNjcmlwdD4=)", visible: "data:text/html" },
  { name: "html in table cell", md: "| a | b |\n|---|---|\n| <b onmouseover=alert(1)>x</b> | y |", visible: "<b onmouseover=alert(1)>x</b>" },
  { name: "html in table header", md: "| <img src=x onerror=alert(1)> | b |\n|---|---|\n| 1 | 2 |", visible: "<img src=x onerror=alert(1)>" },
  { name: "code fence with html", md: "```html\n<script>alert(1)</script>\n```", visible: "<script>alert(1)</script>" },
  { name: "hostile language label", md: '```" onclick="alert(1)\nx\n```' },
  { name: "html in blockquote", md: "> <img src=x onerror=alert(1)>", visible: "<img src=x onerror=alert(1)>" },
  { name: "html in task item", md: "- [x] <script>alert(1)</script>", visible: "<script>" },
  { name: "html in list", md: "* <a href='javascript:alert(1)'>x</a>", visible: "<a href=" },
  { name: "css overlay", md: '<div style="position:fixed;inset:0;background:red">x</div>', visible: 'style="position:fixed' },
  { name: "tex href", md: "$\\href{javascript:alert(1)}{x}$" },
  { name: "tex url", md: "$\\url{javascript:alert(1)}$" },
  { name: "tex htmlData", md: "$$\\htmlData{foo=bar}{x}$$" },
  { name: "tex htmlClass", md: "$\\htmlClass{evil}{x}$" },
  { name: "tex style", md: "$\\style{color:red;position:fixed}{x}$" },
  { name: "tex htmlStyle", md: "$\\htmlStyle{position:fixed}{x}$" },
  { name: "tex includegraphics", md: "$\\includegraphics{https://evil/x.png}$" },
  { name: "tex text with html", md: "$\\text{<img src=x onerror=alert(1)>}$" },
  { name: "tex macro bomb", md: "$\\def\\a{\\a\\a}\\a$" },
  { name: "tex huge rule", md: "$\\rule{99999em}{99999em}$" },
  { name: "tex bracket display html", md: "\\[ \\text{</math><script>alert(1)</script>} \\]" },
  { name: "bidi override", md: "abc \u202Eevil\u202C def", visible: "\u202Eevil" },
];

describe("hostile corpus", () => {
  for (const c of hostile) {
    it(c.name, () => {
      const root = renderMarkdown(c.md);
      assertSafe(root);
      if (c.visible) expect(root.textContent).toContain(c.visible);
    });
  }
  it("stays safe when every hostile input is concatenated", () => {
    assertSafe(renderMarkdown(hostile.map((h) => h.md).join("\n\n")));
  });
});

describe("tables", () => {
  const md = "| Name | Qty | Note |\n|:---|:---:|---:|\n| **a** | 1 | x \\| y |\n| b | 2 | |";
  it("renders thead/tbody in a scroll wrapper with alignment classes", () => {
    const root = renderMarkdown(md);
    const wrap = root.querySelector("div.md-table")!;
    expect(wrap.querySelector("table > thead > tr > th")).not.toBeNull();
    expect(wrap.querySelectorAll("tbody tr")).toHaveLength(2);
    const ths = wrap.querySelectorAll("thead th");
    expect([...ths].map((t) => t.className)).toEqual(["al-left", "al-center", "al-right"]);
    expect(wrap.querySelector("tbody tr td strong")?.textContent).toBe("a");
    expect(wrap.querySelectorAll("tbody td")[2].textContent).toBe("x | y");
    expect(root.querySelector("[style]")).toBeNull();
  });
  it("cross tables use row headers", () => {
    const root = renderMarkdown("|  | A | B |\n|---|---|---|\n| r1 | 1 | 2 |\n| r2 | 3 | 4 |");
    const rows = root.querySelectorAll("tbody tr");
    expect(rows).toHaveLength(2);
    for (const r of rows) {
      expect(r.firstElementChild?.tagName).toBe("TH");
      expect(r.firstElementChild?.getAttribute("scope")).toBe("row");
      expect(r.querySelectorAll("td")).toHaveLength(2);
    }
    expect(root.querySelector("thead th")?.textContent).toBe("");
  });
  it("a normal table keeps td in the first column", () => {
    const root = renderMarkdown("| h | i |\n|---|---|\n| 1 | 2 |");
    expect(root.querySelectorAll("tbody th")).toHaveLength(0);
  });
});

describe("code, quotes, rules, tasks", () => {
  it("code blocks are pre > code text with an optional plain label", () => {
    const root = renderMarkdown("```python\nprint('<b>')\n```");
    expect(root.querySelector("div.md-code > span.md-code-lang")?.textContent).toBe("python");
    expect(root.querySelector("div.md-code pre > code")?.textContent).toBe("print('<b>')");
    expect(root.querySelector("b")).toBeNull();
  });
  it("a code block without a label has no label element", () => {
    expect(renderMarkdown("```\nx\n```").querySelector(".md-code-lang")).toBeNull();
  });
  it("blockquote, hr, strikethrough", () => {
    const root = renderMarkdown("> quoted ~~gone~~\n\n---\n\nafter");
    expect(root.querySelector("blockquote del")?.textContent).toBe("gone");
    expect(root.querySelector("hr")).not.toBeNull();
  });
  it("task items are disabled checkboxes created by code", () => {
    const root = renderMarkdown("- [ ] a\n- [x] b");
    const boxes = [...root.querySelectorAll<HTMLInputElement>("li.task > input")];
    expect(boxes.map((b) => [b.type, b.disabled, b.checked])).toEqual([["checkbox", true, false], ["checkbox", true, true]]);
  });
  it("lists nest to three levels", () => {
    const root = renderMarkdown("- a\n  - b\n    - c\n      - d");
    expect(root.querySelector("ul ul ul")).not.toBeNull();
    expect(root.querySelector("ul ul ul ul")).toBeNull();
  });
});

describe("math", () => {
  it("renders inline and display math as MathML without inline styles", () => {
    const root = renderMarkdown("rate $\\frac{a}{b}$ and\n\n$$\\sum_{i=1}^{n} x_i$$");
    expect(root.querySelector("span.md-math math")).not.toBeNull();
    expect(root.querySelector("div.md-math-display math")).not.toBeNull();
    expect(root.querySelector("mfrac")).not.toBeNull();
    expect(root.querySelector("[style]")).toBeNull();
    assertSafe(root);
  });
  it("supports \\( \\) and \\[ \\] delimiters", () => {
    const root = renderMarkdown("\\(x_1\\)\n\n\\[\ny^2\n\\]");
    expect(root.querySelectorAll("math")).toHaveLength(2);
  });
  it("currency is plain text", () => {
    const root = renderMarkdown("It costs $5 and $10 in total.");
    expect(root.querySelector("math")).toBeNull();
    expect(root.textContent).toBe("It costs $5 and $10 in total.");
  });
  it("invalid TeX does not throw and shows something", () => {
    const root = renderMarkdown("$\\frac{$ and $\\notacommand$");
    expect(root.textContent!.length).toBeGreaterThan(0);
    assertSafe(root);
  });
  it("formulas over the cap render as plain code text", () => {
    const long = "x".repeat(MAX_MATH_CHARS + 1);
    const root = renderMarkdown(`$$${long}$$`);
    expect(root.querySelector("math")).toBeNull();
    expect(root.querySelector("pre code")?.textContent).toBe(long);
  });
  it("bounds the number of formulas per answer", () => {
    const root = renderMarkdown(Array.from({ length: MAX_MATH_NODES + 40 }, (_, i) => `$a_${i}$`).join(" "));
    expect(root.querySelectorAll("span.md-math")).toHaveLength(MAX_MATH_NODES);
    expect(root.querySelectorAll("span.md-math-src").length).toBe(40);
  });
  it("macro expansion bombs terminate quickly", () => {
    const t0 = performance.now();
    renderMarkdown("$\\def\\a{\\a\\a}\\a$");
    expect(performance.now() - t0).toBeLessThan(1500);
  });
});

describe("DOMPurify only ever sees KaTeX output", () => {
  afterEach(() => vi.restoreAllMocks());
  it("is not called for text without math, however hostile", () => {
    const spy = vi.spyOn(DOMPurify, "sanitize");
    renderMarkdown(hostile.filter((h) => !h.name.startsWith("tex")).map((h) => h.md).join("\n\n"));
    expect(spy).not.toHaveBeenCalled();
  });
  it("is called with exactly the KaTeX string for a formula", () => {
    const sanitize = vi.spyOn(DOMPurify, "sanitize");
    const render = vi.spyOn(katex, "renderToString");
    renderMarkdown("$zz_{unique1}$");
    expect(render).toHaveBeenCalledTimes(1);
    expect(sanitize).toHaveBeenCalledTimes(1);
    expect(sanitize.mock.calls[0][0]).toBe(render.mock.results[0].value);
    expect(render.mock.calls[0][1]).toMatchObject({ trust: false, throwOnError: false, maxExpand: 100, output: "mathml" });
  });
  it("source files never use HTML-parsing or attribute-writing sinks", () => {
    for (const f of ["markdown.ts", "math.ts"]) {
      const raw = source(f);
      expect(raw.length).toBeGreaterThan(500);
      const src = raw
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/^\s*\/\/.*$/gm, "");
      for (const sink of ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "setAttribute", "eval(", "new Function", "createContextualFragment", "DOMParser"]) {
        expect(src, `${f} uses ${sink}`).not.toContain(sink);
      }
    }
  });
  it("sanitize is only referenced in math.ts", () => {
    const md = source("markdown.ts");
    expect(md.length).toBeGreaterThan(500);
    expect(md).not.toContain("DOMPurify");
    expect(md).not.toContain("katex");
  });
});

describe("streaming", () => {
  const doc = [
    "### Result", "", "| a | b |", "|:--|--:|", "| 1 | $x^2$ |", "", "```py", "x = 1", "```", "",
    "$$", "\\frac{a}{b}", "$$", "", "> quote", "", "- [x] done", "  - nested", "", "$5 and $10",
  ].join("\n");
  it("never throws on any prefix of a rich answer", () => {
    for (let n = 0; n <= doc.length; n++) assertSafe(renderMarkdown(doc.slice(0, n)));
  });
  it("unclosed constructs are plain text until closed", () => {
    expect(renderMarkdown("```js\nlet x = 1").querySelector("pre")).toBeNull();
    expect(renderMarkdown("```js\nlet x = 1").textContent).toContain("let x = 1");
    expect(renderMarkdown("$$\n\\frac{a}{b}").querySelector("math")).toBeNull();
    expect(renderMarkdown("| a | b |\n|---").querySelector("table")).toBeNull();
    expect(renderMarkdown("| a | b |").querySelector("table")).toBeNull();
    expect(renderMarkdown("```js\nlet x = 1\n```").querySelector("pre")).not.toBeNull();
    expect(renderMarkdown("$$\nx\n$$").querySelector("math")).not.toBeNull();
  });
});

describe("hostile size and shape", () => {
  const fast = (md: string, ms = 3000) => {
    const t0 = performance.now();
    assertSafe(renderMarkdown(md));
    expect(performance.now() - t0).toBeLessThan(ms);
  };
  it("a 200k-char input", () => fast("a".repeat(200_000)));
  it("a 200k-char input of table rows", () => fast("| a | $b$ |\n|---|---|\n" + "| c | d |\n".repeat(20_000)));
  it("deeply nested brackets and asterisks", () => {
    fast("[".repeat(10_000) + "*".repeat(5_000) + "_".repeat(5_000));
    fast("*a **b *c **d ".repeat(1500));
    fast("> ".repeat(10_000) + "x");
    fast("- ".repeat(5_000));
  });
  it("many math delimiters", () => {
    fast("$a ".repeat(7_000));
    fast("$$".repeat(9_000));
    fast("\\(".repeat(9_000));
    fast("$x$ ".repeat(5_000));
  });
});
