import { describe, expect, it } from "vitest";
import { MAX_MARKDOWN_CHARS, MAX_MATH_CHARS, MAX_TABLE_COLS, MAX_TABLE_ROWS, parseMarkdown, renderMarkdown, type Block, type Inline } from "../src/markdown";

const text = (v: string): Inline => ({ t: "text", v });
const para = (...c: Inline[]): Block => ({ t: "p", c });

describe("parseMarkdown blocks", () => {
  it("splits paragraphs on blank lines", () => {
    expect(parseMarkdown("one\n\ntwo")).toEqual([para(text("one")), para(text("two"))]);
  });
  it("keeps single newlines as line breaks", () => {
    expect(parseMarkdown("one\ntwo")).toEqual([para(text("one"), { t: "br" }, text("two"))]);
  });
  it("maps headings to h3 / h4", () => {
    expect(parseMarkdown("# a\n## b\n### c\n#### d")).toEqual([
      { t: "h", level: 3, c: [text("a")] },
      { t: "h", level: 3, c: [text("b")] },
      { t: "h", level: 3, c: [text("c")] },
      { t: "h", level: 4, c: [text("d")] },
    ]);
  });
  it("needs a space after the hashes", () => {
    expect(parseMarkdown("#nope")).toEqual([para(text("#nope"))]);
  });
  it("parses unordered lists with * and -", () => {
    const want: Block = { t: "ul", items: [{ c: [text("a")] }, { c: [text("b")] }] };
    expect(parseMarkdown("* a\n* b")).toEqual([want]);
    expect(parseMarkdown("- a\n- b")).toEqual([want]);
  });
  it("parses ordered lists", () => {
    expect(parseMarkdown("1. a\n2. b")).toEqual([{ t: "ol", items: [{ c: [text("a")] }, { c: [text("b")] }] }]);
  });
  it("nests one level by indentation", () => {
    expect(parseMarkdown("* a\n  * b\n  * c\n* d")).toEqual([
      {
        t: "ul",
        items: [
          { c: [text("a")], sub: { t: "ul", items: [{ c: [text("b")] }, { c: [text("c")] }] } },
          { c: [text("d")] },
        ],
      },
    ]);
  });
  it("nests up to three levels and flattens anything deeper into the third", () => {
    const [list] = parseMarkdown("* a\n  * b\n    * c\n      * d\n        * e") as [Extract<Block, { t: "ul" }>];
    const l2 = list.items[0].sub!;
    const l3 = l2.items[0].sub!;
    expect(l2.items.map((i) => i.c)).toEqual([[text("b")]]);
    expect(l3.items.map((i) => i.c)).toEqual([[text("c")], [text("d")], [text("e")]]);
    expect(l3.items.some((i) => i.sub)).toBe(false);
  });
  it("a list ends at a blank line and a paragraph follows", () => {
    expect(parseMarkdown("* a\n\ntext")).toEqual([{ t: "ul", items: [{ c: [text("a")] }] }, para(text("text"))]);
  });
  it("does not mistake **bold** for a list marker", () => {
    expect(parseMarkdown("**bold** start")).toEqual([para({ t: "b", c: [text("bold")] }, text(" start"))]);
  });
  it("returns nothing for blank input", () => {
    expect(parseMarkdown("  \n\n ")).toEqual([]);
  });
});

describe("parseMarkdown rich blocks", () => {
  const cell = (v: string): Inline[] => [text(v)];
  it("parses a GFM table with alignment", () => {
    const [t] = parseMarkdown("| a | b | c |\n|:---|:---:|---:|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |") as [Extract<Block, { t: "table" }>];
    expect(t.t).toBe("table");
    expect(t.align).toEqual(["left", "center", "right"]);
    expect(t.head).toEqual([cell("a"), cell("b"), cell("c")]);
    expect(t.rows).toEqual([[cell("1"), cell("2"), cell("3")], [cell("4"), cell("5"), cell("6")]]);
    expect(t.rowHeaders).toBe(false);
  });
  it("accepts tables without outer pipes and applies inline formatting in cells", () => {
    const [t] = parseMarkdown("x | y\n--- | ---\n**b** | `c`") as [Extract<Block, { t: "table" }>];
    expect(t.rows[0]).toEqual([[{ t: "b", c: [text("b")] }], [{ t: "code", v: "c" }]]);
    expect(t.align).toEqual([null, null]);
  });
  it("unescapes \\| inside cells", () => {
    const [t] = parseMarkdown("| a | b |\n|---|---|\n| x \\| y | z |") as [Extract<Block, { t: "table" }>];
    expect(t.rows[0][0]).toEqual(cell("x | y"));
    expect(t.rows[0]).toHaveLength(2);
  });
  it("pads short rows and truncates long ones to the header width", () => {
    const [t] = parseMarkdown("| a | b |\n|---|---|\n| 1 |\n| 1 | 2 | 3 |") as [Extract<Block, { t: "table" }>];
    expect(t.rows[0]).toEqual([cell("1"), []]);
    expect(t.rows[1]).toEqual([cell("1"), cell("2")]);
  });
  it("caps rows and columns", () => {
    const cols = Array.from({ length: 30 }, () => "x");
    const head = `|${cols.join("|")}|\n|${cols.map(() => "-").join("|")}|`;
    const body = Array.from({ length: 300 }, () => `|${cols.join("|")}|`).join("\n");
    const [t] = parseMarkdown(`${head}\n${body}`) as [Extract<Block, { t: "table" }>];
    expect(t.head).toHaveLength(MAX_TABLE_COLS);
    expect(t.rows).toHaveLength(MAX_TABLE_ROWS);
    expect(t.rows[0]).toHaveLength(MAX_TABLE_COLS);
  });
  it("marks cross tables (empty top-left header) as having row headers", () => {
    const [t] = parseMarkdown("|  | A | B |\n|---|---|---|\n| r1 | 1 | 2 |") as [Extract<Block, { t: "table" }>];
    expect(t.rowHeaders).toBe(true);
    expect(t.head[0]).toEqual([]);
  });
  it("half a table (no delimiter row yet) stays a paragraph", () => {
    expect(parseMarkdown("| a | b |")[0].t).toBe("p");
    expect(parseMarkdown("| a | b |\n|---")[0].t).toBe("p");
  });
  it("a delimiter row with a different column count is not a table", () => {
    expect(parseMarkdown("| a | b |\n|---|---|---|")[0].t).toBe("p");
  });
  it("parses fenced code with a whitelisted-looking label", () => {
    expect(parseMarkdown("```python\nx = 1 <b>\n\ny\n```")).toEqual([{ t: "code", lang: "python", v: "x = 1 <b>\n\ny" }]);
    expect(parseMarkdown("~~~\nplain\n~~~")).toEqual([{ t: "code", lang: "", v: "plain" }]);
  });
  it("drops hostile or odd language labels", () => {
    expect(parseMarkdown('```"><img src=x>\nx\n```')).toEqual([{ t: "code", lang: "", v: "x" }]);
    expect(parseMarkdown("```" + "a".repeat(40) + "\nx\n```")).toEqual([{ t: "code", lang: "", v: "x" }]);
  });
  it("a longer closing fence works and markdown inside a fence is literal", () => {
    expect(parseMarkdown("```\n**x** | y\n---\n````")).toEqual([{ t: "code", lang: "", v: "**x** | y\n---" }]);
  });
  it("an unclosed fence renders as plain text, never as code", () => {
    const [b] = parseMarkdown("```js\nlet x") ;
    expect(b.t).toBe("p");
    expect(JSON.stringify(b)).toContain("```js");
  });
  it("parses indented code", () => {
    expect(parseMarkdown("    a < b\n    c")).toEqual([{ t: "code", lang: "", v: "a < b\nc" }]);
  });
  it("parses blockquotes recursively", () => {
    expect(parseMarkdown("> hi\n> **there**\n>\n> - x")).toEqual([
      { t: "quote", c: [para(text("hi"), { t: "br" }, { t: "b", c: [text("there")] }), { t: "ul", items: [{ c: [text("x")] }] }] },
    ]);
  });
  it("bounds blockquote nesting", () => {
    const b = parseMarkdown(">".repeat(50) + " deep");
    let depth = 0;
    let cur: Block | undefined = b[0];
    while (cur && cur.t === "quote") { depth++; cur = cur.c[0]; }
    expect(depth).toBeLessThanOrEqual(4);
  });
  it("parses horizontal rules", () => {
    for (const r of ["---", "***", "___", "- - -", "  ----"]) expect(parseMarkdown(r)).toEqual([{ t: "hr" }]);
    expect(parseMarkdown("--")).toEqual([para(text("--"))]);
  });
  it("parses task list items", () => {
    expect(parseMarkdown("- [ ] todo\n- [x] done\n- [X] DONE\n- plain")).toEqual([
      { t: "ul", items: [
        { c: [text("todo")], checked: false }, { c: [text("done")], checked: true },
        { c: [text("DONE")], checked: true }, { c: [text("plain")] },
      ] },
    ]);
  });
  it("parses display math on one line and across lines", () => {
    expect(parseMarkdown("$$ x^2 $$")).toEqual([{ t: "math", v: "x^2" }]);
    expect(parseMarkdown("$$\n\\frac{a}{b}\n$$")).toEqual([{ t: "math", v: "\\frac{a}{b}" }]);
    expect(parseMarkdown("\\[ y \\]")).toEqual([{ t: "math", v: "y" }]);
    expect(parseMarkdown("\\[\ny\n\\]")).toEqual([{ t: "math", v: "y" }]);
  });
  it("an unclosed display math block stays plain text", () => {
    const [b] = parseMarkdown("$$\n\\frac{a}{b}");
    expect(b.t).toBe("p");
    expect(JSON.stringify(b)).toContain("$$");
  });
  it("oversized display math becomes a code block", () => {
    const [b] = parseMarkdown("$$ " + "x".repeat(MAX_MATH_CHARS + 1) + " $$");
    expect(b.t).toBe("code");
  });
});

describe("parseMarkdown rich inline", () => {
  const inl = (s: string) => (parseMarkdown(s)[0] as Extract<Block, { t: "p" }>).c;
  it("strikethrough", () => {
    expect(inl("a ~~b~~ c")).toEqual([text("a "), { t: "del", c: [text("b")] }, text(" c")]);
    expect(inl("~~open")).toEqual([text("~~open")]);
  });
  it("inline math with $ and \\( \\)", () => {
    expect(inl("let $x^2$ ok")).toEqual([text("let "), { t: "math", v: "x^2" }, text(" ok")]);
    expect(inl("let \\(x_1\\) ok")).toEqual([text("let "), { t: "math", v: "x_1" }, text(" ok")]);
  });
  it("does not treat currency as math", () => {
    expect(inl("costs $5 and $10")).toEqual([text("costs $5 and $10")]);
    expect(inl("$5 and $10 and $x$")).toEqual([text("$5 and $10 and "), { t: "math", v: "x" }]);
    expect(inl("a $ b $ c")).toEqual([text("a $ b $ c")]);
    expect(inl("$x $5")).toEqual([text("$x $5")]);
  });
  it("math content is not parsed as markdown", () => {
    expect(inl("$a*b*c$")).toEqual([{ t: "math", v: "a*b*c" }]);
  });
  it("unclosed or oversized inline math stays literal", () => {
    expect(inl("$x^2")).toEqual([text("$x^2")]);
    const long = "$" + "x".repeat(MAX_MATH_CHARS + 5) + "$";
    expect(inl(long).some((n) => n.t === "math")).toBe(false);
  });
});

describe("parseMarkdown performance", () => {
  const fast = (s: string, ms = 1500) => {
    const t0 = performance.now();
    parseMarkdown(s);
    expect(performance.now() - t0).toBeLessThan(ms);
  };
  it("pathological inline delimiters", () => {
    fast("`a".repeat(10000));
    fast("$a ".repeat(7000));
    fast("\\(a ".repeat(5000));
    fast("~~a ".repeat(5000));
    fast("*".repeat(20000));
    fast("**a*".repeat(5000));
    fast("[".repeat(20000) + "(".repeat(20000) + "*".repeat(5000));
  });
  it("pathological blocks", () => {
    fast("|a|".repeat(6000));
    fast("| a |\n|---|\n".repeat(1500));
    fast("> ".repeat(10000));
    fast("```\n".repeat(4000));
    fast("$$\n".repeat(5000));
    fast("- ".repeat(5000) + "\n" + "  ".repeat(3000) + "- x");
    fast("- - - - - - - - - -\n".repeat(900));
  });
  it("a 200k input is capped before parsing", () => {
    fast("a | b\n--|--\n1 | 2\n".repeat(20000));
  });
});

describe("parseMarkdown inline", () => {
  const inl = (s: string) => (parseMarkdown(s)[0] as Extract<Block, { t: "p" }>).c;
  it("bold, italic, code", () => {
    expect(inl("a **b** *c* `d`")).toEqual([
      text("a "), { t: "b", c: [text("b")] }, text(" "), { t: "i", c: [text("c")] }, text(" "), { t: "code", v: "d" },
    ]);
  });
  it("bold inside a list item", () => {
    expect(parseMarkdown("* **Name:** value")).toEqual([
      { t: "ul", items: [{ c: [{ t: "b", c: [text("Name:")] }, text(" value")] }] },
    ]);
  });
  it("italic nests inside bold", () => {
    expect(inl("**a *b* c**")).toEqual([{ t: "b", c: [text("a "), { t: "i", c: [text("b")] }, text(" c")] }]);
  });
  it("code content is literal", () => {
    expect(inl("`**x** <b>`")).toEqual([{ t: "code", v: "**x** <b>" }]);
  });
  it("unclosed markers stay literal", () => {
    expect(inl("**open")).toEqual([text("**open")]);
    expect(inl("a * b")).toEqual([text("a * b")]);
    expect(inl("`open")).toEqual([text("`open")]);
    expect(inl("*open")).toEqual([text("*open")]);
  });
  it("links stay literal text", () => {
    expect(inl("[click](https://evil.example)")).toEqual([text("[click](https://evil.example)")]);
  });
  it("caps very long input", () => {
    const blocks = parseMarkdown("x".repeat(MAX_MARKDOWN_CHARS * 3));
    const len = (blocks[0] as Extract<Block, { t: "p" }>).c.map((c) => (c.t === "text" ? c.v.length : 0)).reduce((a, b) => a + b, 0);
    expect(len).toBeLessThanOrEqual(MAX_MARKDOWN_CHARS + 2);
  });
  it("stays fast on pathological input", () => {
    const t0 = performance.now();
    parseMarkdown("*a ".repeat(6000) + "**".repeat(3000));
    expect(performance.now() - t0).toBeLessThan(1500);
  });
});

// A minimal document double: records how nodes are built, so we can prove nothing is parsed as HTML.
class FakeNode {
  children: FakeNode[] = [];
  className = "";
  constructor(public tag: string, public data = "") {}
  append(...n: FakeNode[]) { this.children.push(...n); }
  set textContent(v: string) { this.children = [new FakeNode("#text", v)]; }
}
const fakeDoc = {
  createElement: (t: string) => new FakeNode(t),
  createTextNode: (d: string) => new FakeNode("#text", d),
} as unknown as Pick<Document, "createElement" | "createTextNode">;
const tags = (n: FakeNode): string[] => [n.tag, ...n.children.flatMap(tags)];

describe("renderMarkdown", () => {
  it("builds elements for each construct", () => {
    const root = renderMarkdown("### T\n\n**b** and `c`\n\n* x\n* y\n\n1. z", fakeDoc) as unknown as FakeNode;
    expect(tags(root)).toEqual(expect.arrayContaining(["h3", "p", "strong", "code", "ul", "li", "ol"]));
  });
  it("never creates elements from HTML in the input", () => {
    const root = renderMarkdown('<img src=x onerror=alert(1)><script>x</script> **b**', fakeDoc) as unknown as FakeNode;
    const all = tags(root).filter((t) => t !== "#text");
    expect(all).not.toContain("img");
    expect(all).not.toContain("script");
    expect(all).toEqual(["div", "p", "strong"]);
    const literal = JSON.stringify(root);
    expect(literal).toContain("<img src=x onerror=alert(1)>");
  });
});
