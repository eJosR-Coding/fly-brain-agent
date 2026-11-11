import { describe, expect, it } from "vitest";
import { MAX_MARKDOWN_CHARS, parseMarkdown, renderMarkdown, type Block, type Inline } from "../src/markdown";

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
  it("flattens deeper nesting into the second level", () => {
    const [list] = parseMarkdown("* a\n  * b\n    * c") as [Extract<Block, { t: "ul" }>];
    expect(list.items[0].sub?.items.map((i) => i.c)).toEqual([[text("b")], [text("c")]]);
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
