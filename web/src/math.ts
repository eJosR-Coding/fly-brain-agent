/**
 * Math rendering for chat answers. This is the ONLY place where a parsed HTML fragment is created,
 * and it only ever comes from KaTeX output (never from raw model text):
 *   TeX (untrusted) -> KaTeX (trust: false, MathML only) -> DOMPurify (strict allow-list) -> DOM fragment.
 *
 * MathML output needs no inline style attributes, so `style` is stripped wholesale. Failures render the
 * source as text instead (the caller falls back to a code element).
 */
import katex from "katex";
import DOMPurify, { type Config } from "dompurify";

const KATEX_OPTIONS = {
  throwOnError: false,
  trust: false, // no \href, \url, \htmlClass, \htmlStyle, \htmlData, \includegraphics
  strict: "ignore",
  maxSize: 10, // em cap for \rule, \kern and friends
  maxExpand: 100, // macro expansion budget (no expansion bombs)
  output: "mathml",
} as const;

const PURIFY_CONFIG: Config & { RETURN_DOM_FRAGMENT: true } = {
  USE_PROFILES: { mathMl: true },
  ADD_TAGS: ["span"], // KaTeX wraps the <math> element in span.katex / span.katex-error
  FORBID_TAGS: ["style", "script", "iframe", "svg", "img", "a", "object", "embed", "link", "meta", "foreignObject"],
  FORBID_ATTR: ["style", "href", "xlink:href", "src", "srcset", "action", "formaction", "title"],
  ALLOW_DATA_ATTR: false,
  ALLOW_ARIA_ATTR: false,
  RETURN_DOM_FRAGMENT: true,
};

const CACHE_LIMIT = 300;
const cache = new Map<string, DocumentFragment>();

/** Sanitized MathML for a formula, or null when it cannot be rendered safely (caller shows the source). */
export function renderMath(tex: string, display: boolean): DocumentFragment | null {
  if (!DOMPurify.isSupported) return null; // no DOM -> fail closed
  const key = (display ? "D" : "I") + tex;
  const hit = cache.get(key);
  if (hit) return hit.cloneNode(true) as DocumentFragment;
  try {
    const html = katex.renderToString(tex, { ...KATEX_OPTIONS, displayMode: display });
    const frag = DOMPurify.sanitize(html, { ...PURIFY_CONFIG });
    if (!frag.firstChild) return null;
    if (cache.size >= CACHE_LIMIT) cache.clear();
    cache.set(key, frag);
    return frag.cloneNode(true) as DocumentFragment;
  } catch {
    return null;
  }
}
