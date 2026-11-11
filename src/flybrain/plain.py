"""Plain-English descriptions of neurons: a curated glossary plus a template. Pure, no I/O besides the JSON."""

import json
import re
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def glossary() -> dict:
    """data/glossary.json (package data), read once."""
    text = resources.files("flybrain").joinpath("data/glossary.json").read_text(encoding="utf-8")
    g = json.loads(text)
    g["families"] = [(re.compile(f["pattern"]), f["text"]) for f in g["families"]]
    return g


def _neuropils(names: list[str]) -> str:
    g = glossary()["neuropils"]
    words = [f"{g.get(n, n)} ({n})" for n in names]
    return " and ".join(words) if len(words) < 3 else ", ".join(words[:-1]) + " and " + words[-1]


def _optic_class(cls: str) -> str | None:
    """'ME>LO.LOP' style names: neuropil codes, '>' = from..to, '.' = several. None when it is not one."""
    codes = glossary()["neuropils"]
    parts = re.split(r"[>.]", cls)
    if not parts or any(p not in codes for p in parts):
        return None
    if ">" in cls:
        src, dst = cls.split(">", 1)
        return f"An optic-lobe neuron that carries signals from the {_neuropils([src])} to the {_neuropils(dst.split('.'))}."
    if "." in cls:
        return f"An optic-lobe neuron that spans the {_neuropils(cls.split('.'))}."
    return f"An optic-lobe neuron that stays inside the {_neuropils([cls])}."


def _what_it_is(cell_type, cell_class, super_class) -> str | None:
    g = glossary()
    for pattern, text in g["families"]:
        if cell_type and pattern.search(cell_type):
            return text
    if cell_class:
        return g["cell_class"].get(cell_class) or _optic_class(cell_class) or g["super_class"].get(super_class)
    return g["super_class"].get(super_class)


def plain_description(info: dict, output_super_class: str | None = None) -> str:
    """1-3 sentences for a non-specialist: what it is, what its transmitter does, its strongest output.

    `info` has cell_type / cell_class / super_class / top_nt (any may be None).
    """
    g = glossary()
    what = _what_it_is(info.get("cell_type"), info.get("cell_class"), info.get("super_class"))
    nt = g["nt"].get(info.get("top_nt") or "")
    target = g["super_class_target"].get(output_super_class or "")
    if what is None and nt is None:
        return "A neuron with no annotation in this atlas."
    parts = [what or "A neuron of unrecorded class.", nt]
    if target:
        parts.append(f"Its strongest output is {target}.")
    return " ".join(p for p in parts if p)
