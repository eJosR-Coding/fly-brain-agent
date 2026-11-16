"""The doll's conditions: which stimuli are driven (and how hard), which neurons are silenced, which readouts are watched.

Pure Python, no I/O. These are inputs of the neuron model only; temperature, hunger or sleep are not model inputs.
"""

MAX_STIMULI = 6


def _fmt(rate: float) -> str:
    return f"{rate:g}"


class Doll:
    """Persistent conditions plus a summary of the previous run (`last`), for before/after comparisons."""

    def __init__(self) -> None:
        self.stimuli: dict[str, float] = {}
        self.silenced: set[str] = set()
        self.readouts: list[str] = []  # explicit ones; the stimulus presets' own readouts are derived per run
        self.last: dict | None = None

    def has_stimulus(self) -> bool:
        return bool(self.stimuli)

    def conditions(self) -> dict:
        """A copy: {"stimuli": {name: rate_hz}, "silenced": sorted names, "readouts": explicit names}."""
        return {"stimuli": dict(self.stimuli), "silenced": sorted(self.silenced), "readouts": list(self.readouts)}

    def apply(self, *, stimuli=None, remove=None, silence=None, readouts=None, reset: bool = False) -> list[str]:
        """reset -> remove -> set stimuli/silence/readouts, all or nothing. Returns notes (unknown names to remove)."""
        new_stimuli = {} if reset else dict(self.stimuli)
        new_silenced = set() if reset else set(self.silenced)
        new_readouts = [] if reset else list(self.readouts)
        notes = []
        for name in remove or ():
            found = False
            if name in new_stimuli:
                del new_stimuli[name]
                found = True
            if name in new_silenced:
                new_silenced.discard(name)
                found = True
            if name in new_readouts:
                new_readouts.remove(name)
                found = True
            if not found:
                notes.append(f"{name!r} was not a stimulus, silenced set or readout of the doll; nothing to remove")
        for name, rate in (stimuli or {}).items():
            new_stimuli[name] = float(rate)
        new_silenced.update(silence or ())
        for name in readouts or ():
            if name not in new_readouts:
                new_readouts.append(name)
        if len(new_stimuli) > MAX_STIMULI:
            raise ValueError(
                f"the doll takes at most {MAX_STIMULI} stimuli at once, this would make {len(new_stimuli)}; "
                "remove one first"
            )
        self.stimuli, self.silenced, self.readouts = new_stimuli, new_silenced, new_readouts
        return notes


def describe_change(before: dict, after: dict) -> str:
    """Short plain text of how `after` differs from `before` (both from `Doll.conditions()`)."""
    parts = []
    for name, rate in after["stimuli"].items():
        old = before["stimuli"].get(name)
        if old is None:
            parts.append(f"added {name} {_fmt(rate)} Hz")
        elif old != rate:
            parts.append(f"{name} {_fmt(old)} -> {_fmt(rate)} Hz")
    parts += [f"removed {name}" for name in before["stimuli"] if name not in after["stimuli"]]
    parts += [f"silenced {name}" for name in after["silenced"] if name not in before["silenced"]]
    parts += [f"unsilenced {name}" for name in before["silenced"] if name not in after["silenced"]]
    parts += [f"watching {name}" for name in after["readouts"] if name not in before["readouts"]]
    parts += [f"stopped watching {name}" for name in before["readouts"] if name not in after["readouts"]]
    return ", ".join(parts) or "no change in conditions"
