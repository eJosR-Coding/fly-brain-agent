"""Spontaneous drive: OUR labelled behaviour scheduler for the live glass cube (not the connectome, not validated).

The connectome model is silent without input, so a live fly with no stimulus stands still forever. Real flies alternate
resting, walking, turning, grooming and (rarely) taking off. This module is a small state machine over
rest / walk / turn_left / turn_right / groom / takeoff. It never moves the body: each state only INJECTS Poisson rates into
the brain's command neurons (DNg100 for walking, unilateral DNa02 for turns, the anterior-grooming DNs DNg10/11/12, DNp01
for the giant-fibre takeoff trigger) and the usual brain -> DN readout -> body path does the rest. If a commanded DN does
not respond, the session's message shows it (`spont.response`): nothing is faked.

Bout durations and transition probabilities are OUR numbers (see DURATIONS_S / TRANSITIONS and SOURCES for what the
literature does and does not give); they are not fitted to data. The drive is seeded, so a seed reproduces the sequence.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

STATES = ("rest", "walk", "turn_left", "turn_right", "groom", "takeoff")
MODES = ("ethogram", "off")

# mean bout duration (s) and the (min, max) clamp of an exponential draw: OUR numbers
DURATIONS_S = {"rest": (2.0, 0.8, 5.0), "walk": (3.0, 1.0, 8.0), "turn_left": (0.7, 0.4, 1.5), "turn_right": (0.7, 0.4, 1.5),
               "groom": (3.0, 1.5, 6.0)}
# P(next | current): OUR numbers. Takeoff is rare; it is followed by rest once the fly has landed.
TRANSITIONS = {
    "rest": {"walk": 0.50, "groom": 0.25, "turn_left": 0.07, "turn_right": 0.07, "takeoff": 0.04},
    "walk": {"rest": 0.35, "turn_left": 0.20, "turn_right": 0.20, "groom": 0.15, "takeoff": 0.03},
    "turn_left": {"walk": 0.60, "rest": 0.25, "groom": 0.10, "takeoff": 0.02},
    "turn_right": {"walk": 0.60, "rest": 0.25, "groom": 0.10, "takeoff": 0.02},
    "groom": {"rest": 0.45, "walk": 0.40, "turn_left": 0.05, "turn_right": 0.05, "takeoff": 0.02},
    "takeoff": {"rest": 1.0},
}
# Poisson rate (Hz) injected into each commanded group while a state lasts; calibrated against the brain's DN response
# (scratch calibration: a directly driven DN fires about at its input rate, so 40 Hz on DNg100 gives forward ~0.7 and 40 Hz on one DNa02 gives turn ~0.65 without recruiting other DNs). OUR constants.
INJECT_HZ = {"walk": {"dng100": 40.0}, "turn": {"dng100": 40.0, "dna02": 40.0}, "groom": {"groom": 20.0}, "takeoff": {"dnp01": 60.0}}
TAKEOFF_PULSE_MS = 150.0  # the giant-fibre drive lasts this long (the live GF rule needs DNp01 in 2 of 3 chunks of 50 ms)
CALIBRATION_NOTE = "injection rates were chosen so the commanded neurons answer in the model; see `response` for what they did"
GROOM_ON_HZ = 8.0  # mean Hz of the grooming DNs that makes the body groom (ours); off below GROOM_OFF_HZ (hysteresis)
GROOM_OFF_HZ = 3.0
HEAD_SACCADE_DEG = 9.0  # head saccade amplitude at rest / walking (ours)
HEAD_SACCADE_MS = 140.0
HEAD_SACCADE_EVERY_S = (1.2, 4.0)

SOURCES = {
    "states": "Berman, Choi, Bialek, Shaevitz 2014, J R Soc Interface 11:20140672 (https://royalsocietypublishing.org/doi/10.1098/rsif.2014.0672): "
              "free flies show >100 stereotyped states incl. locomotion gaits, grooming and idle/slow movement. VERIFIED (abstract). State durations and "
              "transition matrix: NOT FOUND in what was read.",
    "grooming_bouts": "Individual grooming bouts ~150 ms - 2 s (Mueller et al. 2019, PLoS Comput Biol, https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007105) "
                      "and a typical spontaneous bout > 3 s (bioRxiv 2017, k-nearest-neighbors grooming classifier, https://www.biorxiv.org/content/10.1101/166561v1.full): VERIFIED as search snippets only; our 3 s mean groom is in that range.",
    "grooming_circuit": "DNg10 / DNg12 (with aDN1/aDN2) are anterior-grooming DNs (Braun et al. 2024 Nature, https://www.nature.com/articles/s41586-024-07523-9): VERIFIED (snippet); "
                        "aDN1/aDN2 are not annotated by that name in this atlas, so DNg10/11/12 are used.",
    "walk_rest_turn_takeoff": "bout durations and transition probabilities: NOT FOUND; OUR labelled numbers.",
}

NOTE = ("Spontaneous drive (ours, not the connectome, not validated): a state machine over rest / walk / turn / groom / takeoff injects "
        "Poisson rates into command neurons (DNg100, DNa02, grooming DNg10/11/12, DNp01); the brain -> body path does the rest. "
        "Bout durations and transitions are our labelled numbers.")


@dataclass
class Spontaneous:
    """rows: {'dng100','dna02_left','dna02_right','groom','dnp01'} atlas rows. `step` returns the injection for the chunk."""

    rows: dict[str, np.ndarray]
    seed: int = 0
    mode: str = "ethogram"
    chunk_ms: float = 50.0
    state: str = "rest"
    left_ms: float = 0.0
    log: list = field(default_factory=list)  # (t_ms, state)
    counts: dict = field(default_factory=lambda: {s: 0 for s in STATES})

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"spont mode must be one of {MODES}")
        self._rng = random.Random(self.seed * 7919 + 13)
        self._pulse_ms = 0.0
        self._next_saccade_ms = self._rng.uniform(*HEAD_SACCADE_EVERY_S) * 1000.0
        self._saccade = (0.0, 0.0)  # (target deg, ms left)
        self.left_ms = self._draw("rest")
        self.log.append((0.0, "rest"))
        self.counts["rest"] = 1

    # -- scheduling ------------------------------------------------------------------------

    def _draw(self, state: str) -> float:
        mean, lo, hi = DURATIONS_S[state]
        return min(max(self._rng.expovariate(1.0 / mean), lo), hi) * 1000.0

    def _next(self, state: str) -> str:
        probs = TRANSITIONS[state]
        return self._rng.choices(list(probs), weights=list(probs.values()))[0]

    def _enter(self, t_ms: float, state: str) -> None:
        self.state = state
        self.counts[state] += 1
        self.log.append((round(t_ms, 1), state))
        if state == "takeoff":
            self._pulse_ms = TAKEOFF_PULSE_MS
        else:
            self.left_ms = self._draw(state)

    def step(self, t_ms: float, *, airborne: bool = False) -> dict:
        """Advance one chunk. Returns {'rates': {group: Hz}, 'state', 'head_deg'}: group rates to inject this chunk and the
        head saccade target (deg, + left). While the body is in its takeoff / flight the machine waits for it to finish."""
        if self.mode == "off":
            return {"rates": {}, "state": "off", "head_deg": 0.0}
        dt = self.chunk_ms
        if airborne and self.state != "takeoff":  # the brain's own giant-fibre rule fired (e.g. a wall looming in the eye): follow the body
            self._enter(t_ms, "takeoff")
            self._pulse_ms = 0.0
        if self.state == "takeoff":
            self._pulse_ms -= dt
            if self._pulse_ms <= -dt and not airborne:  # pulse over and the body is no longer busy: it has landed
                self._enter(t_ms, "rest")
        else:
            self.left_ms -= dt
            if self.left_ms <= 0:
                self._enter(t_ms, self._next(self.state))
        rates: dict[str, float] = {}
        s = self.state
        if s == "walk":
            rates = dict(INJECT_HZ["walk"])
        elif s in ("turn_left", "turn_right"):
            rates = {"dng100": INJECT_HZ["turn"]["dng100"], ("dna02_left" if s == "turn_left" else "dna02_right"): INJECT_HZ["turn"]["dna02"]}
        elif s == "groom":
            rates = dict(INJECT_HZ["groom"])
        elif s == "takeoff" and self._pulse_ms > 0:
            rates = dict(INJECT_HZ["takeoff"])
        # head saccades: brief small yaw steps at random intervals (not while grooming or flying: those have their own head motion)
        head = 0.0
        if s not in ("groom", "takeoff"):
            self._next_saccade_ms -= dt
            if self._next_saccade_ms <= 0:
                self._saccade = (self._rng.choice((-1, 1)) * HEAD_SACCADE_DEG * self._rng.uniform(0.5, 1.0), HEAD_SACCADE_MS)
                self._next_saccade_ms = self._rng.uniform(*HEAD_SACCADE_EVERY_S) * 1000.0
            if self._saccade[1] > 0:
                head = self._saccade[0]
                self._saccade = (self._saccade[0], self._saccade[1] - dt)
        return {"rates": rates, "state": s, "head_deg": head}

    def inject(self, rates: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        """(rows, hz) for the group rates of `step` (unknown or empty groups are skipped)."""
        rr, hh = [], []
        for g, hz in rates.items():
            r = self.rows.get(g)
            if r is not None and len(r):
                rr.append(np.asarray(r, dtype=np.int64))
                hh.append(np.full(len(r), float(hz)))
        if not rr:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        return np.concatenate(rr), np.concatenate(hh)
