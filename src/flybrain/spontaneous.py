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

STATES = ("rest", "walk", "turn_left", "turn_right", "groom", "takeoff", "seek", "feed", "avoid")
MODES = ("ethogram", "off")

# mean bout duration (s) and the (min, max) clamp of an exponential draw: OUR numbers
DURATIONS_S = {"rest": (2.0, 0.8, 5.0), "walk": (3.0, 1.0, 8.0), "turn_left": (0.7, 0.4, 1.5), "turn_right": (0.7, 0.4, 1.5),
               "groom": (3.0, 1.5, 6.0)}
# P(next | current): OUR numbers. Takeoff is raised so flights are frequent (about one per 10-20 s on average, not validated); it is followed by rest once the fly has landed.
TRANSITIONS = {
    "rest": {"walk": 0.38, "groom": 0.20, "turn_left": 0.06, "turn_right": 0.06, "takeoff": 0.30},
    "walk": {"rest": 0.25, "turn_left": 0.14, "turn_right": 0.14, "groom": 0.10, "takeoff": 0.37},
    "turn_left": {"walk": 0.45, "rest": 0.15, "groom": 0.05, "takeoff": 0.35},
    "turn_right": {"walk": 0.45, "rest": 0.15, "groom": 0.05, "takeoff": 0.35},
    "groom": {"rest": 0.35, "walk": 0.30, "turn_left": 0.05, "turn_right": 0.05, "takeoff": 0.25},
    "takeoff": {"rest": 1.0},
}
# Poisson rate (Hz) injected into each commanded group while a state lasts; calibrated against the brain's DN response
# (scratch calibration: a directly driven DN fires about at its input rate, so 40 Hz on DNg100 gives forward ~0.7 and 40 Hz on one DNa02 gives turn ~0.65 without recruiting other DNs). OUR constants.
INJECT_HZ = {"walk": {"dng100": 40.0}, "turn": {"dng100": 40.0, "dna02": 40.0}, "groom": {"groom": 20.0}, "takeoff": {"dnp01": 60.0}}
TAKEOFF_PULSE_MS = 150.0  # the giant-fibre drive lasts this long (the live GF rule needs DNp01 in 2 of 3 chunks of 50 ms)
CALIBRATION_NOTE = "injection rates were chosen so the commanded neurons answer in the model; see `response` for what they did"
GROOM_ON_HZ = 8.0  # mean Hz of the grooming DNs that makes the body groom (ours); off below GROOM_OFF_HZ (hysteresis)
GROOM_OFF_HZ = 3.0
# Food / bitter rules (OURS, labelled, not validated). The brain-only model does not lateralize odour, so seeking is a rule here:
# it only injects DNg100 + a unilateral DNa02 proportional to the bearing error; the body is never moved by it.
SMELL_RADIUS_MM = 40.0  # a sugar cube closer than this "is smelled" (proxy: no odour model)
BITTER_RADIUS_MM = 10.0  # coffee closer than this is avoided (bitter avoidance is real in flies; radius and gains are ours)
FLOOR_Z_UM = 1600.0  # fallback floor test when the body gives no orientation (physics body): thorax this low = on the floor
TURN_GAIN = 60.0 / 40.0  # Hz of DNa02 per degree of bearing error (saturates at SEEK_TURN_MAX_HZ at 40 deg); dead zone below
SEEK_TURN_MAX_HZ = 60.0  # ours: a stronger unilateral DNa02 drive for seeking (the body turn saturates at 1)
SEEK_WALK_DEG = 25.0  # seek walks (DNg100) only when the bearing error is below this; otherwise it turns first (ours)
TURN_DEADZONE_DEG = 6.0
FEED_S = (5.0, 15.0)  # stand still this long after sugar contact (uniform draw)
BITTER_PAUSE_S = 1.0  # stop this long after coffee contact, then walk away
COOLDOWN_S = 60.0  # a fed-from cube is ignored this long (satiety); a cube that was avoided likewise after contact
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

FOOD_NOTE = (f"Seek / feed / avoid are OUR labelled rules (not the connectome, not validated): a sugar cube within {SMELL_RADIUS_MM:g} mm (smell proxy) makes the drive inject "
             f"DNg100 + a unilateral DNa02 proportional to the bearing error; on contact it stands still {FEED_S[0]:g}-{FEED_S[1]:g} s, then ignores that cube {COOLDOWN_S:g} s; "
             f"coffee within {BITTER_RADIUS_MM:g} mm turns the drive away (DNa02 on the far side); contact stops it {BITTER_PAUSE_S:g} s first. The body is moved only by the brain's DNs.")
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
        self._target: dict | None = None
        self._ignore: dict[str, float] = {}  # object id -> t_ms until which it is ignored
        self._feed_ms = 0.0
        self._feed_kind = "sugar"
        self.food: dict = {"state": "none", "target": None, "bearing_deg": None, "distance_mm": None, "note": FOOD_NOTE}
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
        elif state in ("seek", "avoid"):
            self.left_ms = 1e12  # held by the rule while its object is in range
        elif state == "feed":
            self.left_ms = self._feed_ms
        else:
            self.left_ms = self._draw(state)

    @staticmethod
    def _surface_dir(ctx: dict) -> tuple | None:
        """Off the floor: the world direction the fly should walk to get back down (ours): straight down on a wall, toward the nearest wall on the
        ceiling. Returns (bearing deg + left in the surface frame, label) or None when the pose has no orientation or the fly is on the floor."""
        n, f = ctx.get("n"), ctx.get("f")
        if n is None or f is None or n[2] > 0.5:
            return None
        if n[2] < -0.5:  # ceiling: head to the nearest wall (outward direction of that wall, tangent to the ceiling)
            b = ctx.get("bounds_mm")
            if not b:
                return None
            x, y = ctx["x_mm"], ctx["y_mm"]
            cands = [(x - b[0], (-1.0, 0.0, 0.0)), (b[1] - x, (1.0, 0.0, 0.0)), (y - b[2], (0.0, -1.0, 0.0)), (b[3] - y, (0.0, 1.0, 0.0))]
            w, label = min(cands)[1], "ceiling"
        else:
            w, label = (0.0, 0.0, -1.0), "wall"
        n3, f3, w3 = np.asarray(n, float), np.asarray(f, float), np.asarray(w, float)
        p = w3 - float(w3 @ n3) * n3
        if np.linalg.norm(p) < 1e-6:
            return 0.0, label
        left = np.cross(n3, f3)
        return math.degrees(math.atan2(float(left @ p), float(f3 @ p))), label

    def _rule(self, t_ms: float, ctx: dict) -> tuple[str | None, dict | None, float, float]:
        """(wanted state, target, bearing deg + left, distance mm) from the pose and objects, or (None, ...) when no rule applies."""
        x, y, yaw = ctx["x_mm"], ctx["y_mm"], ctx["yaw_deg"]
        off = self._surface_dir(ctx)
        if off is not None:  # on a wall / the ceiling: any sugar within smell range (3D) sends the fly down; coffee is ignored here
            z = ctx["z_um"] / 1000.0
            near = [(math.dist((o["pos_mm"][0], o["pos_mm"][1], o["pos_mm"][2]), (x, y, z)), o) for o in ctx["objects"]
                    if o.get("kind") == "sugar" and self._ignore.get(o["id"], -1.0) <= t_ms]
            near = [q for q in near if q[0] <= SMELL_RADIUS_MM]
            if not near:
                return None, None, 0.0, 0.0
            d, o = min(near, key=lambda q: q[0])
            return "seek", o, off[0], d
        best = None
        for o in ctx["objects"]:
            if o.get("kind") not in ("sugar", "coffee") or self._ignore.get(o["id"], -1.0) > t_ms:
                continue
            d = math.hypot(o["pos_mm"][0] - x, o["pos_mm"][1] - y)
            rad = BITTER_RADIUS_MM if o["kind"] == "coffee" else SMELL_RADIUS_MM
            if d <= rad and (best is None or (o["kind"] == "coffee", -d) > (best[0]["kind"] == "coffee", -best[2])):
                best = (o, math.degrees(math.atan2(o["pos_mm"][1] - y, o["pos_mm"][0] - x)) - yaw, d)
        if best is None:
            return None, None, 0.0, 0.0
        o, b, d = best
        flip = 180.0 if o["kind"] == "coffee" else 0.0  # avoid: the wanted heading is directly away from the cube
        return ("avoid" if o["kind"] == "coffee" else "seek"), o, (b + flip + 180.0) % 360.0 - 180.0, d

    def step(self, t_ms: float, *, airborne: bool = False, ctx: dict | None = None) -> dict:
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
        elif self.state == "feed":
            self.left_ms -= dt
            if self.left_ms <= 0:
                self._enter(t_ms, "rest")  # ethogram resumes; the cube is ignored for COOLDOWN_S (set at contact)
        else:
            self.left_ms -= dt
            if self.left_ms <= 0 and self.state not in ("seek", "avoid"):
                self._enter(t_ms, self._next(self.state))
        bearing = dist = None
        if ctx is not None and self.state != "takeoff":
            for oid in ctx.get("touched", ()):
                o = next((q for q in ctx["objects"] if q["id"] == oid), None)
                if o and o["kind"] in ("sugar", "coffee") and self._ignore.get(oid, -1.0) <= t_ms and self.state != "feed":
                    self._ignore[oid] = t_ms + (COOLDOWN_S if o["kind"] == "sugar" else BITTER_PAUSE_S + 3.0) * 1000.0
                    self._target = o
                    if o["kind"] == "sugar":
                        self._feed_ms = self._rng.uniform(*FEED_S) * 1000.0
                    else:
                        self._feed_ms = BITTER_PAUSE_S * 1000.0
                    self._feed_kind = o["kind"]
                    self._enter(t_ms, "feed")
            if self.state != "feed":
                on_floor = (ctx["n"][2] > 0.5) if ctx.get("n") is not None else ctx["z_um"] < FLOOR_Z_UM
                want, tgt, bearing, dist = self._rule(t_ms, ctx) if (on_floor or self._surface_dir(ctx) is not None) else (None, None, 0.0, 0.0)
                if want and self.state != want:
                    self._enter(t_ms, want)
                elif not want and self.state in ("seek", "avoid"):
                    self._enter(t_ms, "walk")
                self._target = tgt if want else self._target
        s = self.state
        rates: dict[str, float] = {}
        if s in ("seek", "avoid") and bearing is not None:
            lim = 90.0 if s == "avoid" else SEEK_WALK_DEG  # avoid: turn first, walk once facing away; seek: walk once facing the cube
            rates = {} if abs(bearing) > lim else {"dng100": INJECT_HZ["walk"]["dng100"]}
            if abs(bearing) > TURN_DEADZONE_DEG:
                rates["dna02_left" if bearing > 0 else "dna02_right"] = min(abs(bearing) * TURN_GAIN, SEEK_TURN_MAX_HZ)
        elif s == "feed":
            rates = {}
        elif s == "walk":
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
        self.food = {"state": s if s in ("seek", "feed", "avoid") else "none", "target": self._target["id"] if self._target and s in ("seek", "feed", "avoid") else None,
                     "bearing_deg": None if bearing is None else round(bearing, 1), "distance_mm": None if dist is None else round(dist, 2), "note": FOOD_NOTE}
        return {"rates": rates, "state": s, "head_deg": head}

    def flight_target(self, ctx: dict) -> dict | None:
        """In flight with sugar present: the nearest sugar cube (xy mm) the flight is steered to and lands beside (OUR labelled rule), else None."""
        if self.mode == "off" or ctx is None:
            return None
        t_ms = self.log[-1][0] if self.log else 0.0
        best = None
        for o in ctx["objects"]:
            if o.get("kind") != "sugar" or self._ignore.get(o["id"], -1.0) > ctx.get("t_ms", t_ms):
                continue
            d = math.hypot(o["pos_mm"][0] - ctx["x_mm"], o["pos_mm"][1] - ctx["y_mm"])
            if d <= SMELL_RADIUS_MM and (best is None or d < best[0]):
                best = (d, o)
        return None if best is None else {"id": best[1]["id"], "x_mm": best[1]["pos_mm"][0], "y_mm": best[1]["pos_mm"][1], "dist_mm": best[0]}

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
