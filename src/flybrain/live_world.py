"""Playground world: sugar / coffee cubes in the live glass cube and the contact rule that feeds the taste neurons.

Everything here is OUR choice, labelled as such (not published, not validated):
* the fly's mouth point is the thorax plus HEAD_AHEAD_MM along its heading (a fixed proxy for proboscis / front legs);
* "touch" = that point within TOUCH_MM of the cube's surface;
* a touched cube drives its taste neurons at a constant TASTE_HZ (Poisson): sugar -> the sweet GRNs (preset "sugar"),
  coffee -> the bitter GRNs (preset "bitter"), a PROXY for caffeine bitterness.
What the fly then does is whatever the brain does; nothing is scripted.
"""

from __future__ import annotations

import math

import numpy as np

ITEMS = {"sugar": "sugar", "coffee": "bitter"}  # item -> preset of the taste neurons it excites on touch
HEAD_AHEAD_MM = 0.8
TOUCH_MM = 1.0
TASTE_HZ = 100.0
CUBE_MM = 1.5
MAX_OBJECTS = 16
MOUTH_MAX_Z_MM = CUBE_MM + TOUCH_MM + 1.0  # contact needs the thorax this low (cubes sit on the floor); 2D contact on a wall was a false feed
# "Lanzale algo" / "asustala": a dark sphere thrown at the fly's head on a straight line (OUR choice, not validated).
# It reproduces the eye calibration's looming disc: r / v = 1 mm / 25 mm/s = 40 ms, starting 20 mm away (half-angle 2.9 deg),
# half-angle growing as asin(r / d) up to 75 deg after 800 ms; there it stops (no collision physics) and is held THREAT_HOLD_MS
# (the calibration held its disc at 75 deg too) before it vanishes. Azimuth +-65 deg = the eye axis (left / right) or 0 (front).
THREAT_R_MM = 1.0
THREAT_V_MM_S = 25.0
THREAT_START_MM = 20.0
THREAT_STOP_HALF_ANGLE_DEG = 75.0
THREAT_AZ_DEG = {"left": 65.0, "right": -65.0, "front": 0.0}
THREAT_HOLD_MS = 300.0
NOTE = ("world rule (ours, not validated): touching a cube (mouth point within 1 mm of its surface) drives its taste neurons "
        f"at {TASTE_HZ:g} Hz; coffee drives the bitter neurons as a caffeine proxy; the fly's response is the brain's")


class World:
    def __init__(self, bounds_mm=None) -> None:
        self.bounds = bounds_mm
        self.objects: list[dict] = []
        self._n = 0
        self.used = False  # once a cube was added, every chunk reports the object list (so a clear reaches the viewer)

    def add(self, item: str, x_mm: float, y_mm: float) -> dict:
        key = str(item).strip().lower()
        if key not in ITEMS:
            raise ValueError(f"item must be one of {', '.join(ITEMS)}")
        if len(self.objects) >= MAX_OBJECTS:
            raise ValueError(f"at most {MAX_OBJECTS} objects; clear the world first")
        if self.bounds:
            m = CUBE_MM / 2
            x_mm = min(max(x_mm, self.bounds[0] + m), self.bounds[1] - m)
            y_mm = min(max(y_mm, self.bounds[2] + m), self.bounds[3] - m)
        self._n += 1
        self.used = True
        obj = {"id": f"{key}{self._n}", "kind": key, "pos_mm": [round(x_mm, 3), round(y_mm, 3), CUBE_MM / 2], "size_mm": CUBE_MM}
        self.objects.append(obj)
        return obj

    def add_threat(self, side: str, head_mm, yaw_deg: float, t_ms: float) -> dict:
        """A dark sphere starting THREAT_START_MM from the head point `head_mm` (x, y, z) at azimuth `side` from the heading,
        flying straight at that point. Shortened (same speed) so it starts inside the glass."""
        key = str(side).strip().lower()
        if key not in THREAT_AZ_DEG:
            raise ValueError(f"side must be one of {', '.join(THREAT_AZ_DEG)}")
        if len(self.objects) >= MAX_OBJECTS:
            raise ValueError(f"at most {MAX_OBJECTS} objects; clear the world first")
        az = math.radians(yaw_deg + THREAT_AZ_DEG[key])
        u = np.array([math.cos(az), math.sin(az)])
        dist = THREAT_START_MM
        if self.bounds:
            m = THREAT_R_MM
            for axis, lo, hi in ((0, self.bounds[0], self.bounds[1]), (1, self.bounds[2], self.bounds[3])):
                if u[axis] > 1e-9:
                    dist = min(dist, (hi - m - head_mm[axis]) / u[axis])
                elif u[axis] < -1e-9:
                    dist = min(dist, (lo + m - head_mm[axis]) / u[axis])
        stop_d = THREAT_R_MM / math.sin(math.radians(THREAT_STOP_HALF_ANGLE_DEG))
        dist = max(dist, 2.0 * stop_d)
        p0 = (float(head_mm[0] + dist * u[0]), float(head_mm[1] + dist * u[1]), float(head_mm[2]))
        vel = (-THREAT_V_MM_S * u[0], -THREAT_V_MM_S * u[1], 0.0)
        self._n += 1
        self.used = True
        obj = {"id": f"threat{self._n}", "kind": "threat", "pos_mm": [round(p, 3) for p in p0], "size_mm": 2 * THREAT_R_MM,
               "p0_mm": [float(p) for p in p0], "vel_mm_s": [float(v) for v in vel], "t0_ms": float(t_ms),
               "stop_ms": (dist - stop_d) / THREAT_V_MM_S * 1000.0,
               "life_ms": (dist - stop_d) / THREAT_V_MM_S * 1000.0 + THREAT_HOLD_MS}
        self.objects.append(obj)
        return obj

    def spheres(self) -> list:
        from flybrain.fly_eye import MovingSphere

        return [MovingSphere(tuple(o["p0_mm"]), tuple(o["vel_mm_s"]), o["size_mm"] / 2.0, o["t0_ms"], o["life_ms"], o["stop_ms"])
                for o in self.objects if o["kind"] == "threat"]

    def advance(self, t_ms: float) -> None:
        """Move every threat to its position at `t_ms` (the end of the chunk) and drop the ones that are gone."""
        keep = []
        for o in self.objects:
            if o["kind"] == "threat":
                te = t_ms - o["t0_ms"]
                if te > o["life_ms"]:
                    continue
                te = min(te, o["stop_ms"])
                o["pos_mm"] = [round(p + v * te / 1000.0, 3) for p, v in zip(o["p0_mm"], o["vel_mm_s"])]
            keep.append(o)
        self.objects = keep

    def clear(self) -> int:
        n = len(self.objects)
        self.objects = []
        return n

    def touching(self, pos_um, yaw_deg) -> list[str]:
        """Ids of the cubes touched by the fly's mouth point at the LAST frame of a chunk."""
        if not self.objects:
            return []
        x, y = float(pos_um[-1][0]) / 1000.0, float(pos_um[-1][1]) / 1000.0
        if float(pos_um[-1][2]) / 1000.0 > MOUTH_MAX_Z_MM:  # on a wall / ceiling / in the air the mouth is far above the floor cubes
            return []
        yaw = math.radians(float(yaw_deg[-1]))
        hx, hy = x + HEAD_AHEAD_MM * math.cos(yaw), y + HEAD_AHEAD_MM * math.sin(yaw)
        out = []
        for o in self.objects:
            if o["kind"] not in ITEMS:
                continue
            h = o["size_mm"] / 2
            dx, dy = max(abs(hx - o["pos_mm"][0]) - h, 0.0), max(abs(hy - o["pos_mm"][1]) - h, 0.0)
            if math.hypot(dx, dy) <= TOUCH_MM:
                out.append(o["id"])
        return out

    def taste(self, touched: list[str], rows: dict[str, np.ndarray]) -> dict[str, float]:
        """{preset name: Hz} for the touched cubes (rows only gates presets that exist in this atlas)."""
        hz: dict[str, float] = {}
        for o in self.objects:
            preset = ITEMS.get(o["kind"])
            if preset and o["id"] in touched and preset in rows:
                hz[preset] = TASTE_HZ
        return hz
