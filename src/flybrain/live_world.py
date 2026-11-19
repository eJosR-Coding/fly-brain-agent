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

    def clear(self) -> int:
        n = len(self.objects)
        self.objects = []
        return n

    def touching(self, pos_um, yaw_deg) -> list[str]:
        """Ids of the cubes touched by the fly's mouth point at the LAST frame of a chunk."""
        if not self.objects:
            return []
        x, y = float(pos_um[-1][0]) / 1000.0, float(pos_um[-1][1]) / 1000.0
        yaw = math.radians(float(yaw_deg[-1]))
        hx, hy = x + HEAD_AHEAD_MM * math.cos(yaw), y + HEAD_AHEAD_MM * math.sin(yaw)
        out = []
        for o in self.objects:
            h = o["size_mm"] / 2
            dx, dy = max(abs(hx - o["pos_mm"][0]) - h, 0.0), max(abs(hy - o["pos_mm"][1]) - h, 0.0)
            if math.hypot(dx, dy) <= TOUCH_MM:
                out.append(o["id"])
        return out

    def taste(self, touched: list[str], rows: dict[str, np.ndarray]) -> dict[str, float]:
        """{preset name: Hz} for the touched cubes (rows only gates presets that exist in this atlas)."""
        hz: dict[str, float] = {}
        for o in self.objects:
            preset = ITEMS[o["kind"]]
            if o["id"] in touched and preset in rows:
                hz[preset] = TASTE_HZ
        return hz
