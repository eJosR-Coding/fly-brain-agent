"""Surface walking in the glass cube: a labelled BODY/WORLD RULE (adhesive pads), not a brain decision (pure numpy).

The fly walks on ONE current surface (floor, x_min, x_max, y_min, y_max, ceiling) of the cube. Each surface has an inward
normal n; the fly's dorsal axis is n, its heading f is a unit vector tangent to the surface and its left is n x f, so the
gait clips (forward / turn / lateral) run unchanged in the surface's local frame.

TRANSITION RULE (convex edge between the current surface and a perpendicular one, d = the outward direction through the edge):
the fly's root reaching the edge is carried onto the new surface by a rigid 90 degree rotation about the edge. The in-plane
heading keeps its tangential component along the edge and its into-wall component (f . d) becomes the direction away from
the old surface (the old normal): head-on -> straight up the wall, oblique -> diagonal; wall -> ceiling at the top edge
(upside down), wall -> adjacent wall at a vertical corner, ceiling -> wall, wall -> floor going down. The overshoot of the step
continues along the new surface. The REPORTED orientation pitches onto the new plane over BLEND_S (slerp); the root is the
foot point on the surface plus the dorsal axis x the thorax height, so the thorax never leaves the cube volume.
BODY REACH: the foot point stays BODY_MARGIN_MM inside every other glass plane (the head tip and legs reach about that far
from it): an edge counts as reached when the foot is that close, and the fly then stands BODY_MARGIN_MM above the old surface
on the new one, so no body part crosses the glass.
"""

from __future__ import annotations

import math

import numpy as np

BODY_MARGIN_MM = 2.4  # OUR labelled reach of the body (head tip / legs) around the foot point: contact and transitions use it, so no body part crosses the glass
BLEND_S = 0.1  # the pitch onto the new surface is eased over this time (ours)
FLOOR = "floor"


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    t = r[0, 0] + r[1, 1] + r[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def matrix_from_quat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + (b - a) * u
        return q / np.linalg.norm(q)
    th = math.acos(min(d, 1.0))
    return (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)


def rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SurfaceWalker:
    """Foot point q (mm, arena frame: x forward at the start, y left, z up from the floor), surface and heading."""

    def __init__(self, bounds: tuple[float, float, float, float], height_mm: float) -> None:
        xmin, xmax, ymin, ymax = bounds
        self.height_mm = float(height_mm)
        # name -> (inward normal, c): the plane is n . p = c and the cube interior is n . p >= c
        self.surfaces = {
            FLOOR: (np.array([0.0, 0.0, 1.0]), 0.0), "ceiling": (np.array([0.0, 0.0, -1.0]), -float(height_mm)),
            "x_max": (np.array([-1.0, 0.0, 0.0]), -float(xmax)), "x_min": (np.array([1.0, 0.0, 0.0]), float(xmin)),
            "y_max": (np.array([0.0, -1.0, 0.0]), -float(ymax)), "y_min": (np.array([0.0, 1.0, 0.0]), float(ymin)),
        }
        self.place_floor(0.0, 0.0, 0.0)

    def place_floor(self, x: float, y: float, yaw: float) -> None:
        self.surface = FLOOR
        self.q = np.array([x, y, 0.0])
        self.f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        self._yaw = float(yaw)
        self._r_last = self._target()
        self._shift = np.zeros(3)  # rendered foot minus true foot: eases the foot across an edge with the pitch
        self._q_last = self.q.copy()
        self._r_from, self._u = self._r_last, 1.0
        self.events: list[dict] = []

    @property
    def n(self) -> np.ndarray:
        return self.surfaces[self.surface][0]

    def _target(self) -> np.ndarray:
        n = self.n
        return np.column_stack([self.f, np.cross(n, self.f), n])

    @property
    def yaw(self) -> float:
        """Heading projected on the floor plane (the floor yaw; held when the heading is vertical)."""
        if math.hypot(self.f[0], self.f[1]) > 1e-6:
            self._yaw = math.atan2(self.f[1], self.f[0])
        return self._yaw

    def _rotate_heading(self, ang: float) -> None:
        n = self.n
        self.f = self.f * math.cos(ang) + np.cross(n, self.f) * math.sin(ang)
        self.f /= np.linalg.norm(self.f)

    def step(self, forward_mm: float, lateral_mm: float, dyaw: float) -> None:
        """Walk one frame on the current surface (heading turns by dyaw about the dorsal axis); transitions at edges."""
        self._rotate_heading(dyaw * 0.5)
        self.q = self.q + forward_mm * self.f + lateral_mm * np.cross(self.n, self.f)
        self._rotate_heading(dyaw * 0.5)
        for _ in range(3):
            n_old = self.n
            worst, name = 1e-9, None
            for nm, (nn, c) in self.surfaces.items():
                if abs(float(nn @ n_old)) > 0.5:
                    continue
                e = c + BODY_MARGIN_MM - float(nn @ self.q)  # >0: the body reach touches this glass plane
                if e > worst:
                    worst, name = e, nm
            if name is None:
                return
            nn = self.surfaces[name][0]
            d = -nn
            rel = math.degrees(math.atan2(float(np.cross(self.f, d) @ n_old), float(self.f @ d)))  # edge direction from the heading, left +

            def rot(v):
                return v - float(v @ d) * d - float(v @ n_old) * n_old + float(v @ d) * n_old - float(v @ n_old) * d

            self.f = rot(self.f)
            self.f /= np.linalg.norm(self.f)
            a = float(nn @ self.q) - self.surfaces[name][1]  # distance of the foot from the glass plane (< 0: past it)
            q_before = self._q_last
            self.q = self.q - a * nn + (BODY_MARGIN_MM + max(0.0, -a)) * n_old
            self.events.append({"from": self.surface, "to": name, "contact_rel_deg": round(rel, 1)})
            self.surface = name
            self._r_from, self._u = self._r_last, 0.0
            self._shift = q_before - self.q  # the rendered foot starts where it was and eases onto the new surface

    def render(self, h_mm: float, target: np.ndarray | None = None, free_pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(thorax position mm, quaternion) of this frame, the orientation eased toward the surface frame (or `target`)."""
        tgt = self._target() if target is None else target
        self._u = min(1.0, self._u + 1.0 / (BLEND_S * 200.0))
        if self._u >= 1.0:
            r = tgt
        else:
            r = matrix_from_quat(_slerp(quat_from_matrix(self._r_from), quat_from_matrix(tgt), _smooth(self._u)))
        self._r_last = r
        s = _smooth(self._u)
        self._q_last = self.q + self._shift * (1.0 - s)
        pos = free_pos if free_pos is not None else self._q_last + r[:, 2] * h_mm
        return pos, quat_from_matrix(r)

    def start_free(self) -> None:
        """Flight starts: the orientation eases from the current surface frame to upright."""
        self._r_from, self._u = self._r_last, 0.0
