"""Grooming and head-saccade kinematics for the kinematic body (pure numpy). OURS, labelled, not validated.

flygym 2.1.1 (the installed package) ships no grooming kinematics (searched its package data: no grooming clip, no
grooming controller), so the front-leg rubbing below is a keyframed illustration: each front leg is rotated RIGIDLY about
its coxa so the leg tip rises in front of the head and rubs up and down (GROOM_HZ, alternating legs). The head sweeps
(yaw) at the same time. Head saccades: small fast head yaw steps; the head also leads the turn command.
"""

from __future__ import annotations

import math

import numpy as np

GROOM_HZ = 4.0  # rubbing frequency in SIMULATED time (ours; the real one is higher)
GROOM_RAMP_S = 0.15
GROOM_TIP_UM = (1000.0, 130.0, 950.0)  # leg-tip target in the thorax frame (x anterior, |y| lateral, z up): in front of the head (ours)
GROOM_RUB_UM = 220.0  # vertical rubbing amplitude of the target
GROOM_HEAD_DEG = 10.0  # head sweep while grooming
GROOM_HEAD_HZ = 1.5
HEAD_TAU_S = 0.03  # head yaw is fast: first-order, 30 ms
HEAD_TURN_GAIN_DEG = 14.0  # the head leads the turn command by this at turn = 1 (ours)
HEAD_NOTE = "head yaw (ours, not validated): small fast saccades at random intervals, leading the turn command, sweeping while grooming; real flies make head saccades and counter-rotate the head in turns"
GROOM_NOTE = ("grooming (ours, not validated): flygym 2.1.1 has no grooming kinematics, so the front legs are rotated rigidly about their coxae "
              f"to rub in front of the head at {GROOM_HZ:g} Hz while the head sweeps; played while the grooming DNs (DNg10/11/12) are active")


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    n = np.linalg.norm(axis)
    if n < 1e-9 or abs(angle) < 1e-9:
        return np.eye(3)
    k = axis / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, (w, x, y, z); broadcasts over the leading axes."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def groom_pose(pos: np.ndarray, quat: np.ndarray, legs: dict[str, list[int]], coxae: dict[str, int], g: float, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Front legs rotated rigidly about their coxae by fraction `g` (0..1) of the way to the rubbing target. `legs` maps 'lf'/'rf'
    to the 8 body indices of that leg, `coxae` to the coxa body index. Returns new (pos, quat); inputs untouched."""
    if g <= 0.0:
        return pos, quat
    pos, quat = pos.copy(), quat.copy()
    for side, sgn, phase in (("lf", 1.0, 0.0), ("rf", -1.0, math.pi)):
        idx = legs[side]
        c = pos[coxae[side]].copy()
        tip = pos[idx[-1]] - c
        tx, ty, tz = GROOM_TIP_UM
        want = np.array([tx, sgn * ty, tz + GROOM_RUB_UM * math.sin(2 * math.pi * GROOM_HZ * t_s + phase)]) - c
        u, d = tip / np.linalg.norm(tip), want / np.linalg.norm(want)
        axis = np.cross(u, d)
        ang = math.acos(max(-1.0, min(1.0, float(u @ d)))) * g
        R = _rot(axis, ang)
        pos[idx] = c + (pos[idx] - c) @ R.T
        n = np.linalg.norm(axis)
        if n > 1e-9:
            h = ang / 2.0
            qr = np.array([math.cos(h), *(axis / n * math.sin(h))])
            quat[idx] = _qmul(qr, quat[idx])
    return pos, quat


class Head:
    """Head yaw (deg, + left): first-order follow of a target made of the saccade command, the turn lead and the grooming sweep."""

    def __init__(self) -> None:
        self.deg = 0.0
        self.saccade = 0.0
        self.turn = 0.0
        self.grooming = False

    def command(self, saccade_deg: float, turn: float, grooming: bool) -> None:
        self.saccade, self.turn, self.grooming = float(saccade_deg), float(turn), bool(grooming)

    def step(self, dt_s: float, t_s: float, flying: bool) -> float:
        if flying:
            tgt = 0.0
        else:
            tgt = self.saccade + HEAD_TURN_GAIN_DEG * max(-1.0, min(1.0, self.turn))
            if self.grooming:
                tgt += GROOM_HEAD_DEG * math.sin(2 * math.pi * GROOM_HEAD_HZ * t_s)
        self.deg += (tgt - self.deg) * (1.0 - math.exp(-dt_s / HEAD_TAU_S))
        return self.deg
