"""Descending-neuron (DN) readouts -> the walking controller's descending drive (pure, no flygym import).

The brain model and the NeuroMechFly v2 body are published; THIS MAPPING IS OURS (thresholds, scales, smoothing and
the escape rule below are our choices, not a published DN -> motor map). Inputs are a recording's named readout traces
(mean Hz of a neuron group per time bin, as sim_service.ActivityRecord.readouts); output is one drive per bin:

* forward  (0..1) from DNg100: 0 below FORWARD_THRESHOLD_HZ, linear up to 1 at FORWARD_SATURATION_HZ.
* turn     (-1..1) from LEFT minus RIGHT within each steering cell type. DNa01 and DNa02 are two cell types with ONE
  neuron on EACH side of the brain (FlyWire annotations v783), so the readouts are per side: dna02_left, dna02_right,
  dna01_left, dna01_right. turn = (DNA02_GAIN*(l02 - r02) + DNA01_GAIN*(l01 - r01)) / TURN_SATURATION_HZ, clamped.
  POSITIVE = turn LEFT (counter-clockwise seen from above, +yaw). Direction: Rayshubskiy et al. (eLife 102230) report
  that DNa01/DNa02 activity predicts IPSILATERAL turning (stronger left neuron -> turn left); DNa02 is the transient
  high-gain steering neuron, DNa01 the slower low-gain one. The 1.0 / 0.4 weights are OUR choice, not published values.
* escape   from DNp01: strength 0..1 between ESCAPE_THRESHOLD_HZ and ESCAPE_SATURATION_HZ; the bin is "escape" when the
  strength is above 0. Escape takes precedence: the legs stop and the viewer raises the wings (no takeoff model).
* drive    the flygym HybridTurningController descending signal [left, right]. That controller uses |drive| as the
  CPG amplitude of each side (a smaller amplitude on one side turns the fly towards that side; verified in
  flygym_demo/complex_terrain/turning_controller.py and by the gated integration test). Walking drive lies in
  [DRIVE_MIN, DRIVE_MAX] (the range exercised in our spike runs); stopped bins are exactly [0, 0].

Readouts are matched by preset name, case-insensitively. A missing readout contributes zero. Rates are smoothed with a
causal exponential low-pass over bins (time constant SMOOTH_TAU_MS) before the mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FORWARD_THRESHOLD_HZ = 5.0
FORWARD_SATURATION_HZ = 40.0
TURN_SATURATION_HZ = 30.0
ESCAPE_THRESHOLD_HZ = 10.0
ESCAPE_SATURATION_HZ = 40.0
SMOOTH_TAU_MS = 60.0
DRIVE_MIN = 0.4  # lowest walking amplitude per side
DRIVE_MAX = 1.2  # highest (flygym's own demos use 1.0 for straight walking)
BASE_MAX = 1.0  # both sides at full forward
TURN_GAIN = 0.5  # fraction of the base amplitude removed from the inner side / added to the outer side at turn = 1
TURN_ONLY_FORWARD = 0.5  # a turn command with no DNg100 still walks, at this forward level

DNA02_GAIN = 1.0  # high-gain steering neuron (our weight)
DNA01_GAIN = 0.4  # low-gain steering neuron (our weight)

FORWARD_NAME, ESCAPE_NAME = "dng100", "dnp01"
TURN_NAMES = (("dna02_left", "dna02_right", DNA02_GAIN), ("dna01_left", "dna01_right", DNA01_GAIN))
READOUT_NAMES = (FORWARD_NAME, "dna01_left", "dna01_right", "dna02_left", "dna02_right", ESCAPE_NAME)

CAPTION = (
    "Body drive: our own mapping of descending-neuron rates (DNg100 forward, left-minus-right DNa02/DNa01 turn, DNp01 stop and raise "
    "wings) onto NeuroMechFly's walking controller; the mapping is not published and the body is not the connectome."
)


@dataclass(frozen=True)
class BodyDrive:
    bin_ms: float
    forward: np.ndarray
    turn: np.ndarray
    escape_strength: np.ndarray
    escape: np.ndarray
    drive: np.ndarray  # (n_bins, 2) [left, right]
    caption: str = CAPTION

    @property
    def n_bins(self) -> int:
        return int(len(self.forward))

    def as_dict(self) -> dict:
        r = lambda a: [round(float(v), 4) for v in a]  # noqa: E731
        return {"bin_ms": self.bin_ms, "forward": r(self.forward), "turn": r(self.turn),
                "escape_strength": r(self.escape_strength), "escape": [bool(v) for v in self.escape],
                "drive": [r(row) for row in self.drive], "caption": self.caption}


def _trace(readouts, name: str) -> np.ndarray | None:
    for r in readouts or []:
        n = r.get("name")
        if isinstance(n, str) and n.lower() == name:
            return np.nan_to_num(np.asarray(r.get("hz") or [], dtype=float), nan=0.0, posinf=1e12, neginf=0.0)
    return None


def _smooth(x: np.ndarray, bin_ms: float, tau_ms: float) -> np.ndarray:
    if tau_ms <= 0 or len(x) == 0:
        return x
    a = 1.0 - math.exp(-bin_ms / tau_ms)
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += a * (v - acc)
        out[i] = acc
    return out


def _ramp(hz: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((hz - lo) / (hi - lo), 0.0, 1.0)


def _map_rates(rate, n: int):
    """Smoothed rates -> (forward, turn, escape_strength, escape, drive). `rate(name)` returns an (n,) array.

    The one place the mapping lives: the batch (`drive_from_readouts`) and the live (`DriveMapper`) paths share it."""
    forward = _ramp(rate(FORWARD_NAME), FORWARD_THRESHOLD_HZ, FORWARD_SATURATION_HZ)
    steer = sum(g * (rate(left) - rate(right)) for left, right, g in TURN_NAMES)
    turn = np.clip(steer / TURN_SATURATION_HZ, -1.0, 1.0)
    escape_strength = _ramp(rate(ESCAPE_NAME), ESCAPE_THRESHOLD_HZ, ESCAPE_SATURATION_HZ)
    escape = escape_strength > 0

    pace = np.maximum(forward, np.abs(turn) * TURN_ONLY_FORWARD)
    base = np.where(pace > 0, DRIVE_MIN + (BASE_MAX - DRIVE_MIN) * pace, 0.0)
    # lower LEFT drive turns left: left = base * (1 - gain*turn), right = base * (1 + gain*turn)
    left = base * (1.0 - TURN_GAIN * turn)
    right = base * (1.0 + TURN_GAIN * turn)
    drive = np.stack([left, right], axis=-1) if n else np.zeros((0, 2))
    walking = (pace > 0) & ~escape
    drive = np.where(walking[:, None], np.clip(drive, DRIVE_MIN, DRIVE_MAX), 0.0)
    return forward, turn, escape_strength, escape, drive


def drive_from_readouts(readouts, bin_ms: float, smooth_tau_ms: float = SMOOTH_TAU_MS) -> BodyDrive:
    """Per-bin body drive from a list of {"name", "hz": [...]} readouts (extra keys ignored)."""
    if not bin_ms or not math.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError("bin_ms must be positive")
    found = {n: _trace(readouts, n) for n in READOUT_NAMES}
    n = max((len(t) for t in found.values() if t is not None), default=0)

    def rate(name: str) -> np.ndarray:
        t = found[name]
        out = np.zeros(n)
        if t is not None:
            out[: len(t)] = _smooth(t, bin_ms, smooth_tau_ms)
        return out

    forward, turn, escape_strength, escape, drive = _map_rates(rate, n)
    return BodyDrive(float(bin_ms), forward, turn, escape_strength, escape, drive)


@dataclass(frozen=True)
class DriveSample:
    """The mapping of ONE chunk (live session): scalars, drive is [left, right]."""

    forward: float
    turn: float
    escape_strength: float
    escape: bool
    drive: np.ndarray

    def as_dict(self) -> dict:
        return {"forward": round(self.forward, 4), "turn": round(self.turn, 4),
                "escape_strength": round(self.escape_strength, 4), "escape": self.escape,
                "drive": [round(float(v), 4) for v in self.drive]}


class DriveMapper:
    """Streaming `drive_from_readouts`: feed the DN rates of each chunk (Hz per readout name, missing = 0) and get
    that chunk's drive. The low-pass state is carried between calls, so chunk i gives exactly bin i of the batch
    mapping with `bin_ms = chunk_ms` (tested). `reset()` forgets the state (a new session)."""

    def __init__(self, chunk_ms: float, smooth_tau_ms: float = SMOOTH_TAU_MS) -> None:
        if not chunk_ms or not math.isfinite(chunk_ms) or chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        self.chunk_ms, self.tau = float(chunk_ms), smooth_tau_ms
        self._acc: dict[str, float] = {}

    def reset(self) -> None:
        self._acc.clear()

    def step(self, rates_hz: dict) -> DriveSample:
        a = 1.0 if self.tau <= 0 else 1.0 - math.exp(-self.chunk_ms / self.tau)
        smoothed = {}
        for name in READOUT_NAMES:
            raw = float(np.nan_to_num(rates_hz.get(name, 0.0) or 0.0, nan=0.0, posinf=1e12, neginf=0.0))
            acc = self._acc.get(name, 0.0)
            acc += a * (raw - acc)
            self._acc[name] = smoothed[name] = acc
        forward, turn, strength, escape, drive = _map_rates(lambda name: np.array([smoothed[name]]), 1)
        return DriveSample(float(forward[0]), float(turn[0]), float(strength[0]), bool(escape[0]), drive[0])
