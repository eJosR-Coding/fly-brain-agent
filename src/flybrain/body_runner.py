"""Run NeuroMechFly v2 (flygym 2.1.1) with a time-varying descending drive and export a fly-walk/1 clip.

flygym and MuJoCo are an OPTIONAL extra (`uv sync --extra body`); they are imported lazily, and BodyUnavailable is raised
when they are missing. The module is also the single source of the shared exporter logic (quaternion maths, the neutral
reference, the simulation rig) used by scripts/export_flygym_walk.py, which writes the committed looping walking asset.

Clip format: fly-walk/1 exactly as the walking asset (thorax-relative leg-body transforms, see that script), plus these
OPTIONAL fields written by run_body (the viewer ignores what it does not know):
  "loop": false, "duration_s": simulated seconds,
  "escape": per-frame 0/1 (the drive said "escape": legs stopped; the viewer raises the wings),
  "drive": per-frame [left, right] descending drive applied,
  "thorax_world": {"pos_um": per-frame [x, y, z] integers (x, y relative to the start; z absolute), "yaw_deg":
                   per-frame yaw of the thorax about the vertical axis (unwrapped, 0 at the start, POSITIVE = turning
                   LEFT / counter-clockwise from above)}  in flygym's world frame (x forward at the start, y left, z up).
"""

from __future__ import annotations

import math
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np

SAMPLE_HZ = 200
CPG_HZ = 12.0  # the controller's fixed CPG frequency
# Physics and CPG step (the controller integrates its oscillators with the same step). Measured against the dt = 1e-4 s
# reference (G2, odd/tasks/realtime.md; drive [1,1] 1.5 s and [0.4,1.2] 1.0 s, seed 0): at 2e-4 s the speed is -2.8 %,
# the turn yaw rate -4.9 %, stepping 12.0 Hz, tripod kept, joint-angle median error 0.2 deg, 1.97x faster. 3e-4 s also
# passes but does not divide the 200 Hz clip frame period; 5e-4 s bounces (speed +23 %, thorax up to 1.96 mm).
DEFAULT_DT_S = 2e-4
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LINKS = ("coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5")
BODIES = [f"{leg}_{link}" for leg in LEGS for link in LINKS]
POS_UM_PER_MM = 1000.0
QUAT_SCALE = 10000
INSTALL_HINT = "uv sync --extra body"


class BodyUnavailable(RuntimeError):
    """flygym / MuJoCo are not installed (the optional `body` extra)."""


def require_flygym() -> None:
    try:
        import flygym  # noqa: F401
        import flygym_demo.complex_terrain  # noqa: F401
        import mujoco  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised through BodyUnavailable tests with a patched import
        raise BodyUnavailable(
            f"the physics body needs flygym 2.1.1 (not installed: {e}); install it with `{INSTALL_HINT}`"
        ) from e


# ---- pure maths ----

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions, broadcasting over the leading axes."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (..., 3) by unit quaternions q (..., 4)."""
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def thorax_relative(pos_mm: np.ndarray, quat: np.ndarray, thorax: int):
    """World (pos in mm, quat wxyz) of all bodies -> thorax frame, positions in micrometres."""
    qt = quat[thorax] / np.linalg.norm(quat[thorax])
    inv = quat_conj(qt)
    rel_pos = quat_rotate(np.broadcast_to(inv, pos_mm.shape[:-1] + (4,)), pos_mm - pos_mm[thorax]) * POS_UM_PER_MM
    rel_quat = quat_mul(np.broadcast_to(inv, quat.shape), quat / np.linalg.norm(quat, axis=-1, keepdims=True))
    return rel_pos, rel_quat


def continuous(quats: np.ndarray, prev: np.ndarray | None = None) -> np.ndarray:
    """Flip signs so consecutive frames of each body stay in one hemisphere (q and -q are the same rotation).

    Without `prev` the FIRST frame is then made w >= 0 (needed so the viewer's slerp never takes the long way round).
    With `prev` (the last frame of the previous chunk, (bodies, 4)) the chunk continues that hemisphere instead."""
    out = quats.copy()
    if prev is not None:
        flip = np.sum(out[0] * prev, axis=-1) < 0
        out[0][flip] *= -1
    for t in range(1, len(out)):
        flip = np.sum(out[t] * out[t - 1], axis=-1) < 0
        out[t][flip] *= -1
    if prev is None:
        flip0 = out[0, :, 0] < 0
        out[:, flip0] *= -1
    return out


def unwrap_continue(raw_deg, last_raw: float | None, last_abs: float | None):
    """Unwrap wrapped headings (degrees) of one chunk, continuing the previous chunk's unwrapped heading.

    Returns (unwrapped, last raw, last unwrapped); pass the last two back with the next chunk (None at the start)."""
    raw = np.asarray(raw_deg, dtype=float)
    if raw.size == 0:
        return raw, last_raw, last_abs
    prev_raw = raw[0] if last_raw is None else last_raw
    base = raw[0] if last_abs is None else last_abs
    step = np.diff(np.concatenate([[prev_raw], raw]))
    out = base + np.cumsum((step + 180.0) % 360.0 - 180.0)
    return out, float(raw[-1]), float(out[-1])


def chunk_frames(ms: float) -> int:
    """Clip frames in a chunk of `ms` (must be a whole number of 200 Hz frames)."""
    n = ms / 1000.0 * SAMPLE_HZ if isinstance(ms, (int, float)) and math.isfinite(ms) else 0.0
    if n < 1 or abs(n - round(n)) > 1e-6:
        raise ValueError(f"a chunk must be a whole number of {1000 // SAMPLE_HZ} ms frames, got {ms}")
    return int(round(n))


def yaw_deg(thorax_quat: np.ndarray) -> np.ndarray:
    """Heading (degrees, counter-clockwise from above, wrapped) of the thorax x axis; thorax_quat (n, 4) wxyz."""
    q = thorax_quat / np.linalg.norm(thorax_quat, axis=-1, keepdims=True)
    fwd = quat_rotate(q, np.broadcast_to(np.array([1.0, 0.0, 0.0]), q.shape[:-1] + (3,)))
    return np.degrees(np.arctan2(fwd[..., 1], fwd[..., 0]))


def sample_stride(dt_s: float) -> int:
    """Physics steps per 200 Hz clip frame; the step must divide the frame period exactly."""
    stride = 1.0 / (SAMPLE_HZ * dt_s)
    if abs(stride - round(stride)) > 1e-6 or round(stride) < 1:
        raise ValueError(f"physics step {dt_s} s does not divide the {SAMPLE_HZ} Hz frame period (1/{SAMPLE_HZ} s)")
    return int(round(stride))


def drive_schedule(drive_bins, escape_bins, bin_ms: float, n_steps: int, dt_s: float):
    """Per-physics-step drive (n_steps, 2) and escape flags (n_steps,) from per-bin values.

    The drive is linearly interpolated between bin centres (held before the first and after the last); the escape flag is
    held per bin (a step belongs to the bin containing its time) and forces a zero drive."""
    d = np.asarray(drive_bins, dtype=float).reshape(-1, 2)
    t = np.arange(n_steps) * dt_s * 1000.0
    if len(d) == 0:
        return np.zeros((n_steps, 2)), np.zeros(n_steps, bool)
    centres = (np.arange(len(d)) + 0.5) * bin_ms
    out = np.stack([np.interp(t, centres, d[:, k]) for k in (0, 1)], axis=-1)
    esc = np.asarray(escape_bins if escape_bins is not None else np.zeros(len(d)), dtype=bool).reshape(-1)
    esc = np.pad(esc, (0, max(0, len(d) - len(esc))))[: len(d)]
    idx = np.clip((t // bin_ms).astype(int), 0, len(d) - 1)
    flags = esc[idx]
    out[flags] = 0.0
    return out, flags


# ---- flygym rig (lazy imports) ----

def reference_neutral():
    """Thorax-relative transforms of the leg bodies in the EXACT neutral pose our mesh is assembled in.

    flygym's own `neutral` keyframe of the walking model is an IK fit of this pose to a different joint axis order
    (whole-degree angles, a 2-DoF trochanter), so its bodies sit tens of micrometres away from the spec pose. The
    mesh is attached to the body frames in the spec pose (pose/_manual_specs/neutral.yaml, axis order roll-pitch-yaw),
    so the reference here is a model built in that axis order, which is the one flygym itself uses as the reference
    of its pose conversion. Body frames are rigid in both models, so M(t) * M_ref^-1 moves the mesh correctly."""
    require_flygym()
    import mujoco as mj
    from flygym import assets_dir
    from flygym.anatomy import JointPreset, Skeleton
    from flygym.compose import KinematicPose, NeuroMechFly

    pose = KinematicPose(path=assets_dir / "model/neuromechfly/pose/_manual_specs/neutral.yaml")
    fly = NeuroMechFly()
    fly.add_joints(Skeleton(axis_order=pose.axis_order, joint_preset=JointPreset.ALL_BIOLOGICAL), neutral_pose=pose)
    model, data = fly.compile()
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_kinematics(model, data)
    ids = {mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)}
    order = ["c_thorax", *BODIES]
    ref_pos = data.xpos[[ids[n] for n in order]]
    ref_quat = data.xquat[[ids[n] for n in order]]
    # the viewer re-applies each thorax-relative motion about the thorax origin of the mesh frame; that needs the
    # thorax frame to be the mesh frame translated (not rotated)
    if abs(abs(ref_quat[0][0]) - 1.0) > 1e-9:
        raise RuntimeError(f"the reference thorax is rotated in the mesh frame (quat {ref_quat[0]}); not supported")
    rel_pos, rel_quat = thorax_relative(ref_pos, ref_quat, 0)
    return rel_pos[1:], rel_quat[1:], ref_pos[0] * POS_UM_PER_MM


def build_sim(dt_s: float = None):
    require_flygym()
    from flygym import Simulation
    from flygym.anatomy import ContactBodiesPreset
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import make_locomotion_fly

    fly = make_locomotion_fly(name="w", add_adhesion=True, colorize=False)
    world = FlatGroundWorld()
    world.add_fly(
        fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY, add_ground_contact_sensors=True,
    )
    return fly, Simulation(world, timestep=DEFAULT_DT_S if dt_s is None else float(dt_s))


def make_rig(dt_s: float = None) -> SimpleNamespace:
    """The simulation plus the body bookkeeping (thorax index, leg-body indices, leg order)."""
    fly, sim = build_sim(dt_s)
    order = [b.name for b in fly.get_bodysegs_order()]
    return SimpleNamespace(
        fly=fly, sim=sim, thorax=order.index("c_thorax"), idx=[order.index(n) for n in BODIES],
        legs_order=[str(leg) for leg in fly.get_legs_order()],
    )


def start_controller(rig: SimpleNamespace, seed: int):
    """Reset the simulation, create and reset the HybridTurningController, settle (warmup). Returns the controller."""
    from flygym_demo.complex_terrain import (
        HybridTurningController, LocomotionAction, PreprogrammedSteps, apply_locomotion_action,
    )

    steps = PreprogrammedSteps()
    dofs = rig.fly.get_actuated_jointdofs_order("position")
    ctl = HybridTurningController(timestep=rig.sim.timestep, preprogrammed_steps=steps, output_dof_order=dofs)
    rig.sim.reset()
    ctl.reset(seed=seed)
    apply_locomotion_action(rig.sim, rig.fly.name, LocomotionAction(steps.default_pose_by_dof_order(dofs), np.ones(6, bool)))
    rig.sim.warmup()
    return ctl


def control_step(rig: SimpleNamespace, ctl, drive) -> object:
    """One controller + physics step with the descending signal [left, right]; returns the applied action."""
    from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action

    obs = HybridControllerObservation.from_sim(rig.sim, rig.fly.name)
    action = ctl.step(np.array(drive, float), obs)
    apply_locomotion_action(rig.sim, rig.fly.name, action)
    rig.sim.step()
    return action


def sample_frame(rig: SimpleNamespace):
    """The pose now: (leg-body positions um, leg-body quats, both thorax-relative; thorax world position mm and quat;
    per-leg ground contact). The one sampling code of the clip runner and the live stream."""
    bp, bq = rig.sim.get_body_positions(rig.fly.name), rig.sim.get_body_rotations(rig.fly.name)
    p, q = thorax_relative(bp, bq, rig.thorax)
    contact = np.asarray(rig.sim.get_ground_contact_info(rig.fly.name)[0]) > 0
    return p[rig.idx], q[rig.idx], bp[rig.thorax].copy(), np.asarray(bq[rig.thorax], float).copy(), contact


# ---- the runner ----

def run_body(drive_bins, bin_ms: float, duration_ms: float, seed: int = 0, escape=None, dt_s: float = None) -> dict:
    """Simulate `duration_ms` of the body under the per-bin drive and return a fly-walk/1 clip (see module doc).

    drive_bins: BodyDrive (flybrain.body_drive) or an (n_bins, 2) sequence of [left, right]; escape: per-bin flags
    (taken from a BodyDrive when not given); dt_s: physics (and CPG) step, DEFAULT_DT_S when None. Raises BodyUnavailable when flygym is not installed."""
    require_flygym()
    if hasattr(drive_bins, "drive"):
        if escape is None:
            escape = drive_bins.escape
        drive_bins = drive_bins.drive
    if not (bin_ms > 0 and math.isfinite(bin_ms)) or not (duration_ms > 0 and math.isfinite(duration_ms)):
        raise ValueError("bin_ms and duration_ms must be positive")

    rig = make_rig(dt_s)
    sim = rig.sim
    neutral_pos, neutral_quat, thorax_origin_um = reference_neutral()
    ctl = start_controller(rig, seed)
    dt = float(sim.timestep)
    stride = sample_stride(dt)
    n_frames = max(1, int(round(duration_ms / 1000.0 * SAMPLE_HZ)))
    n_steps = n_frames * stride
    sched, esc = drive_schedule(drive_bins, escape, bin_ms, n_steps, dt)

    pos, quat, adhesion, contact, esc_f, drive_f, thorax_mm, thorax_q = [], [], [], [], [], [], [], []
    for i in range(n_steps):
        action = control_step(rig, ctl, sched[i])
        if i % stride == 0:
            p, q, t_mm, t_q, touching = sample_frame(rig)
            pos.append(p); quat.append(q)
            adhesion.append(np.asarray(action.adhesion_onoff, bool))
            contact.append(touching)
            thorax_mm.append(t_mm); thorax_q.append(t_q)
            esc_f.append(bool(esc[i])); drive_f.append(sched[i].copy())
    pos = np.array(pos); quat = continuous(np.array(quat))
    flip = np.sum(neutral_quat * quat[0], axis=-1) < 0
    neutral_quat = np.where(flip[:, None], -neutral_quat, neutral_quat)
    thorax_mm = np.array(thorax_mm)
    yaw = np.degrees(np.unwrap(np.radians(yaw_deg(np.array(thorax_q)))))
    yaw -= yaw[0]
    world = np.round((thorax_mm - np.array([thorax_mm[0, 0], thorax_mm[0, 1], 0.0])) * POS_UM_PER_MM).astype(int)

    q4 = lambda a: np.round(a * QUAT_SCALE).astype(int)  # noqa: E731
    return {
        "format": "fly-walk/1",
        "units": {"position": "um", "quaternion": f"w,x,y,z x {QUAT_SCALE}", "time": "s"},
        "frame": "fly frame relative to the thorax body: x anterior, y left, z dorsal; rotation = body frame -> thorax frame",
        "quat_scale": QUAT_SCALE,
        "thorax_origin_um": np.round(thorax_origin_um).astype(int).tolist(),
        "bodies": BODIES,
        "legs": rig.legs_order,
        "sample_rate_hz": SAMPLE_HZ,
        "cycle_hz": CPG_HZ,
        "n_frames": n_frames,
        "duration_s": n_frames / SAMPLE_HZ,
        "loop": False,
        "neutral": {"pos_um": np.round(neutral_pos).astype(int).tolist(), "quat": q4(neutral_quat).tolist()},
        "frames": np.concatenate([np.round(pos).astype(int), q4(quat)], axis=-1).tolist(),
        "adhesion": np.array(adhesion).astype(int).tolist(),
        "contact": np.array(contact).astype(int).tolist(),
        "escape": [int(v) for v in esc_f],
        "drive": np.round(np.array(drive_f), 4).tolist(),
        "thorax_world": {
            "pos_um": world.tolist(),
            "yaw_deg": [round(float(v), 3) for v in yaw],
            "note": "flygym world frame: x forward at the start, y left, z up; yaw positive = turning left",
        },
        "provenance": {
            "generator": "flybrain.body_runner.run_body",
            "flygym": version("flygym"),
            "mujoco": version("mujoco"),
            "controller": "flygym_demo.complex_terrain.HybridTurningController (CPG tripod network + preprogrammed steps + adhesion rules)",
            "seed": int(seed),
            "bin_ms": float(bin_ms),
            "physics_dt_s": dt,
            "world": "FlatGroundWorld, fly at z = 0.8 mm, tibia+tarsus ground contact",
            "note": "The descending drive comes from OUR mapping of brain readouts (flybrain.body_drive); no takeoff model.",
        },
    }


# ---- the live stream ----

class BodyStream:
    """The physics body as a live process: keeps the flygym simulation and the controller alive between chunks.

    `advance(ms, drive, escape)` steps the body by a chunk (a whole number of 5 ms frames) under a CONSTANT descending
    drive [left, right] (zero while `escape`: the legs stop, as in `run_body`) and returns the frames of that chunk in the
    fly-walk/1 format (`header` holds the static part: bodies, legs, neutral pose, scales). Frames are sampled on the
    global 200 Hz clock exactly like `run_body`, quaternion signs and the unwrapped heading continue across chunks, and
    the world position is relative to the first frame, so concatenated chunks equal one long advance. Not thread-safe;
    the caller serialises chunks. Raises BodyUnavailable without the `body` extra."""

    mode = "physics"
    takes_command = False
    caption = "Body: full MuJoCo physics (NeuroMechFly v2), stepped live from the brain's drive; slower than the kinematic mode."

    def __init__(self, seed: int = 0, dt_s: float = None) -> None:
        require_flygym()
        self.rig = make_rig(dt_s)
        neutral_pos, neutral_quat, thorax_origin_um = reference_neutral()
        self.ctl = start_controller(self.rig, seed)
        self.dt = float(self.rig.sim.timestep)
        self.stride = sample_stride(self.dt)
        self.steps = 0
        _, q0, _, _, _ = sample_frame(self.rig)
        self._prev_quat = continuous(q0[None])[0]  # the pose before the first step fixes the sign convention
        flip = np.sum(neutral_quat * self._prev_quat, axis=-1) < 0
        neutral_quat = np.where(flip[:, None], -neutral_quat, neutral_quat)
        self._origin_xy: np.ndarray | None = None
        self._yaw = (None, None)  # last raw, last unwrapped
        self._yaw0: float | None = None
        self.header = {
            "format": "fly-walk/1",
            "units": {"position": "um", "quaternion": f"w,x,y,z x {QUAT_SCALE}", "time": "s"},
            "frame": "fly frame relative to the thorax body: x anterior, y left, z dorsal; rotation = body frame -> thorax frame",
            "quat_scale": QUAT_SCALE,
            "thorax_origin_um": np.round(thorax_origin_um).astype(int).tolist(),
            "bodies": BODIES,
            "legs": self.rig.legs_order,
            "sample_rate_hz": SAMPLE_HZ,
            "cycle_hz": CPG_HZ,
            "neutral": {"pos_um": np.round(neutral_pos).astype(int).tolist(),
                        "quat": np.round(neutral_quat * QUAT_SCALE).astype(int).tolist()},
            "physics_dt_s": self.dt,
            "frame_layout": "frames[i][body] = [x, y, z (um), qw, qx, qy, qz (x quat_scale)]",
        }

    @property
    def t_ms(self) -> float:
        return self.steps * self.dt * 1000.0

    def advance(self, ms: float, drive, escape: bool = False) -> dict:
        n_frames = chunk_frames(ms)
        d = np.asarray(drive, dtype=float).reshape(-1)
        if d.shape != (2,) or not np.isfinite(d).all():
            raise ValueError(f"drive must be two finite numbers [left, right], got {drive!r}")
        applied = np.zeros(2) if escape else d
        pos, quat, contact, thorax_mm, thorax_q = [], [], [], [], []
        for i in range(n_frames * self.stride):
            control_step(self.rig, self.ctl, applied)
            if i % self.stride == 0:  # the chunk starts on a frame boundary, so this is the global 200 Hz clock
                p, q, t_mm, t_q, touching = sample_frame(self.rig)
                pos.append(p); quat.append(q); contact.append(touching); thorax_mm.append(t_mm); thorax_q.append(t_q)
        self.steps += n_frames * self.stride
        quat = continuous(np.array(quat), prev=self._prev_quat)
        self._prev_quat = quat[-1]
        thorax_mm = np.array(thorax_mm)
        if self._origin_xy is None:
            self._origin_xy = thorax_mm[0, :2].copy()
        world = np.round((thorax_mm - np.array([self._origin_xy[0], self._origin_xy[1], 0.0])) * POS_UM_PER_MM).astype(int)
        yaw, *self._yaw = unwrap_continue(yaw_deg(np.array(thorax_q)), *self._yaw)
        if self._yaw0 is None:
            self._yaw0 = float(yaw[0])
        frames = np.concatenate([np.round(np.array(pos)).astype(int), np.round(quat * QUAT_SCALE).astype(int)], axis=-1)
        return {
            "n_frames": n_frames,
            "frames": frames,
            "contact": np.array(contact).astype(int),
            "escape": np.full(n_frames, int(bool(escape))),
            "drive": np.tile(np.round(applied, 4), (n_frames, 1)),
            "thorax": {"pos_um": world, "yaw_deg": np.round(yaw - self._yaw0, 3)},
        }
