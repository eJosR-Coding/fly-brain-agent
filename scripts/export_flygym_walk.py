"""Export a looping physics walking clip of NeuroMechFly v2 (flygym 2.1.1) for the viewer.

This is NOT part of the project's Python environment: flygym and MuJoCo are heavy and stay out of the project's
dependencies. Run it with an external environment that has flygym 2.1.1 (+ the flygym_demo tutorial helpers) installed:

    <flygym-env>/bin/python -I scripts/export_flygym_walk.py --out web/public/fly-walk.json

What it does: flygym's HybridTurningController (the 2.1.1 turning-controller tutorial recipe: CPG tripod network +
preprogrammed steps + rule-based leg adhesion) walks straight (descending drive [1, 1]) on flat ground. After a
start transient is discarded, the world transform (MuJoCo xpos / xquat) of EVERY leg body (6 legs x coxa,
trochanterfemur, tibia, tarsus1..5 = 48 bodies) is expressed RELATIVE TO THE THORAX (so the fly walks in place) and
sampled at 200 Hz for a whole number of 12 Hz stepping cycles (12 cycles = exactly 1 s), so the clip loops. The same
bodies' thorax-relative transforms at the NEUTRAL pose (the pose our mesh is assembled in) are written next to them.

Output frame: fly frame, thorax-relative (thorax_origin_um places that origin in the mesh frame), x anterior, y left, z dorsal; positions in micrometres (integers),
quaternions (w, x, y, z) scaled by 10000 (integers). The controller is a published locomotion model, not the brain.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

DRIVE = (1.0, 1.0)  # descending drive [left, right]: straight walking
SEED = 0
SAMPLE_HZ = 200
CPG_HZ = 12.0  # the controller's fixed CPG frequency
TRANSIENT_S = 0.5  # discarded start-up (the CPG and the contacts settle)
TARGET_S = 1.0
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LINKS = ("coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus2", "tarsus3", "tarsus4", "tarsus5")
BODIES = [f"{leg}_{link}" for leg in LEGS for link in LINKS]
POS_UM_PER_MM = 1000.0
QUAT_SCALE = 10000


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


def continuous(quats: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive frames of each body stay in one hemisphere (q and -q are the same rotation),
    then make the FIRST frame w >= 0. Needed so the viewer's slerp never takes the long way round."""
    out = quats.copy()
    for t in range(1, len(out)):
        flip = np.sum(out[t] * out[t - 1], axis=-1) < 0
        out[t][flip] *= -1
    flip0 = out[0, :, 0] < 0
    out[:, flip0] *= -1
    return out


def reference_neutral():
    """Thorax-relative transforms of the leg bodies in the EXACT neutral pose our mesh is assembled in.

    flygym's own `neutral` keyframe of the walking model is an IK fit of this pose to a different joint axis order
    (whole-degree angles, a 2-DoF trochanter), so its bodies sit tens of micrometres away from the spec pose. The
    mesh is attached to the body frames in the spec pose (pose/_manual_specs/neutral.yaml, axis order roll-pitch-yaw),
    so the reference here is a model built in that axis order, which is the one flygym itself uses as the reference
    of its pose conversion. Body frames are rigid in both models, so M(t) * M_ref^-1 moves the mesh correctly."""
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


def build_sim():
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
    return fly, Simulation(world)


def export(out_path: Path) -> dict:
    from flygym.anatomy import BodySegment
    from flygym_demo.complex_terrain import (
        HybridControllerObservation, HybridTurningController, LocomotionAction, PreprogrammedSteps,
        apply_locomotion_action,
    )

    fly, sim = build_sim()
    order = [b.name for b in fly.get_bodysegs_order()]
    thorax = order.index("c_thorax")
    idx = [order.index(n) for n in BODIES]
    legs_order = [str(l) for l in fly.get_legs_order()]

    neutral_pos, neutral_quat, thorax_origin_um = reference_neutral()
    # the walking model's own neutral keyframe, only to report how far flygym's fitted pose is from the spec pose
    sim.reset()
    sim.forward()
    kpos, _ = thorax_relative(sim.get_body_positions(fly.name), sim.get_body_rotations(fly.name), thorax)
    keyframe_offset_um = float(np.linalg.norm(kpos[idx] - neutral_pos, axis=-1).max())

    steps = PreprogrammedSteps()
    dofs = fly.get_actuated_jointdofs_order("position")
    ctl = HybridTurningController(timestep=sim.timestep, preprogrammed_steps=steps, output_dof_order=dofs)
    sim.reset()
    ctl.reset(seed=SEED)
    apply_locomotion_action(sim, fly.name, LocomotionAction(steps.default_pose_by_dof_order(dofs), np.ones(6, bool)))
    sim.warmup()

    stride = int(round(1.0 / (SAMPLE_HZ * sim.timestep)))
    cycles = max(1, round(CPG_HZ * TARGET_S))  # whole cycles closest to the target length
    n_frames = int(round(cycles / CPG_HZ * SAMPLE_HZ))
    n_transient = int(round(TRANSIENT_S / sim.timestep))
    n_total = n_transient + (n_frames + 1) * stride  # one extra frame to measure the loop seam
    pos, quat, adhesion, contact = [], [], [], []
    thorax_mm = []
    for i in range(n_total):
        obs = HybridControllerObservation.from_sim(sim, fly.name)
        action = ctl.step(np.array(DRIVE, float), obs)
        apply_locomotion_action(sim, fly.name, action)
        sim.step()
        j = i - n_transient
        if j >= 0 and j % stride == 0:
            p, q = thorax_relative(sim.get_body_positions(fly.name), sim.get_body_rotations(fly.name), thorax)
            pos.append(p[idx]); quat.append(q[idx])
            adhesion.append(np.asarray(action.adhesion_onoff, bool))
            c = sim.get_ground_contact_info(fly.name)[0]
            contact.append(np.asarray(c) > 0)
            thorax_mm.append(sim.get_body_positions(fly.name)[thorax].copy())
    pos = np.array(pos); quat = continuous(np.array(quat)); adhesion = np.array(adhesion); contact = np.array(contact)
    thorax_mm = np.array(thorax_mm)

    # loop seam: the extra frame (one whole period later) against frame 0
    seam_pos = np.abs(pos[-1] - pos[0]).max()
    d = np.abs(np.sum(quat[-1] * quat[0], axis=-1)).clip(0, 1)
    seam_deg = float(np.degrees(2 * np.arccos(d)).max())
    step_pos = float(np.abs(np.diff(pos[:-1], axis=0)).max())
    pos, quat, adhesion, contact = pos[:n_frames], quat[:n_frames], adhesion[:n_frames], contact[:n_frames]
    # make the neutral quaternions use the clip's hemisphere (continuity with frame 0)
    flip = np.sum(neutral_quat * quat[0], axis=-1) < 0
    neutral_quat = np.where(flip[:, None], -neutral_quat, neutral_quat)

    span = (thorax_mm[-1] - thorax_mm[0])[:2]
    speed_mm_s = float(np.linalg.norm(span) / (cycles / CPG_HZ))
    q4 = lambda a: np.round(a * QUAT_SCALE).astype(int)  # noqa: E731
    doc = {
        "format": "fly-walk/1",
        "units": {"position": "um", "quaternion": f"w,x,y,z x {QUAT_SCALE}", "time": "s"},
        "frame": "fly frame relative to the thorax body: x anterior, y left, z dorsal; rotation = body frame -> thorax frame",
        "quat_scale": QUAT_SCALE,
        # thorax body origin in the mesh (neutral-pose world) frame, um: the frame all transforms below are relative to
        "thorax_origin_um": np.round(thorax_origin_um).astype(int).tolist(),
        "bodies": BODIES,
        "legs": legs_order,
        "sample_rate_hz": SAMPLE_HZ,
        "cycle_hz": CPG_HZ,
        "cycles": cycles,
        "n_frames": n_frames,
        "duration_s": n_frames / SAMPLE_HZ,
        "loop": True,
        "neutral": {"pos_um": np.round(neutral_pos).astype(int).tolist(), "quat": q4(neutral_quat).tolist()},
        # frames[f][b] = [x, y, z, qw, qx, qy, qz] of body b (order = "bodies") at frame f
        "frames": np.concatenate([np.round(pos).astype(int), q4(quat)], axis=-1).tolist(),
        # per frame and leg (order = "legs"): 1 = adhesion on (the controller's stance command), contact = ground contact
        "adhesion": adhesion.astype(int).tolist(),
        "contact": contact.astype(int).tolist(),
        "provenance": {
            "generator": "scripts/export_flygym_walk.py",
            "flygym": version("flygym"),
            "mujoco": version("mujoco"),
            "controller": "flygym_demo.complex_terrain.HybridTurningController (CPG tripod network + preprogrammed steps + adhesion rules)",
            "drive": list(DRIVE),
            "seed": SEED,
            "physics_dt_s": float(sim.timestep),
            "transient_discarded_s": TRANSIENT_S,
            "world": "FlatGroundWorld, fly at z = 0.8 mm, tibia+tarsus ground contact",
            "neutral_reference": "flygym model built in the axis order of pose/_manual_specs/neutral.yaml (the pose the mesh is assembled in)",
            "walking_model_keyframe_vs_reference_max_um": round(keyframe_offset_um, 1),
            "thorax_speed_mm_s": round(speed_mm_s, 2),
            "loop_seam": {"max_position_um": round(float(seam_pos), 2), "max_rotation_deg": round(seam_deg, 3),
                          "max_frame_to_frame_um": round(step_pos, 1)},
            "note": "A published locomotion controller replayed as an execution model; it is not driven by the brain model.",
        },
    }
    out_path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="web/public/fly-walk.json", type=Path)
    args = ap.parse_args(argv)
    doc = export(args.out)
    p = doc["provenance"]
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes): {doc['n_frames']} frames, {doc['cycles']} cycles, "
          f"speed {p['thorax_speed_mm_s']} mm/s, seam {p['loop_seam']}", file=sys.stderr)


if __name__ == "__main__":
    main()
