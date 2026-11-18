"""Export a looping physics walking clip of NeuroMechFly v2 (flygym 2.1.1) for the viewer.

flygym and MuJoCo are an optional extra of the project (`uv sync --extra body`); the shared logic (quaternion maths, the
neutral reference, the simulation rig) lives in src/flybrain/body_runner.py, which this script imports. Run it with any
environment that has flygym 2.1.1 installed (flygym_demo ships in the same wheel):

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybrain.body_runner import (  # noqa: E402
    BODIES, CPG_HZ, QUAT_SCALE, SAMPLE_HZ, continuous, control_step, make_rig,
    reference_neutral, start_controller, thorax_relative,
)

DRIVE = (1.0, 1.0)  # descending drive [left, right]: straight walking
SEED = 0
TRANSIENT_S = 0.5  # discarded start-up (the CPG and the contacts settle)
TARGET_S = 1.0


def export(out_path: Path) -> dict:
    rig = make_rig()
    fly, sim, thorax, idx, legs_order = rig.fly, rig.sim, rig.thorax, rig.idx, rig.legs_order

    neutral_pos, neutral_quat, thorax_origin_um = reference_neutral()
    # the walking model's own neutral keyframe, only to report how far flygym's fitted pose is from the spec pose
    sim.reset()
    sim.forward()
    kpos, _ = thorax_relative(sim.get_body_positions(fly.name), sim.get_body_rotations(fly.name), thorax)
    keyframe_offset_um = float(np.linalg.norm(kpos[idx] - neutral_pos, axis=-1).max())

    ctl = start_controller(rig, SEED)

    stride = int(round(1.0 / (SAMPLE_HZ * sim.timestep)))
    cycles = max(1, round(CPG_HZ * TARGET_S))  # whole cycles closest to the target length
    n_frames = int(round(cycles / CPG_HZ * SAMPLE_HZ))
    n_transient = int(round(TRANSIENT_S / sim.timestep))
    n_total = n_transient + (n_frames + 1) * stride  # one extra frame to measure the loop seam
    pos, quat, adhesion, contact = [], [], [], []
    thorax_mm = []
    for i in range(n_total):
        action = control_step(rig, ctl, DRIVE)
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
