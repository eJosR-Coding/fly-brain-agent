"""Build web/public/fly-motion-library.json: recorded physics gait cycles for the kinematic live body.

Needs the optional `body` extra (flygym 2.1.1 + MuJoCo):

    uv run --extra body python scripts/build_motion_library.py [--out web/public/fly-motion-library.json]

For every drive pair on a (pace, turn) grid the NeuroMechFly v2 HybridTurningController is run in the physics
simulation (the same rig as flybrain.body_runner), the start-up transient is discarded and 12 stepping cycles (1 s) are
recorded at 200 Hz as thorax-relative leg-body transforms, together with the MEASURED thorax forward speed (mm/s) and
yaw rate (deg/s) of that very run. The drive pair of a grid point is the one flybrain.body_drive builds for that pace
and turn (`walking_drive`), so the library is indexed by the same command the brain's mapping produces.

Cycle: the controller steps at exactly 12 Hz, so 3 cycles are 50 frames at 200 Hz and the phase of frame k is
(3k mod 50)/50; the 4 repeats of the 3-cycle block are averaged into one 50-bin cycle (the residual between repeats is
reported per clip). The bins are rotated so that bin 0 is the LEFT-FRONT (LF) SWING ONSET, the frame at which the
controller's LF adhesion command goes from stance (on) to swing (off): every clip then has the same gait phase at the
same index, which is what lets the runtime blend clips at equal phase. Every second bin is kept (25 phase samples
per cycle; the runtime restores 50 with a periodic cubic) and the reconstruction error of that decimation is reported.

Also stored: a standing pose (zero drive, legs settled) and the escape entry (the standing pose + a wings-raised flag,
as in the viewer). Spin-in-place clips (opposite signs of the drive: the controller walks the two sides in opposite
directions) are included when they really rotate the body.

Quantisation as fly-walk/1: positions in micrometres and quaternions (w, x, y, z) x 10000 as int16, written as base64
of little-endian int16 arrays with shape (phase, body, 7). This is a recording of a published locomotion controller;
it is not driven by, and not part of, the connectome.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybrain import body_drive  # noqa: E402
from flybrain.body_runner import (  # noqa: E402
    BODIES, CPG_HZ, QUAT_SCALE, SAMPLE_HZ, continuous, control_step, make_rig,
    reference_neutral, sample_frame, start_controller, yaw_deg,
)

PACE_GRID = (0.0, 0.33, 0.67, 1.0)
TURN_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
SPIN_DRIVE = 0.8  # |drive| of each side for the spin-in-place clips
SEED = 0
PHYSICS_DT_S = 2e-4  # the live physics body's step
TRANSIENT_S = 0.6
CYCLES = 12  # 1 s recorded: 4 repeats of a 3-cycle (50-frame) block
BLOCK = 50  # frames per 3 cycles at 12 Hz / 200 Hz
BINS = 50
KEEP = 2  # decimation: 25 phase samples per cycle


def record(rig, drive, seed: int = SEED) -> dict:
    """Run the controller with a constant drive; return the per-frame thorax-relative pose, adhesion, contact, world."""
    ctl = start_controller(rig, seed)
    stride = int(round(1.0 / (SAMPLE_HZ * rig.sim.timestep)))
    n_frames = int(round(CYCLES / CPG_HZ * SAMPLE_HZ))
    n_transient = int(round(TRANSIENT_S / rig.sim.timestep))
    n_total = n_transient + n_frames * stride
    pos, quat, adh, con, thorax, tq = [], [], [], [], [], []
    for i in range(n_total):
        action = control_step(rig, ctl, drive)
        j = i - n_transient
        if j >= 0 and j % stride == 0:
            p, q, t_mm, t_q, touching = sample_frame(rig)
            pos.append(p); quat.append(q); adh.append(np.asarray(action.adhesion_onoff, bool))
            con.append(touching); thorax.append(t_mm); tq.append(t_q)
    return {"pos": np.array(pos), "quat": continuous(np.array(quat)), "adh": np.array(adh), "con": np.array(con),
            "thorax_mm": np.array(thorax), "thorax_q": np.array(tq)}


def thorax_motion(rec: dict) -> tuple[float, float, float, float]:
    """(forward speed mm/s, lateral speed mm/s, yaw rate deg/s, mean thorax z um) over the recording."""
    dt = 1.0 / SAMPLE_HZ
    yaw = np.unwrap(np.radians(yaw_deg(rec["thorax_q"])))
    xy = rec["thorax_mm"][:, :2]
    mid = 0.5 * (yaw[1:] + yaw[:-1])
    heading = np.stack([np.cos(mid), np.sin(mid)], axis=-1)
    vel = np.diff(xy, axis=0) / dt
    fwd = float(np.mean(np.sum(vel * heading, axis=-1)))
    lat = float(np.mean(vel[:, 0] * -heading[:, 1] + vel[:, 1] * heading[:, 0]))
    rate = float(np.degrees(yaw[-1] - yaw[0]) / ((len(yaw) - 1) * dt))
    return fwd, lat, rate, float(rec["thorax_mm"][:, 2].mean() * 1000.0)


def to_cycle(rec: dict, neutral_quat: np.ndarray) -> dict:
    """Frames -> one 50-bin cycle ordered by gait phase, rotated to start at the LF swing onset (see module doc)."""
    n = len(rec["pos"])
    assert n == CYCLES * SAMPLE_HZ // int(CPG_HZ)  # 200
    flip = np.sum(rec["quat"] * neutral_quat[None], axis=-1) < 0
    quat = np.where(flip[..., None], -rec["quat"], rec["quat"])
    reps = n // BLOCK

    def by_phase(a):  # (n, ...) -> (reps, BINS, ...) with bin p = frame k where 3k = p (mod 50), i.e. k = 17 p (mod 50)
        order = (17 * np.arange(BINS)) % BLOCK
        return a.reshape(reps, BLOCK, *a.shape[1:])[:, order]

    pos, qs = by_phase(rec["pos"]), by_phase(quat)
    pos_m = pos.mean(axis=0)
    q_m = qs.mean(axis=0)
    q_m /= np.linalg.norm(q_m, axis=-1, keepdims=True)
    residual = float(np.abs(pos - pos_m).max())
    adh = by_phase(rec["adh"].astype(float)).mean(axis=0) > 0.5  # (BINS, 6) stance command
    con = by_phase(rec["con"].astype(float)).mean(axis=0) > 0.5
    lf = adh[:, 0]
    onsets = [p for p in range(BINS) if lf[p - 1] and not lf[p]]
    if len(onsets) != 1:
        raise RuntimeError(f"LF stance/swing is not a clean single cycle (onsets {onsets}); the clip is not periodic")
    s = onsets[0]
    roll = lambda a: np.roll(a, -s, axis=0)  # noqa: E731
    return {"pos": roll(pos_m), "quat": roll(q_m), "con": roll(con), "adh": roll(adh), "residual_um": residual,
            "swing_onset_bin": s}


def decimate(c: dict) -> dict:
    return {**c, "pos": c["pos"][::KEEP], "quat": c["quat"][::KEEP], "con": c["con"][::KEEP]}


def cubic_periodic(a: np.ndarray, factor: int) -> np.ndarray:
    """Upsample the first axis (periodic) by `factor` with Catmull-Rom; the runtime uses the same rule."""
    n = len(a)
    out = np.empty((n * factor,) + a.shape[1:])
    for k in range(n * factor):
        i, f = divmod(k, factor)
        t = f / factor
        p0, p1, p2, p3 = a[(i - 1) % n], a[i % n], a[(i + 1) % n], a[(i + 2) % n]
        out[k] = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
    return out


def seam_stats(pos: np.ndarray) -> dict:
    steps = np.abs(np.diff(np.concatenate([pos, pos[:1]]), axis=0)).max(axis=(1, 2))
    return {"wrap_step_um": round(float(steps[-1]), 1), "max_step_um": round(float(steps[:-1].max()), 1),
            "wrap_over_max_step": round(float(steps[-1] / steps[:-1].max()), 3)}


def pack(pos: np.ndarray, quat: np.ndarray) -> str:
    f = np.concatenate([np.round(pos), np.round(quat * QUAT_SCALE)], axis=-1)
    if np.abs(f).max() > 32767:
        raise ValueError("a value does not fit int16")
    return base64.b64encode(f.astype("<i2").tobytes()).decode()


def mask_bits(con: np.ndarray) -> list[int]:
    return [int(sum(int(b) << i for i, b in enumerate(row))) for row in np.atleast_2d(con)]


def build(out_path: Path, spin: bool = True) -> dict:
    rig = make_rig(PHYSICS_DT_S)
    n_pos, n_quat, thorax_origin_um = reference_neutral()
    clips, report = [], []
    jobs = [("walk", p, t, body_drive.walking_drive(p, t)) for p in PACE_GRID for t in TURN_GRID]
    if spin:
        jobs += [("spin", 0.0, 1.0, np.array([-SPIN_DRIVE, SPIN_DRIVE])), ("spin", 0.0, -1.0, np.array([SPIN_DRIVE, -SPIN_DRIVE]))]
    for kind, pace, turn, drive in jobs:
        rec = record(rig, drive)
        fwd, lat, rate, z_um = thorax_motion(rec)
        cyc = to_cycle(rec, n_quat)
        full_pos = cyc["pos"]
        d = decimate(cyc)
        up = cubic_periodic(d["pos"], KEEP)
        recon = float(np.abs(up - full_pos).max())
        cid = f"{kind[0]}{pace:.2f}_t{turn:+.2f}"
        clip = {"id": cid, "kind": kind, "pace": pace, "turn": turn, "drive": [round(float(v), 4) for v in drive],
                "speed_mm_s": round(fwd, 3), "lateral_mm_s": round(lat, 3), "yaw_rate_deg_s": round(rate, 3),
                "thorax_z_um": round(z_um, 1), "frames_b64": pack(d["pos"], d["quat"]), "contact": mask_bits(d["con"]),
                "seam": {**seam_stats(d["pos"]), "repeat_residual_um": round(cyc["residual_um"], 1),
                         "decimation_error_um": round(recon, 1), "lf_swing_onset_bin_before_rotation": cyc["swing_onset_bin"]}}
        clips.append(clip)
        report.append((cid, fwd, lat, rate, clip["seam"]))
        print(f"{cid}: drive {clip['drive']} speed {fwd:.2f} mm/s lateral {lat:.2f} yaw {rate:.1f} deg/s seam {clip['seam']}", file=sys.stderr)

    # spin clips that do not rotate (or walk away) are not a spin-in-place: keep the library honest
    clips = [c for c in clips if c["kind"] == "walk" or (abs(c["yaw_rate_deg_s"]) > 100 and np.hypot(c["speed_mm_s"], c["lateral_mm_s"]) < 8)]

    # standing pose: zero drive, legs settled
    ctl = start_controller(rig, SEED)
    for _ in range(int(round(0.4 / rig.sim.timestep))):
        control_step(rig, ctl, [0.0, 0.0])
    sp, sq, st_mm, _, scon = sample_frame(rig)
    flip = np.sum(sq * n_quat, axis=-1) < 0
    sq = np.where(flip[:, None], -sq, sq)
    stand = {"frames_b64": pack(sp[None], sq[None]), "contact": mask_bits(scon), "thorax_z_um": round(float(st_mm[2] * 1000.0), 1)}

    doc = {
        "format": "fly-motion-library/1",
        "units": {"position": "um", "quaternion": f"w,x,y,z x {QUAT_SCALE}", "time": "s", "speed": "mm/s", "yaw_rate": "deg/s"},
        "frame": "fly frame relative to the thorax body: x anterior, y left, z dorsal; rotation = body frame -> thorax frame",
        "quat_scale": QUAT_SCALE,
        "thorax_origin_um": np.round(thorax_origin_um).astype(int).tolist(),
        "bodies": BODIES,
        "legs": ["lf", "lm", "lh", "rf", "rm", "rh"],
        "sample_rate_hz": SAMPLE_HZ,
        "cycle_hz": CPG_HZ,
        "phase_samples": BINS // KEEP,
        "phase_alignment": "bin 0 of every clip is the LEFT-FRONT (LF) SWING ONSET: the controller's LF adhesion command goes "
                           "stance -> swing there; phase advances 12 Hz x t, so equal phase = equal gait phase in every clip",
        "encoding": {"dtype": "<i2", "shape": ["phase", "body", 7], "layout": "[x, y, z (um), qw, qx, qy, qz (x quat_scale)]",
                     "container": "base64 of little-endian int16", "contact": "per phase sample: bitmask, bit i = leg i (order legs) on the ground"},
        "mapping": {"drive": "flybrain.body_drive.walking_drive(pace, turn)", "pace": "0..1", "turn": "-1..1, positive = left = +yaw"},
        "pace_grid": list(PACE_GRID),
        "turn_grid": list(TURN_GRID),
        "neutral": {"pos_um": np.round(n_pos).astype(int).tolist(), "quat": np.round(n_quat * QUAT_SCALE).astype(int).tolist()},
        "stand": stand,
        "escape": {"pose": "stand", "wings_raised": True, "note": "the viewer raises the wings; the legs stand (no takeoff model)"},
        "clips": clips,
        "provenance": {
            "generator": "scripts/build_motion_library.py",
            "flygym": version("flygym"), "mujoco": version("mujoco"),
            "controller": "flygym_demo.complex_terrain.HybridTurningController (CPG tripod network + preprogrammed steps + adhesion rules)",
            "seed": SEED, "physics_dt_s": PHYSICS_DT_S, "transient_discarded_s": TRANSIENT_S, "recorded_cycles": CYCLES,
            "world": "FlatGroundWorld, fly at z = 0.8 mm, tibia+tarsus ground contact",
            "speed_definition": "mean thorax velocity component along the thorax heading over the 12 recorded cycles; yaw rate = net heading change / time",
            "note": "A recording of a published locomotion controller replayed as an execution model; it is not driven by the brain model.",
        },
    }
    out_path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="web/public/fly-motion-library.json", type=Path)
    ap.add_argument("--no-spin", action="store_true")
    args = ap.parse_args(argv)
    doc = build(args.out, spin=not args.no_spin)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes): {len(doc['clips'])} clips, {doc['phase_samples']} phase samples", file=sys.stderr)


if __name__ == "__main__":
    main()
