"""Build src/flybrain/data/fly-flight-clips.json from the flybody flight-imitation dataset (HDF5 + NPY; no pickle).

Source (GPL-3.0+): Vaxenburg et al. 2025, Nature, "Whole-body simulation of realistic fruit fly locomotion with deep
reinforcement learning"; Figshare article 25309105 (DOI 10.25378/janelia.25309105.v4), file datasets_flight-imitation.zip:
  flight-dataset_saccade-evasion_augmented.hdf5  -> trajectories/NNN/com_qpos (T x 7: x y z, quat w x y z; dt 0.2 ms)
  wing_pattern_fmech.npy                         -> (500, 3) one wingbeat: stroke, deviation, rotation (rad)
The trajectories are real fly flight (Muijres et al. 2014 Science 344:172; 2015 J Exp Biol 218:864).

Usage (needs h5py; reads data only): python scripts/build_flight_clips.py <flight dir with the 2 files> [out.json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

STEP = 5  # keep every 5th sample: 1 kHz
OUT = Path(__file__).resolve().parents[1] / "src" / "flybrain" / "data" / "fly-flight-clips.json"
WING_KEEP = 100  # samples of the one-wingbeat pattern


def yaw_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main(src: Path, out: Path) -> None:
    f = h5py.File(src / "flight-dataset_saccade-evasion_augmented.hdf5", "r")
    dt = float(f["timestep_seconds"][()])
    t = f["trajectories"]
    rows = []
    for k in t:
        if t[k]["trajectory_type"][()].decode() != "evasion_original":
            continue
        q = t[k]["com_qpos"][:].astype(float)
        yaw = yaw_of(q[:, 3:])
        rows.append((k, len(q), math.degrees(yaw[-1] - yaw[0]), math.degrees(np.ptp(yaw)), q))
    straight = sorted((r for r in rows if r[1] >= 900), key=lambda r: r[3])[:4]
    left = sorted((r for r in rows if r[2] > 40 and r[1] >= 900), key=lambda r: -r[1])[:2]
    right = sorted((r for r in rows if r[2] < -40 and r[1] >= 900), key=lambda r: -r[1])[:2]
    clips = []
    for tag, grp in (("straight", straight), ("left", left), ("right", right)):
        for k, n, net, ptp, q in grp:
            yaw = yaw_of(q[:, 3:])
            h0 = yaw[0]
            c, s = math.cos(-h0), math.sin(-h0)
            d = q[::STEP, :3] - q[0, :3]
            x = c * d[:, 0] - s * d[:, 1]
            y = s * d[:, 0] + c * d[:, 1]
            clips.append({
                "id": k, "kind": tag, "net_yaw_deg": round(net, 1), "duration_s": round((len(d) - 1) * dt * STEP, 4),
                "x": np.round(x, 4).tolist(), "y": np.round(y, 4).tolist(), "z_abs": np.round(q[::STEP, 2], 4).tolist(),
                "yaw_deg": np.round(np.degrees(yaw[::STEP] - h0), 2).tolist(),
            })
    w = np.load(src / "wing_pattern_fmech.npy", allow_pickle=False).astype(float)
    pat = w[:: len(w) // WING_KEEP][:WING_KEEP]
    doc = {
        "format": "fly-flight-clips/1",
        "provenance": {
            "work": "Vaxenburg et al. 2025, Nature (flybody); data on Figshare",
            "doi": "10.25378/janelia.25309105.v4", "article": 25309105, "file": "datasets_flight-imitation.zip",
            "members": ["flight-dataset_saccade-evasion_augmented.hdf5", "wing_pattern_fmech.npy"],
            "license": "GPL-3.0-or-later",
            "citation": "Vaxenburg R. et al. Whole-body physics simulation of fruit fly locomotion. Nature (2025); trajectories from "
                        "Muijres F.T. et al. Science 344:172-177 (2014) and J Exp Biol 218:864-875 (2015).",
            "processing": "trajectory type evasion_original only; resampled to 1 kHz (every 5th sample of 5 kHz); position relative to the "
                          "first sample and rotated so the initial body heading is +x; yaw relative to the initial yaw; pitch/roll dropped; "
                          "wing pattern decimated to 100 samples of one wingbeat. Selected by net yaw: 4 straightest, 2 left, 2 right.",
        },
        "position_unit": "dataset length unit, assumed cm (undocumented in the file; implied by ~20 units/s flight speeds)",
        "sample_rate_hz": round(1.0 / (dt * STEP), 3),
        "wing_pattern": {"columns": ["stroke_rad", "deviation_rad", "rotation_rad"], "samples_per_wingbeat": len(pat),
                         "rad": np.round(pat, 4).tolist()},
        "clips": clips,
    }
    out.write_text(json.dumps(doc, separators=(",", ":")))
    print(out, out.stat().st_size, "bytes", [(c["id"], c["kind"], c["net_yaw_deg"], c["duration_s"]) for c in clips])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else OUT)
