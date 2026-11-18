"""Recording -> body clip: DN traces (body_readouts) -> drive (body_drive, OUR mapping) -> physics (body_runner).

Cost: the physics runs at ~8 s of wall time per simulated second on one CPU core, so the clip length is capped at
MAX_BODY_MS (3 s, ~25 s of compute) and at the recording's own duration (the drive is not extrapolated).
"""

from __future__ import annotations

import math

from flybrain import body_drive, body_readouts, body_runner

MAX_BODY_MS = 3000.0
WALL_S_PER_SIM_S = 8  # measured on the development machine (flygym 2.1.1, MuJoCo, CPU)
NOTE = ("The brain recording is the connectome model; the mapping from descending-neuron rates to the walking "
        "controller's drive is OUR choice (not published) and the body is NeuroMechFly, not the connectome. "
        f"Physics costs about {WALL_S_PER_SIM_S} s of compute per simulated second.")


def effective_duration_ms(record, requested: float | None) -> float:
    """The simulated length: the request (default: the recording), never above the recording or MAX_BODY_MS."""
    want = float(record.duration_ms) if requested is None else float(requested)
    if not (math.isfinite(want) and want > 0):
        raise ValueError("duration_ms must be a positive number")
    return min(want, float(record.duration_ms), MAX_BODY_MS)


def body_from_record(record, atlas, duration_ms: float | None = None, seed: int = 0) -> dict:
    """{"clip", "drive", "caption", "note"} for one recording. Raises body_runner.BodyUnavailable without the extra."""
    duration = effective_duration_ms(record, duration_ms)
    readouts = body_readouts.dn_readouts(record, atlas)
    drive = body_drive.drive_from_readouts(readouts, bin_ms=float(record.bin_ms))
    clip = body_runner.run_body(drive, float(record.bin_ms), duration, seed=int(seed))
    summary = drive.as_dict()
    summary["duration_ms"] = duration
    summary["readouts"] = [{k: r[k] for k in ("name", "neurons", "hz")} for r in readouts]
    return {"clip": clip, "drive": summary, "caption": drive.caption, "note": NOTE}
