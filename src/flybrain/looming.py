"""Looming escape experiment: does the model tell a looming disc from a receding one or from dimming?

One eye's photoreceptors (R1-6) are driven inside a disc that follows real looming geometry (eye.py).
Four conditions with EXACTLY equal total input drive (sum over neurons of rate x active time):
looming, the same schedule reversed in time (receding), the whole eye dimmed at one constant rate,
and looming with LPLC2 + LC4 silenced. The readout is the giant fiber DNp01.
Everything is a MODEL prediction: trials differ only by Poisson input noise.
"""

import math
import time

import numpy as np

from flybrain.atlas import Atlas
from flybrain.experiments import MODEL_NOTE, _Progress, _r1
from flybrain.eye import (
    COLLISION_LEAD_MS, FULL_FIELD_DEG, LOOM_L_OVER_V_MS, disc_schedule, eye_map, looming_radii, schedule_drive,
)
from flybrain.simulation import StimulusSchedule
from flybrain.stats import ALPHA, compare, holm, summarize
from flybrain.tools import (
    ACTIVE_HZ, MAX_DURATION_MS, MAX_RATE_HZ, MAX_TRIALS, MIN_DURATION_MS, MIN_RATE_HZ, SIM_BINS, SIM_NOTE,
    _clamp, _int, _number, _readout_trace,
)

MIN_STEPS, MAX_STEPS = 3, 20
CONDITIONS = ("looming", "receding", "dimming", "control")
READOUT, DETECTORS = "DNp01", ("LPLC2", "LC4")
CAVEAT = (
    "photoreceptor signs come from FlyWire's PREDICTED transmitters (real R1-6 are histamine, inhibitory); "
    "no T4/T5 motion-detector dynamics are guaranteed, so looming selectivity is not validated"
)


def _rows(atlas: Atlas, cell_type: str) -> np.ndarray:
    rows = np.flatnonzero((atlas.neurons["cell_type"] == cell_type).to_numpy())
    if not len(rows):
        raise ValueError(f"no {cell_type} neurons in the atlas")
    return rows


def _reversed(schedule: StimulusSchedule, duration: float) -> StimulusSchedule:
    return StimulusSchedule([(round(duration - e, 1), round(duration - s, 1), r, hz) for s, e, r, hz in schedule.segments])


def _dimming(eye_rows, drive: float, duration: float, notes: list[str]) -> tuple[StimulusSchedule, float]:
    """The whole eye at one constant rate with the same total drive; shortened if that rate is under the floor."""
    rate = drive * 1000.0 / (len(eye_rows) * duration)
    if rate >= MIN_RATE_HZ:
        return StimulusSchedule([(0.0, duration, eye_rows, rate)]), rate
    window = max(0.1, round(drive * 1000.0 / (len(eye_rows) * MIN_RATE_HZ), 1))
    notes.append(
        f"dimming at equal drive would need {rate:.2f} Hz, under the {MIN_RATE_HZ:g} Hz floor: drove the whole eye "
        f"at {MIN_RATE_HZ:g} Hz for a shorter {window:g} ms instead"
    )
    return StimulusSchedule([(0.0, window, eye_rows, MIN_RATE_HZ)]), MIN_RATE_HZ


def _verdict(comps: list[dict], trials: int) -> str:
    if any(c["p"] is None for c in comps):
        return "model prediction: need at least 2 trials per condition to tell a real change from noise"
    sig = [c for c in comps if c["significant"]]
    quiet = [c["vs"] for c in comps if not c["significant"]]
    parts = [f"{'more' if c['delta_hz'] > 0 else 'less'} DNp01 than {c['vs']} ({c['delta_hz']:+g} Hz)" for c in sig]
    text = "model prediction (Welch t-test on simulated trials, Holm across 3): looming drives " + "; ".join(parts) if sig \
        else "model prediction: looming is indistinguishable from noise against every other condition"
    if sig and quiet:
        text += f"; indistinguishable from noise with {trials} trials vs {', '.join(quiet)}"
    return text


def looming_escape(
    atlas: Atlas, sims, trials=5, rate_hz=100, duration_ms=500, steps=10, side="right", seed: int = 0,
) -> dict:
    """Looming vs receding vs dimming (equal total drive) vs LPLC2+LC4 silenced, read at DNp01."""
    notes: list[str] = []
    rate = float(_clamp("rate_hz", _number("rate_hz", rate_hz), MIN_RATE_HZ, MAX_RATE_HZ, notes))
    duration = round(_clamp("duration_ms", _number("duration_ms", duration_ms), MIN_DURATION_MS, MAX_DURATION_MS, notes), 1)
    n_trials = _clamp("trials", _int("trials", trials), 1, MAX_TRIALS, notes)
    n_steps = _clamp("steps", _int("steps", steps), MIN_STEPS, MAX_STEPS, notes)
    eye = eye_map(atlas, side)
    readout = _rows(atlas, READOUT)
    detectors = {name: _rows(atlas, name) for name in DETECTORS}
    silenced = np.sort(np.concatenate(list(detectors.values())))

    loom = disc_schedule(eye, looming_radii(duration, n_steps), duration, rate)
    drive = schedule_drive(loom)
    dim, dim_rate = _dimming(eye.rows, drive, duration, notes)
    schedules = {"looming": loom, "receding": _reversed(loom, duration), "dimming": dim, "control": loom}
    bin_ms = max(1.0, math.ceil(duration / SIM_BINS * 10) / 10)
    stimulus = {"neurons": len(eye), "rate_hz": rate, "duration_ms": duration, "trials": n_trials, "steps": n_steps}
    trace = _readout_trace(atlas, readout, bin_ms, n_trials)

    t0 = time.perf_counter()
    results, cached_runs = {}, 0
    with sims.exclusive():
        sim = sims.simulation()
        with _Progress(sims, "looming_escape", len(CONDITIONS)) as progress:
            for name in CONDITIONS:
                res, hit = sims.run_schedule(
                    schedules[name], duration, n_trials, seed=seed, bin_ms=bin_ms,
                    silence_rows=silenced if name == "control" else None,
                )
                cached_runs += hit
                results[name] = res
                progress.step()
        signs = sim.outgoing_signs(eye.rows)
    runtime = time.perf_counter() - t0

    conditions, hz = {}, {}
    for name, res in results.items():
        hz[name] = res.group_trial_hz(readout)
        s = summarize(hz[name])
        conditions[name] = {
            "dnp01_hz": _r1(s.mean), "ci_low": _r1(s.ci_low), "ci_high": _r1(s.ci_high),
            **{f"{d.lower()}_hz": _r1(res.group_trial_hz(rows).mean()) for d, rows in detectors.items()},
            "simulation_id": sims.add(
                res.active_rows, np.minimum(res.frames, 255).astype(np.uint8), bin_ms, duration, n_trials, stimulus,
                active_count=int((res.rate_hz > ACTIVE_HZ).sum()), readout=trace(res),
                silenced_rows=silenced if name == "control" else None,
            ),
        }

    raw = [compare(hz[o], hz["looming"]) for o in CONDITIONS[1:]]  # delta = looming - other
    adjusted = holm([c.p for c in raw])
    comparisons = [
        {"vs": o, "delta_hz": _r1(c.delta), "ci_low": _r1(c.delta_ci_low), "ci_high": _r1(c.delta_ci_high),
         "p": None if c.p is None else float(f"{c.p:.3g}"), "p_adj": None if p is None else float(f"{p:.3g}"),
         "cohens_d": None if c.cohens_d is None else round(c.cohens_d, 2), "significant": p is not None and p < ALPHA}
        for o, c, p in zip(CONDITIONS[1:], raw, adjusted)
    ]
    if n_trials < 2:
        notes.append("need at least 2 trials per condition to test: nothing can be called significant")
    n = len(eye)
    exc, inh = (round(100 * float((signs == s).sum()) / n, 1) for s in (1, -1))
    notes.append(
        f"loom: theta=2*atan(l/v / time-to-collision), l/v={LOOM_L_OVER_V_MS:g} ms, collision {COLLISION_LEAD_MS:g} ms "
        f"after the end, {FULL_FIELD_DEG:g} deg = whole eye; equal drive = sum of rate x active time over photoreceptors"
    )
    notes.append(f"R1-6 here: {exc:g}% excitatory, {inh:g}% inhibitory weights; {CAVEAT}")
    full = np.asarray(trace(results["looming"])["hz"])
    return {
        "experiment": "looming_escape", "stimulus": {**stimulus, "side": side},
        "photoreceptors": {"eye": side, "n": n, "excitatory_pct": exc, "inhibitory_pct": inh},
        "drive": {"input_events": round(drive, 1), "dimming_rate_hz": round(dim_rate, 2)},
        "conditions": conditions, "comparisons": comparisons, "verdict": _verdict(comparisons, n_trials),
        "looming_dnp01_trace_hz": [_r1(c.mean()) for c in np.array_split(full, n_steps)],
        "simulation_id": conditions["looming"]["simulation_id"],
        "runtime_s": round(runtime, 1), "cached_runs": cached_runs,
        "note": "; ".join([*notes, MODEL_NOTE, SIM_NOTE]),
    }
