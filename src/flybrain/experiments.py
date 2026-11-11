"""Experiment bench: dose-response sweeps over the LIF simulator (the silencing screen is in silencing.py).

Both experiments hold the simulator's exclusive lock for the whole experiment (one simulation at a time on a
shared machine), report progress through the service, and keep a small chart payload for the viewer.
Everything they report is a MODEL prediction: trials differ only by Poisson input noise.
"""

import time

from flybrain.atlas import Atlas
from flybrain.selections import SelectionStore
from flybrain.stats import summarize
from flybrain.tools import (
    ACTIVE_HZ, MAX_DURATION_MS, MAX_RATE_HZ, MAX_TRIALS, MIN_DURATION_MS, MIN_RATE_HZ, SIM_NOTE,
    _clamp, _group_label, _int, _number, _selection_rows,
)

DEFAULT_RATES_HZ = (10, 25, 50, 100, 200)
MAX_RATES = 8
MODEL_NOTE = "model prediction (LIF model, Shiu et al. 2024); trials differ only by Poisson input noise"


def half_max_rate(rates, means) -> float | None:
    """Rate where the mean response first reaches half of its maximum, by linear interpolation.

    None when there is no response (max <= 0) or the curve is already above half-max at the lowest
    tested rate (the half-max point lies below the tested range).
    """
    rates, means = [float(r) for r in rates], [float(m) for m in means]
    half = max(means) / 2
    if half <= 0:
        return None
    for i, m in enumerate(means):
        if m >= half:
            if i == 0:
                return None
            r0, r1, m0, m1 = rates[i - 1], rates[i], means[i - 1], m
            return r1 if m1 == m0 else r0 + (half - m0) / (m1 - m0) * (r1 - r0)
    return None


def _rates(rates_hz, notes: list[str]) -> list[float]:
    if isinstance(rates_hz, (str, bytes)) or not isinstance(rates_hz, (list, tuple)) or not rates_hz:
        raise TypeError("rates_hz must be a non-empty list of numbers")
    rates = sorted({_clamp("rate_hz", _number("rate_hz", r), MIN_RATE_HZ, MAX_RATE_HZ, notes) for r in rates_hz})
    if len(rates) > MAX_RATES:
        notes.append(f"at most {MAX_RATES} rates: kept the first {MAX_RATES} of {len(rates)}")
        rates = rates[:MAX_RATES]
    return rates


def _r1(v):
    return None if v is None else round(float(v), 1)


class _Progress:
    """Context manager: reports (done, total) steps for one experiment and always clears the state."""

    def __init__(self, sims, experiment: str, total: int) -> None:
        self.sims, self.experiment, self.total, self.done = sims, experiment, total, 0

    def __enter__(self):
        self.sims.report_progress(self.experiment, 0, self.total)
        return self

    def step(self) -> None:
        self.done += 1
        self.sims.report_progress(self.experiment, self.done, self.total)

    def __exit__(self, *exc) -> None:
        self.sims.finish_progress()


def dose_response(
    atlas: Atlas, store: SelectionStore, sims, selection_id, readout_selection_id,
    rates_hz=DEFAULT_RATES_HZ, duration_ms=300, trials=3, seed: int = 0,
) -> dict:
    """Readout rate (mean +- 95% CI over trials) as the stimulus rate sweeps, plus the half-max rate."""
    notes: list[str] = []
    rates = _rates(list(rates_hz) if isinstance(rates_hz, tuple) else rates_hz, notes)
    duration = round(_clamp("duration_ms", _number("duration_ms", duration_ms), MIN_DURATION_MS, MAX_DURATION_MS, notes), 1)
    n_trials = _clamp("trials", _int("trials", trials), 1, MAX_TRIALS, notes)
    stim = _selection_rows(atlas, store, selection_id)
    readout = _selection_rows(atlas, store, readout_selection_id)

    points, t0, clipped, cached_runs = [], time.perf_counter(), False, 0
    with sims.exclusive():
        sims.simulation()
        with _Progress(sims, "dose_response", len(rates)) as progress:
            for rate in rates:
                res, hit = sims.run(stim, rate, duration, n_trials, seed=seed)
                cached_runs += hit
                s = summarize(res.group_trial_hz(readout))
                low = None if s.ci_low is None else max(0.0, s.ci_low)  # a rate cannot be negative
                clipped = clipped or (s.ci_low is not None and s.ci_low < 0)
                points.append({
                    "rate_hz": rate, "readout_mean_hz": _r1(s.mean), "ci_low": _r1(low), "ci_high": _r1(s.ci_high),
                    "active_neurons": int((res.rate_hz > ACTIVE_HZ).sum()),
                })
                progress.step()
    runtime = time.perf_counter() - t0

    ec50 = half_max_rate([p["rate_hz"] for p in points], [p["readout_mean_hz"] for p in points])
    out: dict = {
        "experiment": "dose_response",
        "stimulus": {"neurons": int(len(stim)), "duration_ms": duration, "trials": n_trials},
        "points": points,
        "ec50_hz": None if ec50 is None else round(ec50, 1),
    }
    means = [p["readout_mean_hz"] for p in points]
    if ec50 is None:
        out["ec50_note"] = (
            "no response at any tested rate" if max(means) <= 0
            else f"already above half of its maximum at {points[0]['rate_hz']:g} Hz: the half-max rate lies below the tested range"
        )
    elif len(points) > 1 and means[-1] > 1.1 * means[-2]:
        out["ec50_note"] = "still rising at the highest rate: the half-max rate is relative to the highest tested rate"
    if clipped:
        notes.append("interval lower bounds clipped at 0 Hz (a rate cannot be negative)")
    if n_trials < 2:
        notes.append("1 trial per rate: no confidence intervals")
    label = _group_label(atlas, readout, store.get(readout_selection_id).description)
    stim_label = _group_label(atlas, stim, store.get(selection_id).description)
    out["chart_id"] = sims.add_chart({
        "kind": "dose_response", "readout_label": label, "stimulus_label": stim_label, "trials": n_trials,
        "duration_ms": duration, "ec50_hz": out["ec50_hz"],
        "points": [{"rate_hz": p["rate_hz"], "mean_hz": p["readout_mean_hz"], "ci_low": p["ci_low"], "ci_high": p["ci_high"]}
                   for p in points],
    })
    out["runtime_s"] = round(runtime, 1)
    out["cached_runs"] = cached_runs
    if cached_runs:
        notes.append(f"{cached_runs} of {len(rates)} rate steps came from the simulation cache")
    out["note"] = "; ".join([*notes, MODEL_NOTE, SIM_NOTE])
    return out
