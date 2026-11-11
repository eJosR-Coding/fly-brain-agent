"""Silencing screen: silence candidate neurons one by one and rank their effect on a readout.

Shares the experiment conventions (exclusive simulator, progress, chart payload, model-only wording)
with dose_response in experiments.py.
"""

import math
import time

import numpy as np

from flybrain.atlas import Atlas
from flybrain.experiments import MODEL_NOTE, _Progress, _r1
from flybrain.selections import SelectionStore
from flybrain.stats import ALPHA, compare, holm, summarize
from flybrain.tools import (
    ACTIVE_HZ, MAX_DURATION_MS, MAX_RATE_HZ, MAX_TRIALS, MIN_DURATION_MS, MIN_RATE_HZ, SENSORY_CLASSES, SIM_BINS,
    SIM_NOTE, _clamp, _group_label, _int, _label, _number, _readout_trace, _selection_rows, _text, _ticket,
)

MAX_CANDIDATES = 12
DEFAULT_SCREEN_RATE_HZ = 100.0
TOP_NON_SENSORY = "top_non_sensory"


def _candidates(atlas, store, mode, base, stim, readout, n, notes) -> np.ndarray:
    """Candidate rows to silence, busiest in the baseline first, never the stimulus or the readout itself."""
    excluded = np.union1d(stim, readout)
    if mode == TOP_NON_SENSORY:
        super_classes = atlas.neurons["super_class"].to_numpy()
        pool = np.flatnonzero(base.rate_hz > ACTIVE_HZ)
        pool = pool[~np.isin(super_classes[pool], SENSORY_CLASSES)]
    elif isinstance(mode, str) and mode.startswith("sel_"):
        given = _selection_rows(atlas, store, mode)
        pool = given[base.rate_hz[given] > ACTIVE_HZ]
        if len(pool) < len(given):
            notes.append(f"{len(given) - len(pool)} of the {len(given)} candidates did not fire in the baseline and were skipped")
    else:
        raise ValueError(f'candidates must be "{TOP_NON_SENSORY}" or a selection ticket (sel_...), got {mode!r}')
    pool = pool[~np.isin(pool, excluded)]
    return pool[np.argsort(-base.rate_hz[pool], kind="stable")][:n]


def silencing_screen(
    atlas: Atlas, store: SelectionStore, sims, selection_id, readout_selection_id,
    candidates=TOP_NON_SENSORY, n=8, duration_ms=300, trials=3, rate_hz=DEFAULT_SCREEN_RATE_HZ, seed: int = 0,
) -> dict:
    """Silence each candidate neuron in turn and rank them by their effect on the readout.

    One baseline, then one run per candidate (same seed). Welch t-test per candidate on the readout's
    per-trial mean rate; significance uses Holm-corrected p-values across the candidates.
    """
    notes: list[str] = []
    rate = _clamp("rate_hz", _number("rate_hz", rate_hz), MIN_RATE_HZ, MAX_RATE_HZ, notes)
    duration = round(_clamp("duration_ms", _number("duration_ms", duration_ms), MIN_DURATION_MS, MAX_DURATION_MS, notes), 1)
    n_trials = _clamp("trials", _int("trials", trials), 1, MAX_TRIALS, notes)
    n = _clamp("n", _int("n", n), 1, MAX_CANDIDATES, notes)
    stim = _selection_rows(atlas, store, selection_id)
    readout = _selection_rows(atlas, store, readout_selection_id)
    bin_ms = max(1.0, math.ceil(duration / SIM_BINS * 10) / 10)
    stimulus = {"neurons": int(len(stim)), "rate_hz": rate, "duration_ms": duration, "trials": n_trials}
    trace = _readout_trace(atlas, readout, bin_ms, n_trials, store.get(readout_selection_id).description)

    def keep(res, silenced=None, compare_id=None) -> str:
        return sims.add(
            res.active_rows, np.minimum(res.frames, 255).astype(np.uint8), bin_ms, duration, n_trials, stimulus,
            active_count=int((res.rate_hz > ACTIVE_HZ).sum()), compare_id=compare_id, silenced_rows=silenced,
            readout=trace(res),
        )

    t0 = time.perf_counter()
    rows: list[dict] = []
    with sims.exclusive():
        sim = sims.simulation()
        base = sim.run(stim, rate, duration, n_trials, seed=seed, bin_ms=bin_ms)
        picked = _candidates(atlas, store, candidates, base, stim, readout, n, notes)
        base_hz = base.group_trial_hz(readout)
        base_id = keep(base)
        with _Progress(sims, "silencing_screen", len(picked)) as progress:
            for row in picked:
                res = sim.run(stim, rate, duration, n_trials, seed=seed, silence_rows=[int(row)], bin_ms=bin_ms)
                c = compare(base_hz, res.group_trial_hz(readout))
                rows.append({"row": int(row), "c": c, "simulation_id": keep(res, [int(row)], base_id)})
                progress.step()
    runtime = time.perf_counter() - t0

    adjusted = holm([r["c"].p for r in rows])
    results = []
    for r, p_adj in zip(rows, adjusted):
        c = r["c"]
        results.append({
            "label": _text(atlas.neurons["cell_type"].iat[r["row"]]) or "unknown", "row": r["row"],
            "root_id": _label(atlas, [r["row"]])[0]["root_id"],
            "delta_hz": _r1(c.delta), "ci_low": _r1(c.delta_ci_low), "ci_high": _r1(c.delta_ci_high),
            "p": None if c.p is None else float(f"{c.p:.3g}"),
            "p_adj": None if p_adj is None else float(f"{p_adj:.3g}"),
            "significant": p_adj is not None and p_adj < ALPHA, "simulation_id": r["simulation_id"],
        })
    results.sort(key=lambda r: (-abs(r["delta_hz"]), r["label"], r["row"]))
    s = summarize(base_hz)
    out: dict = {
        "experiment": "silencing_screen", "stimulus": stimulus, "candidates": len(results),
        "baseline": {"readout_mean_hz": _r1(s.mean), "ci_low": _r1(s.ci_low), "ci_high": _r1(s.ci_high)},
        "baseline_simulation_id": base_id, "results": results,
        "significant_count": sum(r["significant"] for r in results),
    }
    if results:
        out["candidates_selection_id"] = _ticket(
            atlas, store, [r["row"] for r in results], f"silencing candidates after stimulating {selection_id}",
            f"silencing_screen(selection_id={selection_id!r}, readout_selection_id={readout_selection_id!r})",
        )
        if len(results) < n:
            notes.append(f"only {len(results)} candidate(s) fired in the baseline")
    else:
        out["hint"] = "No candidate neuron fired in the baseline. Raise rate_hz or pass a candidates ticket."
    if n_trials < 2:
        notes.append("need at least 2 trials per condition to test: nothing can be called significant")
    notes.append(
        f"significance: Welch t-test per candidate, Holm-corrected across {len(results)} comparisons (p_adj < 0.05); "
        "silencing = zeroing a neuron's outgoing synapses; same random seeds in baseline and silenced runs"
    )
    out["chart_id"] = sims.add_chart({
        "kind": "screen", "readout_label": _group_label(atlas, readout, store.get(readout_selection_id).description),
        "stimulus_label": _group_label(atlas, stim, store.get(selection_id).description), "trials": n_trials,
        "duration_ms": duration, "baseline_simulation_id": base_id,
        "rows": [{k: r[k] for k in ("label", "row", "delta_hz", "ci_low", "ci_high", "p", "significant", "simulation_id")}
                 for r in results],
    })
    out["runtime_s"] = round(runtime, 1)
    out["note"] = "; ".join([*notes, MODEL_NOTE, SIM_NOTE])
    return out

