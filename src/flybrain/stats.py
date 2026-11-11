"""Small statistics for simulated trials: mean +- 95% CI, Welch's t-test, Cohen's d, Holm correction.

Trials of the simulator differ only by their Poisson input noise (and chaotic divergence), so these
numbers say how reproducible a MODEL result is, not how variable real flies are.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

ALPHA = 0.05


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    sd: float | None  # sample SD (n - 1); None with one trial
    ci_low: float | None  # 95% interval of the mean, Student t; None with one trial
    ci_high: float | None


@dataclass(frozen=True)
class Comparison:
    """`second - first` (e.g. silenced minus baseline), with a Welch t-test."""

    delta: float
    delta_ci_low: float | None
    delta_ci_high: float | None
    t: float | None
    df: float | None
    p: float | None  # two-sided; None when a group has fewer than 2 trials
    cohens_d: float | None  # delta / pooled SD; None when undefined


def _values(values, what: str = "values") -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if not len(arr):
        raise ValueError(f"{what}: need at least one value")
    return arr


def summarize(values) -> Summary:
    arr = _values(values)
    n, mean = len(arr), float(arr.mean())
    if n < 2:
        return Summary(n, mean, None, None, None)
    sd = float(arr.std(ddof=1))
    half = float(sps.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    return Summary(n, mean, sd, mean - half, mean + half)


def compare(first, second) -> Comparison:
    a, b = _values(first, "first"), _values(second, "second")
    delta = float(b.mean() - a.mean())
    if len(a) < 2 or len(b) < 2:
        return Comparison(delta, None, None, None, None, None, None)
    va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
    if va == 0 and vb == 0:  # no spread at all: the means either agree or differ for certain
        same = delta == 0
        return Comparison(delta, delta, delta, None, None, 1.0 if same else 0.0, 0.0 if same else None)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    t = delta / math.sqrt(se2)
    df = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = float(2 * sps.t.sf(abs(t), df))
    half = float(sps.t.ppf(0.975, df)) * math.sqrt(se2)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return Comparison(delta, delta - half, delta + half, t, float(df), p, delta / pooled)


def holm(p_values) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values in the input order (None entries are skipped, not counted)."""
    ps = list(p_values)
    present = sorted((p, i) for i, p in enumerate(ps) if p is not None)
    out: list[float | None] = [None] * len(ps)
    running = 0.0
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, p * (len(present) - rank)))
        out[i] = running
    return out


def verdict(p: float | None, trials: int, tests: int = 1, alpha: float = ALPHA) -> str:
    """A plain sentence. With `tests` > 1, `p` must already be the corrected value."""
    if p is None:
        return "need at least 2 trials per condition to tell a real change from noise"
    if p < alpha:
        text = f"significant at p<0.05 with {trials} trials per condition"
        if tests > 1:
            text += f" after Holm correction across {tests} comparisons"
        return f"{text} (model prediction: Welch t-test on simulated trials, not on living flies)"
    text = f"not distinguishable from noise with {trials} trials per condition"
    if tests > 1:
        text += f" (Holm-corrected across {tests} comparisons)"
    return f"{text}; more trials could still reveal a small effect"
