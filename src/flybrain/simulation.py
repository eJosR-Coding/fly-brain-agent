"""Whole-brain leaky integrate-and-fire simulator (Shiu et al., Nature 2024 parameters).

A seeded, vectorised numpy re-implementation of the published Brian2 model
(github.com/philshiu/Drosophila_brain_model, MIT), matching it step for step:

    dv/dt = (v0 - v + g) / t_mbr,   dg/dt = -g / tau      (both frozen while refractory)
    v > v_th  ->  spike: v = v_rst, g = 0, refractory 2.2 ms (none for Poisson-driven neurons)
    a spike adds  weight * w_syn  to the postsynaptic g after a 1.8 ms delay
    a Poisson input at `rate` adds w_syn * 250 to v of a stimulated neuron

Per clock tick (dt = 0.1 ms, exact solution of the linear system, like Brian's method="linear"):
update state of non-refractory neurons -> threshold -> synaptic arrivals + Poisson kicks -> reset
of the neurons that spiked. Reset comes last, so an arrival or kick on a spiking tick is wiped,
exactly as in Brian2.

This is a model of the wiring diagram with fixed parameters: no neuromodulation, no plasticity,
no ventral nerve cord. It is not the living fly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from flybrain.atlas import RowIndex


class SimulationError(ValueError):
    """The simulation inputs are unusable."""


@dataclass(frozen=True)
class LIFParams:
    v0: float = -52.0  # mV, resting potential
    v_rst: float = -52.0  # mV, reset potential
    v_th: float = -45.0  # mV, threshold
    t_mbr: float = 20.0  # ms, membrane time constant
    tau: float = 5.0  # ms, synaptic (g) time constant
    t_rfc: float = 2.2  # ms, refractory period
    t_dly: float = 1.8  # ms, synaptic delay
    w_syn: float = 0.275  # mV per synapse
    f_poi: float = 250.0  # Poisson kick = w_syn * f_poi
    dt: float = 0.1  # ms, clock (Brian2 default)


MAX_SCHEDULE_RATE_HZ = 10_000.0  # one kick per 0.1 ms tick; tools clamp tighter (T3)
MAX_BINS = 200  # time bins per run; a finer request is refused instead of silently blown up


@dataclass
class SimResult:
    """Spike counts summed over trials, plus optional SPARSE time bins (also summed over trials).

    Only neurons that spiked are kept: `active_rows[i]` owns `frames[i]`, spike counts per bin.
    """

    counts: np.ndarray  # (n,) int64
    rate_hz: np.ndarray  # (n,) mean rate per trial
    trials: int
    duration_ms: float
    trial_counts: np.ndarray  # (trials, n) int32: spikes per trial, to measure trial-to-trial spread (always set by _execute)
    bin_ms: float | None = None
    active_rows: np.ndarray | None = None  # (n_active,) int64, sorted
    frames: np.ndarray | None = None  # (n_active, n_bins) uint16

    @property
    def trial_rate_hz(self) -> np.ndarray:
        """(trials, n) rate of every neuron in every trial, Hz."""
        return self.trial_counts / (self.duration_ms / 1000.0)

    def group_trial_hz(self, rows) -> np.ndarray:
        """(trials,) mean rate of a neuron group in each trial: the unit the statistics work on."""
        idx = np.asarray(list(rows), dtype=np.int64)
        if not len(idx):
            return np.zeros(self.trials)
        return self.trial_counts[:, idx].mean(axis=1) / (self.duration_ms / 1000.0)

    @property
    def n_bins(self) -> int | None:
        return None if self.frames is None else self.frames.shape[1]

    @property
    def population(self) -> np.ndarray | None:
        """Spikes per time bin over the whole brain (None without bin_ms)."""
        return None if self.frames is None else self.frames.sum(axis=0, dtype=np.int64)

    @property
    def binned_dense_bytes(self) -> int | None:
        """What a dense (n_bins x n) int32 matrix would have cost: why frames are sparse."""
        return None if self.frames is None else self.frames.shape[1] * len(self.counts) * 4

    @property
    def binned(self) -> np.ndarray | None:
        """Dense (n_bins, n) view for tiny networks and tests. Allocates n_bins * n * 4 bytes."""
        if self.frames is None:
            return None
        dense = np.zeros((self.frames.shape[1], len(self.counts)), dtype=np.int32)
        dense[:, self.active_rows] = self.frames.T
        return dense


@dataclass
class ChunkSpikes:
    """Spikes of ONE live chunk (a `BrainStream.advance`): only neurons that spiked, one trial."""

    active_rows: np.ndarray  # (k,) int64, sorted
    counts: np.ndarray  # (k,) int32, spikes of each active row inside the chunk
    duration_ms: float

    def group_hz(self, rows) -> float:
        """Mean rate (Hz per neuron) of a neuron group over the chunk; rows that never spiked count as silent."""
        idx = np.asarray(list(rows), dtype=np.int64)
        if not len(idx) or not self.duration_ms > 0:
            return 0.0
        spikes = int(self.counts[np.isin(self.active_rows, idx)].sum())
        return spikes / (len(idx) * self.duration_ms / 1000.0)


class _TrialState:
    """Everything one trial carries from tick to tick: membrane, synaptic g, refractory counters, the delay ring."""

    def __init__(self, sim: "Simulation", stim, rng) -> None:
        n, p = sim.n, sim.params
        self.rng = rng
        self.k = 0  # next tick to run (absolute: ring slot and epoch bounds are relative to it)
        self.v = np.full(n, p.v0)
        self.g = np.zeros(n)
        self.rfc = np.empty(n, dtype=np.int16)
        self.set_poisson_targets(sim, stim)
        self.left = np.zeros(n, dtype=np.int16)  # refractory ticks remaining
        self.pending = [np.empty(0, dtype=np.int64)] * sim._delay  # spikes waiting for their delay

    def set_poisson_targets(self, sim: "Simulation", stim) -> None:
        # Brian2 blocks ticks s+1 .. s+rfc-1 after a spike at tick s (ready again at s+rfc)
        self.rfc[:] = sim._rfc - 1
        self.rfc[stim] = 0  # Poisson targets have no refractory period


class StimulusSchedule:
    """Piecewise-constant Poisson drive: segments `(start_ms, end_ms, rows, rate_hz)`.

    A neuron is driven at `rate_hz` while `start_ms <= t < end_ms`. Segments may overlap and
    rates for the same neuron then ADD UP (two 150 Hz segments = one 300 Hz drive). Segment
    order, row order and duplicate rows do not matter: the canonical form (`key`) sorts them, so
    equal schedules compare, hash and cache the same. Bounds against a simulator (row range,
    duration, dt multiples) are checked in `Simulation.run_schedule`.
    """

    __slots__ = ("segments",)

    def __init__(self, segments) -> None:
        canon = []
        for seg in segments:
            try:
                start, end, rows, rate = seg
                start, end, rate = float(start), float(end), float(rate)
                rows = np.unique(np.asarray(list(rows), dtype=np.int64))
            except (TypeError, ValueError):
                raise SimulationError(f"segment must be (start_ms, end_ms, rows, rate_hz), got {seg!r}") from None
            if not np.isfinite(start) or start < 0:
                raise SimulationError(f"segment start_ms must be >= 0, got {start}")
            if not np.isfinite(end) or not start < end:
                raise SimulationError(f"segment needs start_ms < end_ms, got {start} and {end}")
            if not len(rows):
                raise SimulationError("segment rows must not be empty")
            if rows[0] < 0:
                raise SimulationError(f"segment rows must be >= 0, got {int(rows[0])}")
            if not np.isfinite(rate) or not 0 <= rate <= MAX_SCHEDULE_RATE_HZ:
                raise SimulationError(f"segment rate_hz must be within 0..{MAX_SCHEDULE_RATE_HZ:g}, got {rate}")
            canon.append((start, end, tuple(int(r) for r in rows), rate))
        if not canon:
            raise SimulationError("a schedule needs at least one segment")
        self.segments = tuple(sorted(canon))

    @property
    def key(self) -> tuple:
        """Canonical hashable form (used in cache keys)."""
        return self.segments

    @property
    def end_ms(self) -> float:
        return max(seg[1] for seg in self.segments)

    def __eq__(self, other) -> bool:
        return isinstance(other, StimulusSchedule) and self.segments == other.segments

    def __hash__(self) -> int:
        return hash(self.segments)

    def __repr__(self) -> str:
        return f"StimulusSchedule({len(self.segments)} segments, to {self.end_ms:g} ms)"


def _steps(ms: float, dt: float, what: str) -> int:
    n = round(ms / dt)
    if n < 1 or abs(n * dt - ms) > 1e-6:
        raise SimulationError(f"{what} must be a positive multiple of {dt} ms, got {ms}")
    return n


class Simulation:
    backend = "cpu"  # GPUSimulation overrides it

    def __init__(self, csr: sparse.csr_matrix, params: LIFParams) -> None:
        self.n = csr.shape[0]
        self.params = params
        self._indptr = csr.indptr
        self._indices = csr.indices
        self._counts = csr.data  # signed synapse counts, float32 (exact up to 2**24)
        self.n_edges = int(csr.nnz)
        self.dropped_edges = 0
        self.dropped_neurons = 0
        p = params
        self._delay = _steps(p.t_dly, p.dt, "t_dly")
        self._rfc = _steps(p.t_rfc, p.dt, "t_rfc")
        self._decay_v = np.exp(-p.dt / p.t_mbr)
        self._decay_g = np.exp(-p.dt / p.tau)
        # exact integral of the g drive over one step (solution of the coupled linear ODEs)
        self._drive = (self._decay_g - self._decay_v) / (1.0 - p.t_mbr / p.tau)

    @property
    def fingerprint(self) -> tuple:
        """Cheap identity of the loaded weights + params: a different matrix never shares cached runs."""
        if getattr(self, "_fp", None) is None:
            self._fp = (
                self.n, self.n_edges, self.params,
                float(self._counts.sum(dtype=np.float64)),
                int(self._indices.sum(dtype=np.int64)),
                int(self._indptr.sum(dtype=np.int64)),
            )
        return self._fp

    # -- construction ------------------------------------------------------------------

    @classmethod
    def from_arrays(cls, pre, post, weights, n_neurons: int, params: LIFParams | None = None) -> "Simulation":
        """Build from edges in atlas rows; `weights` are SIGNED synapse counts (negative = inhibitory)."""
        pre, post = np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float32)
        if not (len(pre) == len(post) == len(weights)):
            raise SimulationError("pre, post and weights must have the same length")
        if len(pre) and (min(pre.min(), post.min()) < 0 or max(pre.max(), post.max()) >= n_neurons):
            raise SimulationError(f"edge rows out of range 0..{n_neurons - 1}")
        csr = sparse.csr_matrix((weights, (pre, post)), shape=(n_neurons, n_neurons), dtype=np.float32)
        csr.sum_duplicates()
        csr.eliminate_zeros()
        return cls(csr, params or LIFParams())

    @classmethod
    def from_files(cls, connectivity, completeness, atlas, params: LIFParams | None = None) -> "Simulation":
        """Build from the reference repo files (Connectivity_*.parquet, Completeness_*.csv).

        The parquet's Presynaptic_Index / Postsynaptic_Index are positions in the completeness
        CSV (its index column holds the root ids); those are mapped to atlas rows and edges with
        an endpoint missing from the atlas are dropped (counted in `dropped_edges`).
        """
        comp = pd.read_csv(Path(completeness), index_col=0)
        rows = RowIndex(atlas).positions(comp.index.to_numpy(dtype="int64"))
        con = pd.read_parquet(
            Path(connectivity), columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
        )
        pre = rows[con["Presynaptic_Index"].to_numpy()]
        post = rows[con["Postsynaptic_Index"].to_numpy()]
        weights = con["Excitatory x Connectivity"].to_numpy()
        del con
        known = (pre >= 0) & (post >= 0)
        sim = cls.from_arrays(pre[known], post[known], weights[known], len(atlas.neurons), params)
        sim.dropped_edges = int((~known).sum())
        sim.dropped_neurons = int((rows < 0).sum())
        return sim

    @staticmethod
    def _signature(connectivity, completeness, n_neurons: int) -> np.ndarray:
        """What the cache was built from: both source sizes + the atlas size (cheap, no hashing)."""
        return np.array(
            [Path(connectivity).stat().st_size, Path(completeness).stat().st_size, n_neurons], dtype=np.int64
        )

    @classmethod
    def load_weights(cls, connectivity, completeness, atlas, cache=None, params: LIFParams | None = None):
        """`from_files` with a CSR cache (an .npz): returns (simulation, cache_hit).

        The cache is used only when its signature (source file sizes, atlas size) matches; an
        unreadable or stale cache is rebuilt and overwritten. The cached CSR is exactly what
        `from_files` builds, so results do not depend on which path was taken.
        """
        n = len(atlas.neurons)
        if cache is not None and Path(cache).is_file() and Path(connectivity).is_file():
            try:
                with np.load(cache) as z:
                    if np.array_equal(z["signature"], cls._signature(connectivity, completeness, n)):
                        csr = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
                        sim = cls(csr, params or LIFParams())
                        sim.dropped_edges, sim.dropped_neurons = (int(x) for x in z["dropped"])
                        return sim, True
            except (OSError, ValueError, KeyError):
                pass  # unreadable or foreign file: fall through and rebuild
        sim = cls.from_files(connectivity, completeness, atlas, params)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(cache) + ".tmp.npz")
            np.savez(
                tmp, data=sim._counts, indices=sim._indices, indptr=sim._indptr,
                dropped=np.array([sim.dropped_edges, sim.dropped_neurons], dtype=np.int64),
                signature=cls._signature(connectivity, completeness, n),
            )
            tmp.replace(cache)
        return sim, False

    def _csr(self) -> sparse.csr_matrix:
        return sparse.csr_matrix((self._counts, self._indices, self._indptr), shape=(self.n, self.n))

    def outgoing_signs(self, rows) -> np.ndarray:
        """Net sign of each row's outgoing weights: +1 excitatory, -1 inhibitory, 0 no outgoing synapses."""
        idx = self._rows(rows, "sign")
        return np.sign(np.asarray(self._csr()[idx].sum(axis=1)).ravel()).astype(np.int8)

    # -- running -----------------------------------------------------------------------

    def _rows(self, rows, what: str) -> np.ndarray:
        arr = np.unique(np.asarray(list(rows) if rows is not None else [], dtype=np.int64))
        if len(arr) and (arr[0] < 0 or arr[-1] >= self.n):
            raise SimulationError(f"{what} row out of range 0..{self.n - 1}")
        return arr

    def run(self, stim_rows, rate_hz: float, duration_ms: float = 1000.0, trials: int = 1,
            seed: int = 0, silence_rows=None, bin_ms: float | None = None,
            max_bins: int = MAX_BINS) -> SimResult:
        """Poisson-drive `stim_rows` at `rate_hz` and count every neuron's spikes.

        Trials are independent (own RNG stream spawned from `seed`), run serially. Silencing
        zeroes the OUTGOING weights of those rows (they still spike themselves). With `bin_ms`
        the result also carries sparse per-bin counts (only neurons that spiked), at most
        `max_bins` bins.
        """
        p = self.params
        stim = self._rows(stim_rows, "stimulus")
        silenced = np.zeros(self.n, dtype=bool)
        silenced[self._rows(silence_rows, "silenced")] = True
        if not np.isfinite(rate_hz) or not 0 <= rate_hz <= 1000.0 / p.dt:
            raise SimulationError(f"rate_hz must be within 0..{1000.0 / p.dt:g}, got {rate_hz}")
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
            raise SimulationError(f"trials must be an int >= 1, got {trials!r}")
        n_steps, bin_steps, n_bins = self._timeline(duration_ms, bin_ms, max_bins)
        p_event = rate_hz * p.dt / 1000.0
        epochs = [(0, n_steps, stim, np.full(len(stim), p_event))] if len(stim) and p_event > 0 else []
        return self._execute(stim, epochs, n_steps, trials, seed, silenced, bin_steps, n_bins, bin_ms, duration_ms)

    def run_schedule(self, schedule: StimulusSchedule, duration_ms: float = 1000.0, trials: int = 1,
                     seed: int = 0, silence_rows=None, bin_ms: float | None = None,
                     max_bins: int = MAX_BINS) -> SimResult:
        """Like `run`, but the Poisson drive follows `schedule` (per-neuron, per-window rates).

        Each segment must fit in `duration_ms` and sit on dt multiples. Per tick only the neurons
        whose windows are open draw random numbers (precomputed active sets, one per stretch of
        ticks with the same drive). A one-segment full-length schedule gives bit-identical
        results to `run` with the same seed. Every row named by a segment is a Poisson target
        (no refractory period), even outside its window, like `stim_rows` in `run`.
        """
        p = self.params
        silenced = np.zeros(self.n, dtype=bool)
        silenced[self._rows(silence_rows, "silenced")] = True
        if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
            raise SimulationError(f"trials must be an int >= 1, got {trials!r}")
        n_steps, bin_steps, n_bins = self._timeline(duration_ms, bin_ms, max_bins)
        if schedule.end_ms > duration_ms + 1e-9:
            raise SimulationError(f"schedule runs to {schedule.end_ms:g} ms, past duration_ms={duration_ms:g}")
        segs = []  # (first step, end step, rows, rate)
        for start, end, rows, rate in schedule.segments:
            a, b = round(start / p.dt), round(end / p.dt)
            if abs(a * p.dt - start) > 1e-6 or abs(b * p.dt - end) > 1e-6:
                raise SimulationError(f"segment times must be multiples of {p.dt} ms, got {start}..{end}")
            segs.append((a, b, self._rows(rows, "stimulus"), rate))
        stim = np.unique(np.concatenate([s[2] for s in segs]))
        cuts = sorted({0, n_steps} | {x for s in segs for x in s[:2] if x <= n_steps})
        epochs = []
        for a, b in zip(cuts, cuts[1:]):
            rate = np.zeros(self.n)
            for sa, sb, rows, r in segs:
                if sa <= a and b <= sb:
                    rate[rows] += r
            on = np.flatnonzero(rate > 0)
            if len(on):
                epochs.append((a, b, on, rate[on] * p.dt / 1000.0))
        return self._execute(stim, epochs, n_steps, trials, seed, silenced, bin_steps, n_bins, bin_ms, duration_ms)

    def _timeline(self, duration_ms, bin_ms, max_bins) -> tuple[int, int | None, int]:
        p = self.params
        n_steps = _steps(duration_ms, p.dt, "duration_ms")
        bin_steps = _steps(bin_ms, p.dt, "bin_ms") if bin_ms is not None else None
        n_bins = -(-n_steps // bin_steps) if bin_steps else 0
        if n_bins > max_bins:
            raise SimulationError(
                f"at most {max_bins} bins per run, got {n_bins} (bin_ms={bin_ms}, duration_ms={duration_ms}); "
                "use a coarser bin_ms"
            )
        return n_steps, bin_steps, n_bins

    def _execute(self, stim, epochs, n_steps, trials, seed, silenced, bin_steps, n_bins, bin_ms, duration_ms):
        """Run the trials. `epochs` = [(first step, end step, rows, per-tick event prob per row)]."""
        trial_counts = np.zeros((trials, self.n), dtype=np.int32)
        events: list[np.ndarray] = []  # per trial: unique (bin * n + row) keys and their counts
        for t, rng in enumerate(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(trials)):
            self._trial(stim, epochs, n_steps, rng, silenced, trial_counts[t], events if bin_steps else None, bin_steps)
        counts = trial_counts.sum(axis=0, dtype=np.int64)
        active = frames = None
        if bin_steps:
            active, frames = self._sparse_frames(events, n_bins)
        return SimResult(
            counts=counts, rate_hz=counts / (trials * duration_ms / 1000.0), trials=trials,
            duration_ms=float(duration_ms), trial_counts=trial_counts, bin_ms=bin_ms,
            active_rows=active, frames=frames,
        )

    def _sparse_frames(self, events: list[np.ndarray], n_bins: int) -> tuple[np.ndarray, np.ndarray]:
        """(sorted active rows, (n_active, n_bins) uint16) from the per-tick (bin * n + row) keys."""
        keys = np.concatenate(events) if events else np.empty(0, dtype=np.int64)
        uniq, cnt = np.unique(keys, return_counts=True)
        bins, rows = uniq // self.n, uniq % self.n
        active, slot = np.unique(rows, return_inverse=True)
        frames = np.zeros((len(active), n_bins), dtype=np.uint16)
        frames[slot, bins] = np.minimum(cnt, np.iinfo(np.uint16).max)
        return active, frames

    def _trial(self, stim, epochs, n_steps, rng, silenced, counts, events, bin_steps) -> None:
        state = _TrialState(self, stim, rng)
        trial_keys: list[np.ndarray] = []
        self._advance(state, epochs, n_steps, silenced, counts, trial_keys if events is not None else None, bin_steps)
        if events is not None and trial_keys:
            events.append(np.concatenate(trial_keys))

    def _advance(self, state: _TrialState, epochs, k_end: int, silenced, counts, trial_keys, bin_steps) -> None:
        """Run ticks `state.k .. k_end - 1` (the one loop of every run AND of every live chunk).

        `epochs` = [(first tick, end tick, rows, per-tick event prob)] in absolute ticks, consumed front to back.
        Spike counts add into `counts`; with `trial_keys` the (bin * n + row) key of every spike is appended."""
        p, n = self.params, self.n
        v, g, left, rfc, rng, pending = state.v, state.g, state.left, state.rfc, state.rng, state.pending
        kick = p.w_syn * p.f_poi
        epochs = list(epochs)
        epoch = 0
        for k in range(state.k, k_end):
            ready = left == 0
            left -= ~ready  # one refractory tick used up
            v = np.where(ready, p.v0 + (v - p.v0) * self._decay_v + g * self._drive, v)
            g = np.where(ready, g * self._decay_g, g)
            spikes = np.flatnonzero(ready & (v > p.v_th))
            slot = k % self._delay
            self._deliver(pending[slot], g, ready)  # spikes of tick k - delay arrive now
            pending[slot] = spikes[~silenced[spikes]]
            while epoch < len(epochs) and epochs[epoch][1] <= k:
                epoch += 1
            if epoch < len(epochs) and epochs[epoch][0] <= k:
                _, _, rows, p_ev = epochs[epoch]
                v[rows[rng.random(len(rows)) < p_ev]] += kick
            if len(spikes):
                v[spikes] = p.v_rst
                g[spikes] = 0.0
                left[spikes] = rfc[spikes]
                counts[spikes] += 1
                if trial_keys is not None:
                    trial_keys.append((k // bin_steps) * n + spikes)
        state.v, state.g, state.k = v, g, max(state.k, k_end)

    def open_stream(self, seed: int = 0, silenced=None) -> "BrainStream":
        """A live session over this brain: state kept between `advance` calls (1 trial). See BrainStream."""
        return BrainStream(self, seed, silenced)

    def _deliver(self, src: np.ndarray, g: np.ndarray, ready: np.ndarray) -> None:
        """g[post] += weight * w_syn for every outgoing edge of the neurons in `src`.

        Refractory targets ignore the arrival (their g is clamped, as in Brian2).
        """
        if not len(src):
            return
        starts = self._indptr[src]
        cnt = self._indptr[src + 1] - starts
        total = int(cnt.sum())
        if not total:
            return
        offs = np.repeat(starts - (np.cumsum(cnt) - cnt), cnt) + np.arange(total)
        post = self._indices[offs]
        w = self._counts[offs] * self.params.w_syn
        live = ready[post]
        np.add.at(g, post[live], w[live])


class BrainStream:
    """The brain as a live process: `set_stimulus` any time, `advance(ms)` returns the chunk's spikes.

    It owns ONE trial state and runs the very same tick loop as `Simulation.run` (`_advance`), so a stream with a
    constant stimulus is bit-identical to a one-shot run with the same seed. Not thread-safe: one caller at a time.
    Rows with a stimulus are Poisson targets (no refractory period); when the stimulus changes, only the current
    rows are, as in `run`. Silencing zeroes outgoing weights of `silenced` rows for the whole session."""

    backend = "cpu"

    def __init__(self, sim: Simulation, seed: int = 0, silenced=None) -> None:
        self.sim = sim
        self._silenced = np.zeros(sim.n, dtype=bool)
        self._silenced[sim._rows(silenced, "silenced")] = True
        self._state = _TrialState(sim, np.empty(0, dtype=np.int64), np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0]))
        self._rows = np.empty(0, dtype=np.int64)
        self._p_rows = np.empty(0)  # event probability per tick of each stimulated row (aligned with _rows)
        self.stimulus: tuple[np.ndarray, float] | None = None

    @property
    def t_ms(self) -> float:
        return self._state.k * self.sim.params.dt

    def set_stimulus(self, rows, rate_hz: float) -> None:
        """Poisson-drive `rows` at `rate_hz` from now on (None, no rows or 0 Hz = no drive). Validated first."""
        p = self.sim.params
        arr = self.sim._rows(rows, "stimulus")
        if not np.isfinite(rate_hz) or not 0 <= rate_hz <= 1000.0 / p.dt:
            raise SimulationError(f"rate_hz must be within 0..{1000.0 / p.dt:g}, got {rate_hz}")
        self._rows, self._p_rows = arr, np.full(len(arr), rate_hz * p.dt / 1000.0)
        self.stimulus = (arr, float(rate_hz)) if len(arr) and rate_hz > 0 else None
        self._on_stimulus(arr)

    def set_rates(self, rows, rates_hz) -> None:
        """Poisson-drive `rows` at their OWN rates (one per row, rows unique) from now on; zero-rate rows are dropped.

        `stimulus` then reports the driven rows with their mean rate. Validated first: the stream is untouched on error."""
        p = self.sim.params
        rows_in = np.asarray(list(rows) if rows is not None else [], dtype=np.int64)
        rates = np.asarray(list(rates_hz) if rates_hz is not None else [], dtype=np.float64)
        if rows_in.shape != rates.shape or rows_in.ndim != 1:
            raise SimulationError("rates need exactly one value per row")
        if len(np.unique(rows_in)) != len(rows_in):
            raise SimulationError("rows must be unique when each has its own rate")
        if not np.isfinite(rates).all() or (rates < 0).any() or (rates > 1000.0 / p.dt).any():
            raise SimulationError(f"every rate must be within 0..{1000.0 / p.dt:g} Hz")
        self.sim._rows(rows_in, "stimulus")  # range check
        keep = rates > 0
        order = np.argsort(rows_in[keep], kind="stable")
        arr, hz = rows_in[keep][order], rates[keep][order]
        self._rows, self._p_rows = arr, hz * p.dt / 1000.0
        self.stimulus = (arr, float(hz.mean())) if len(arr) else None
        self._on_stimulus(arr)

    def _on_stimulus(self, rows: np.ndarray) -> None:
        self._state.set_poisson_targets(self.sim, rows)

    def advance(self, ms: float) -> ChunkSpikes:
        """Simulate the next `ms` (a multiple of dt) and return the chunk's spike counts."""
        sim, st = self.sim, self._state
        n_steps = _steps(ms, sim.params.dt, "advance ms")
        counts = np.zeros(sim.n, dtype=np.int32)
        epochs = [(st.k, st.k + n_steps, self._rows, self._p_rows)] if self.stimulus else []
        sim._advance(st, epochs, st.k + n_steps, self._silenced, counts, None, None)
        active = np.flatnonzero(counts).astype(np.int64)
        return ChunkSpikes(active, counts[active], float(ms))

    def close(self) -> None:
        """Nothing to free on the CPU (the GPU stream releases its device state)."""
