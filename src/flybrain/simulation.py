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
    trial_counts: np.ndarray | None = None  # (trials, n) int32: spikes per trial, to measure trial-to-trial spread
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


def _steps(ms: float, dt: float, what: str) -> int:
    n = round(ms / dt)
    if n < 1 or abs(n * dt - ms) > 1e-6:
        raise SimulationError(f"{what} must be a positive multiple of {dt} ms, got {ms}")
    return n


class Simulation:
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
        n_steps = _steps(duration_ms, p.dt, "duration_ms")
        bin_steps = _steps(bin_ms, p.dt, "bin_ms") if bin_ms is not None else None
        n_bins = -(-n_steps // bin_steps) if bin_steps else 0
        if n_bins > max_bins:
            raise SimulationError(
                f"at most {max_bins} bins per run, got {n_bins} (bin_ms={bin_ms}, duration_ms={duration_ms}); "
                "use a coarser bin_ms"
            )

        trial_counts = np.zeros((trials, self.n), dtype=np.int32)
        events: list[np.ndarray] = []  # per trial: unique (bin * n + row) keys and their counts
        for t, rng in enumerate(np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(trials)):
            self._trial(stim, rate_hz, n_steps, rng, silenced, trial_counts[t], events if bin_steps else None, bin_steps)
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

    def _trial(self, stim, rate_hz, n_steps, rng, silenced, counts, events, bin_steps) -> None:
        p, n = self.params, self.n
        v = np.full(n, p.v0)
        g = np.zeros(n)
        # Brian2 blocks ticks s+1 .. s+rfc-1 after a spike at tick s (ready again at s+rfc)
        rfc = np.full(n, self._rfc - 1, dtype=np.int16)  # 0 for Poisson targets
        rfc[stim] = 0
        left = np.zeros(n, dtype=np.int16)  # refractory ticks remaining
        kick = p.w_syn * p.f_poi
        p_event = rate_hz * p.dt / 1000.0
        pending = [np.empty(0, dtype=np.int64)] * self._delay  # spikes waiting for their delay
        trial_keys: list[np.ndarray] = []
        for k in range(n_steps):
            ready = left == 0
            left -= ~ready  # one refractory tick used up
            v = np.where(ready, p.v0 + (v - p.v0) * self._decay_v + g * self._drive, v)
            g = np.where(ready, g * self._decay_g, g)
            spikes = np.flatnonzero(ready & (v > p.v_th))
            slot = k % self._delay
            self._deliver(pending[slot], g, ready)  # spikes of tick k - delay arrive now
            pending[slot] = spikes[~silenced[spikes]]
            if len(stim) and p_event > 0:
                v[stim[rng.random(len(stim)) < p_event]] += kick
            if len(spikes):
                v[spikes] = p.v_rst
                g[spikes] = 0.0
                left[spikes] = rfc[spikes]
                counts[spikes] += 1
                if events is not None:
                    trial_keys.append((k // bin_steps) * n + spikes)
        if events is not None and trial_keys:
            events.append(np.concatenate(trial_keys))

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
