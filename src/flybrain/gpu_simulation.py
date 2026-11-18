"""CUDA (CuPy) backend of the whole-brain LIF simulator: same model, same API, same results shape.

`GPUSimulation` subclasses `Simulation` and replaces only the trial loop (`_execute`); input
validation, the timeline, stimulus epochs and the SimResult layout are the CPU code, so the two
backends cannot drift apart on the contract. The numerics are re-implemented in two kernels per
0.1 ms tick, with trials as columns (state is `n x trials`, float32):

    update   exact-decay v/g update (frozen while refractory), threshold, synaptic arrivals of
             this tick, Poisson kicks, reset. One thread per (neuron, trial).
    deliver  every non-silenced spike of the tick is scattered along its CSR row into the slot of
             an 18-tick delay ring buffer (atomicAdd). Targets that are refractory when the
             arrival is consumed ignore it, and a neuron that spikes on that tick has its g wiped by
             the reset, exactly as in the CPU simulator.

Poisson draws use a counter-based hash of (seed, step, neuron, trial), so no RNG state is kept and
trials are independent streams. The streams differ from numpy's: CPU and GPU agree statistically,
never bit for bit (the cache key of the service therefore includes the backend).

GPU SAFETY (the card is shared with the desktop, 4 GB on the target laptop): CuPy never gets the
whole card. Allocations go through CuPy's pool with an explicit limit (`POOL_LIMIT_MB`), a run
that would not fit raises `GPUCapacityError` (the service then runs it on the CPU), the pool is
released after every run, and `probe_gpu` refuses a card with less than `MIN_FREE_VRAM_MB` free.
CuPy is imported lazily: the base install does not need it (`uv sync --extra gpu`).
"""

import importlib
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flybrain.simulation import Simulation, SimResult

log = logging.getLogger(__name__)

# Free VRAM required to even try the GPU: pool limit + the CUDA context (~300-400 MiB) + headroom
# for the desktop compositor and browser that share the card.
MIN_FREE_VRAM_MB = 1536
POOL_LIMIT_MB = 1024  # hard cap on CuPy's memory pool (CSR + one run's state)
MAX_SPIKES_PER_STEP = 262_144  # spike event buffer capacity per tick; beyond it the run raises
DELIVER_BLOCKS = 2048  # grid-stride blocks of the deliver kernel (spike count is read on the device)
CHECK_EVERY = 500  # steps between overflow checks (a device sync)
ARRIVAL_SLOTS = 18  # ring slots; must equal t_dly / dt (checked)


class GPUUnavailable(RuntimeError):
    """The GPU cannot be used here (no cupy, no device, too little VRAM)."""


class GPUCapacityError(RuntimeError):
    """This run does not fit the pool limit or the VRAM left: the caller should run it on the CPU."""


class GPUOverflowError(RuntimeError):
    """More spikes in one tick than the event buffer holds: the run is invalid and is never returned."""


@dataclass(frozen=True)
class GPUProbe:
    usable: bool
    reason: str  # one line: why the GPU is (not) used
    name: str | None = None
    free_mb: float | None = None
    total_mb: float | None = None


def probe_gpu(min_free_mb: float = MIN_FREE_VRAM_MB, loader: Callable | None = None) -> GPUProbe:
    """Can the GPU be used now? Never raises. `loader` returns the cupy module (injectable for tests)."""
    load = loader or (lambda: importlib.import_module("cupy"))
    try:
        cp = load()
    except ImportError as exc:
        return GPUProbe(False, f"cupy is not installed ({exc}); install the `gpu` extra")
    except Exception as exc:  # noqa: BLE001 - a broken CUDA install must not crash the app
        return GPUProbe(False, f"cupy failed to import: {exc}")
    try:
        runtime = cp.cuda.runtime
        if runtime.getDeviceCount() < 1:
            return GPUProbe(False, "no CUDA device found")
        free, total = (b / 2**20 for b in runtime.memGetInfo())
        raw = runtime.getDeviceProperties(0)["name"]
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception as exc:  # noqa: BLE001
        return GPUProbe(False, f"CUDA query failed: {exc}")
    if free < min_free_mb:
        return GPUProbe(False, f"only {free:.0f} MiB VRAM free (< {min_free_mb:.0f} MiB margin)", name, free, total)
    return GPUProbe(True, f"{name}, {free:.0f} of {total:.0f} MiB VRAM free", name, free, total)


def estimate_run_bytes(n: int, n_edges: int, trials: int, n_bins: int,
                       max_spikes: int = MAX_SPIKES_PER_STEP) -> int:
    """Device bytes of the resident CSR plus the state of one run (what the pool limit must cover)."""
    csr = n_edges * (4 + 4) + (n + 1) * 4
    per_neuron = n * (2 + 4 + 1 + 4)  # rfc int16, p float32 (x2 below), silenced u8, counts total
    state = n * trials * (4 + 4 + 2 + 4)  # v, g float32, left int16, trial counts int32
    ring = ARRIVAL_SLOTS * n * trials * 4
    frames = n_bins * n * 4
    return int(csr + per_neuron + state + ring + frames + max_spikes * 4 + n * 4 * 2)


# -- telemetry (best effort) ---------------------------------------------------------------

def parse_nvidia_smi(text: str) -> dict | None:
    """Parse `nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`."""
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        temp, used, total, util = (float(p) for p in parts[1:])
    except ValueError:
        return None
    return {"name": parts[0], "temperature_c": temp, "vram_used_mb": used, "vram_total_mb": total,
            "vram_free_mb": total - used, "utilization_pct": util}


def read_telemetry(timeout_s: float = 3.0) -> dict | None:
    """GPU name, temperature, VRAM and utilisation via nvidia-smi; None when it is not available."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout_s, check=True,
        ).stdout
    except Exception:  # noqa: BLE001 - no driver, no binary, timeout: all mean "unknown"
        return None
    return parse_nvidia_smi(out)


# -- kernels -------------------------------------------------------------------------------

_KERNELS = r"""
__device__ __forceinline__ unsigned long long splitmix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// uniform in [0, 1) from a counter (seed, step, neuron*trials + trial): no generator state
__device__ __forceinline__ float uniform(unsigned long long seed, unsigned int step, unsigned int idx) {
    unsigned long long z = splitmix(seed + (((unsigned long long)step << 32) | idx));
    return (float)(z >> 40) * (1.0f / 16777216.0f);
}

extern "C" __global__ void update(
    float* v, float* g, short* left, const short* rfc, const float* p, const unsigned char* silenced,
    float* arrivals,            // this tick's ring slot: consumed and cleared here
    int* spike_list, int* spike_cnt, int* spike_cnt_next, int* overflow,
    int* trial_counts, unsigned int* frames, int use_frames, int bin,
    int n, int trials, int cap, unsigned long long seed, unsigned int step,
    float decay_v, float decay_g, float drive, float v0, float v_th, float v_rst, float kick)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) *spike_cnt_next = 0;
    if (idx >= n * trials) return;
    int i = idx / trials;
    float arr = arrivals[idx];
    arrivals[idx] = 0.0f;
    short lf = left[idx];
    bool ready = (lf == 0);
    float vv = v[idx], gg = g[idx];
    bool spike = false;
    if (ready) {
        vv = v0 + (vv - v0) * decay_v + gg * drive;
        gg = gg * decay_g;
        spike = vv > v_th;
        gg += arr;                                   // arrivals of refractory neurons are ignored
    } else {
        left[idx] = lf - 1;
    }
    float pi = p[i];
    if (pi > 0.0f && uniform(seed, step, (unsigned int)idx) < pi) vv += kick;
    if (spike) {
        vv = v_rst;
        gg = 0.0f;
        left[idx] = rfc[i];
        atomicAdd(&trial_counts[idx], 1);
        if (use_frames) atomicAdd(&frames[(size_t)bin * n + i], 1u);
        if (!silenced[i]) {
            int pos = atomicAdd(spike_cnt, 1);
            if (pos < cap) spike_list[pos] = idx; else atomicAdd(overflow, 1);
        }
    }
    v[idx] = vv;
    g[idx] = gg;
}

extern "C" __global__ void deliver(
    const int* spike_list, const int* spike_cnt, const int* indptr, const int* indices, const float* weights,
    float* ring_dst, int trials, int cap, float w_syn)
{
    int count = min(*spike_cnt, cap);
    for (int s = blockIdx.x; s < count; s += gridDim.x) {
        int idx = spike_list[s];
        int i = idx / trials, t = idx - i * trials;
        for (int e = indptr[i] + threadIdx.x; e < indptr[i + 1]; e += blockDim.x)
            atomicAdd(&ring_dst[indices[e] * trials + t], weights[e] * w_syn);
    }
}
"""


def _mix(seed: int) -> int:
    """splitmix64 of the user seed on the host, so seeds 0, 1, 2... start far apart."""
    z = (int(seed) + 0x9E3779B97F4A7C15) & (2**64 - 1)
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
    return z ^ (z >> 31)


class GPUSimulation(Simulation):
    """`Simulation` with the trial loop on a CUDA device. Same constructor input (a CSR), same API."""

    backend = "gpu"

    def __init__(self, csr, params, pool_limit_mb: float = POOL_LIMIT_MB,
                 max_spikes_per_step: int = MAX_SPIKES_PER_STEP) -> None:
        super().__init__(csr, params)
        if self._delay != ARRIVAL_SLOTS:
            raise ValueError(f"the GPU ring buffer is built for a delay of {ARRIVAL_SLOTS} ticks, got {self._delay}")
        self.pool_limit_mb = pool_limit_mb
        self.max_spikes = max_spikes_per_step
        self._cp = None
        self._dev: dict | None = None  # resident CSR arrays, built on the first run

    @classmethod
    def from_simulation(cls, sim: Simulation, **kw) -> "GPUSimulation":
        """Share the host CSR of a CPU `Simulation` (no copy); nothing touches the GPU until the first run."""
        out = cls(sim._csr(), sim.params, **kw)
        out.dropped_edges, out.dropped_neurons = sim.dropped_edges, sim.dropped_neurons
        return out

    # -- device state ----------------------------------------------------------------------

    def _cupy(self):
        if self._cp is None:
            try:
                cp = importlib.import_module("cupy")
            except ImportError as exc:
                raise GPUUnavailable(f"cupy is not installed: {exc}") from None
            cp.get_default_memory_pool().set_limit(size=int(self.pool_limit_mb * 2**20))
            self._cp = cp
            self._module = cp.RawModule(code=_KERNELS)
            self._update = self._module.get_function("update")
            self._deliver = self._module.get_function("deliver")
        return self._cp

    def _ensure_resident(self, cp) -> dict:
        if self._dev is None:
            need = estimate_run_bytes(self.n, self.n_edges, 1, 0, self.max_spikes)
            if need > self.pool_limit_mb * 2**20:
                raise GPUCapacityError(f"the weights alone need ~{need / 2**20:.0f} MiB, pool limit is {self.pool_limit_mb:.0f}")
            self._dev = {
                "indptr": cp.asarray(self._indptr, dtype=cp.int32),
                "indices": cp.asarray(self._indices, dtype=cp.int32),
                "weights": cp.asarray(self._counts, dtype=cp.float32),
            }
        return self._dev

    def release(self) -> None:
        """Drop the resident weights and hand every pooled block back to the driver."""
        self._dev = None
        if self._cp is not None:
            self._cp.get_default_memory_pool().free_all_blocks()

    # -- the trial loop (everything else is the CPU base class) --------------------------------

    def _execute(self, stim, epochs, n_steps, trials, seed, silenced, bin_steps, n_bins, bin_ms, duration_ms):
        cp = self._cupy()
        need = estimate_run_bytes(self.n, self.n_edges, trials, n_bins if bin_steps else 0, self.max_spikes)
        if need > self.pool_limit_mb * 2**20:
            raise GPUCapacityError(
                f"{trials} trials x {n_bins} bins need ~{need / 2**20:.0f} MiB, pool limit is {self.pool_limit_mb:.0f} MiB")
        free = cp.cuda.runtime.memGetInfo()[0]
        if free < 256 * 2**20:
            raise GPUCapacityError(f"only {free / 2**20:.0f} MiB VRAM free")
        try:
            return self._run_on_device(cp, stim, epochs, n_steps, trials, seed, silenced, bin_steps, n_bins,
                                       bin_ms, duration_ms)
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise GPUCapacityError(f"out of GPU memory: {exc}") from None
        finally:
            cp.get_default_memory_pool().free_all_blocks()  # the desktop gets its VRAM back

    def _run_on_device(self, cp, stim, epochs, n_steps, trials, seed, silenced, bin_steps, n_bins, bin_ms, duration_ms):
        p, n, T = self.params, self.n, trials
        dev = self._ensure_resident(cp)
        f32, i32 = np.float32, np.int32
        v = cp.full((n, T), p.v0, dtype=cp.float32)
        g = cp.zeros((n, T), dtype=cp.float32)
        left = cp.zeros((n, T), dtype=cp.int16)
        rfc_host = np.full(n, self._rfc - 1, dtype=np.int16)
        rfc_host[stim] = 0  # Poisson targets have no refractory period
        rfc = cp.asarray(rfc_host)
        sil = cp.asarray(silenced.astype(np.uint8))
        ring = cp.zeros((ARRIVAL_SLOTS, n, T), dtype=cp.float32)
        slots = [ring[s] for s in range(ARRIVAL_SLOTS)]
        counts = cp.zeros((n, T), dtype=cp.int32)
        use_frames = bool(bin_steps)
        frames = cp.zeros((n_bins, n), dtype=cp.uint32) if use_frames else cp.zeros(1, dtype=cp.uint32)
        spike_list = cp.zeros(self.max_spikes, dtype=cp.int32)
        spike_cnt = cp.zeros(2, dtype=cp.int32)
        cnt = [spike_cnt[0:], spike_cnt[1:]]  # pointer views, built once
        overflow = cp.zeros(1, dtype=cp.int32)
        p_zero = cp.zeros(n, dtype=cp.float32)
        kick = p.w_syn * p.f_poi
        seed_mixed = np.uint64(_mix(seed))
        consts = (i32(n), i32(T), i32(self.max_spikes), seed_mixed)
        decay = (f32(self._decay_v), f32(self._decay_g), f32(self._drive), f32(p.v0), f32(p.v_th), f32(p.v_rst),
                 f32(kick))
        grid_update = ((n * T + 255) // 256,)
        use_frames_arg = i32(1 if use_frames else 0)
        w_syn, cap = f32(p.w_syn), i32(self.max_spikes)
        trials_arg = i32(T)

        def advance(first: int, last: int, p_dev) -> None:
            for k in range(first, last):
                cur, nxt = k & 1, (k + 1) & 1
                bin_k = i32(k // bin_steps if use_frames else 0)
                self._update(
                    grid_update, (256,),
                    (v, g, left, rfc, p_dev, sil, slots[k % ARRIVAL_SLOTS], spike_list, cnt[cur], cnt[nxt],
                     overflow, counts, frames, use_frames_arg, bin_k, *consts, np.uint32(k), *decay),
                )
                self._deliver(
                    (DELIVER_BLOCKS,), (128,),
                    (spike_list, cnt[cur], dev["indptr"], dev["indices"], dev["weights"],
                     slots[(k + self._delay) % ARRIVAL_SLOTS], trials_arg, cap, w_syn),
                )
                if (k + 1) % CHECK_EVERY == 0 and int(overflow[0]):
                    break  # the check below raises

        pos = 0
        for a, b, rows, p_ev in [*epochs, (n_steps, n_steps, None, None)]:
            if a > pos:
                advance(pos, a, p_zero)
            if b > a:
                p_dev = cp.zeros(n, dtype=cp.float32)
                p_dev[cp.asarray(rows)] = cp.asarray(np.asarray(p_ev, dtype=np.float32))
                advance(a, b, p_dev)
            pos = max(pos, b)
            if int(overflow[0]):
                break
        if int(overflow[0]):
            raise GPUOverflowError(
                f"more than {self.max_spikes} non-silenced spikes in one 0.1 ms tick: the run is invalid "
                "(spikes would have been dropped); use the CPU backend or fewer trials")

        trial_counts = cp.asnumpy(counts.T).astype(np.int32, order="C")
        total = counts.sum(axis=1, dtype=cp.int64)
        active = frames_out = None
        if use_frames:
            idx = cp.flatnonzero(total > 0)
            frames_out = cp.minimum(frames[:, idx].T, np.iinfo(np.uint16).max).astype(cp.uint16)
            active, frames_out = cp.asnumpy(idx).astype(np.int64), cp.asnumpy(frames_out)
        counts_host = cp.asnumpy(total)
        return SimResult(
            counts=counts_host, rate_hz=counts_host / (trials * duration_ms / 1000.0), trials=trials,
            duration_ms=float(duration_ms), trial_counts=trial_counts, bin_ms=bin_ms,
            active_rows=active, frames=frames_out,
        )
