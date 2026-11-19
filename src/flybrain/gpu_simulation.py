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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flybrain.simulation import BrainStream, ChunkSpikes, Simulation, SimResult, SimulationError, _steps

log = logging.getLogger(__name__)

# Free VRAM required to even try the GPU: pool limit + the CUDA context (~300-400 MiB) + headroom
# for the desktop compositor and browser that share the card.
MIN_FREE_VRAM_MB = 1536
POOL_LIMIT_MB = 1024  # hard cap on CuPy's memory pool (CSR + one run's state)
MAX_SPIKES_PER_STEP = 262_144  # spike event buffer capacity per tick; beyond it the run raises
DELIVER_BLOCKS = 2048  # grid-stride blocks of the deliver kernel (spike count is read on the device)
CHECK_EVERY = 500  # steps between overflow checks (a device sync)
ARRIVAL_SLOTS = 18  # ring slots; must equal t_dly / dt (checked)
LIST_SLOTS = ARRIVAL_SLOTS + 1  # work lists of the active-set path (one more than the ring: see _GPUState.advance)
SPARSE_BLOCKS = 160  # grid-stride blocks of the active-set update kernel (the work list length is read on the device)
SPARSE_DELIVER_BLOCKS = 256
SWITCH_TO_DENSE = 8  # plain kernels when more than 1/8 of the neurons are still moving ...
SWITCH_TO_SPARSE = 16  # ... and the active-set ones again below 1/16 (hysteresis)


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
    work_lists = LIST_SLOTS * n * 8  # active-set path: int32 list + int32 flag per slot (single trial only)
    return int(csr + per_neuron + state + ring + frames + work_lists + max_spikes * 4 + n * 4 * 2)


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


THERMAL_LIMIT_C = 80.0  # no new GPU work above this temperature
THERMAL_CHECK_S = 3.0  # nvidia-smi is a subprocess: ask at most this often
THERMAL_RESUME_MARGIN_C = 5.0  # once hot, the GPU must cool this far below the limit before it is used again


class ThermalGuard:
    """Refuses new GPU work while the card is hot. `allow()` is cheap: it re-reads the temperature at most every
    `interval_s`. Unknown telemetry (no nvidia-smi, a failing reader) never blocks: the guard is best effort and
    the VRAM guard stays the hard one. Hysteresis: once above `limit_c` it allows again only below
    `limit_c - resume_margin_c`, so a card hovering at the limit does not flap between backends."""

    def __init__(self, read: Callable[[], dict | None] = read_telemetry, limit_c: float = THERMAL_LIMIT_C,
                 interval_s: float = THERMAL_CHECK_S, resume_margin_c: float = THERMAL_RESUME_MARGIN_C,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._read, self.limit_c, self.interval_s = read, limit_c, interval_s
        self.resume_margin_c, self._clock = resume_margin_c, clock
        self._lock = threading.Lock()
        self._checked_at: float | None = None
        self.temperature_c: float | None = None
        self.hot = False
        self.refusals = 0

    def allow(self) -> bool:
        with self._lock:
            now = self._clock()
            if self._checked_at is None or now - self._checked_at >= self.interval_s:
                self._checked_at = now
                try:
                    info = self._read()
                except Exception:  # noqa: BLE001
                    info = None
                temp = None if not info else info.get("temperature_c")
                self.temperature_c = None if temp is None else float(temp)
                t = self.temperature_c
                if t is None:
                    self.hot = False
                elif t > self.limit_c:
                    self.hot = True
                elif t < self.limit_c - self.resume_margin_c:
                    self.hot = False
            if self.hot:
                self.refusals += 1
            return not self.hot

    def status(self) -> dict:
        return {"limit_c": self.limit_c, "temperature_c": self.temperature_c, "hot": self.hot,
                "refusals": self.refusals}


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

// the free relaxation of a ready neuron (before arrivals): ONE definition, so every kernel that asks "would this
// neuron move?" computes the very same floats as the update itself
__device__ __forceinline__ void relax(float& vv, float& gg, float decay_v, float decay_g, float drive, float v0) {
    vv = v0 + (vv - v0) * decay_v + gg * drive;
    gg = gg * decay_g;
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
    if (arr != 0.0f) arrivals[idx] = 0.0f;            // a zero slot is already clear: no write
    short lf = left[idx];
    bool ready = (lf == 0);
    float v_old = v[idx], g_old = g[idx];
    float vv = v_old, gg = g_old;
    bool spike = false;
    if (ready) {
        relax(vv, gg, decay_v, decay_g, drive, v0);
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
    if (vv != v_old) v[idx] = vv;                      // an unchanged value is not written back
    if (gg != g_old) g[idx] = gg;
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

// ---- active-set path (one trial): only neurons that can change are touched ----------------------------
// A neuron whose update would leave (v, g, left) bit-identical (ready, no arrival, no kick, v and g at a float
// fixed point) is skipped until something reaches it, so skipping it is exact. Work lists are per arrival tick
// (`lists[tick % LIST_SLOTS]`, deduplicated with `flags`): synaptic arrivals are appended by `deliver_sparse` for
// tick+18, still-changing neurons are appended by `update_sparse` for tick+1.
#define DENSE_FRACTION 6
extern "C" __global__ void update_sparse(
    float* v, float* g, short* left, const short* rfc, const float* p, const unsigned char* silenced,
    float* arrivals,
    const int* list_cur, int* cnt_cur, int* flags_cur, int* list_nxt, int* cnt_nxt, int* flags_nxt,
    int* cnt_stale,
    int* spike_list, int* spike_cnt, int* spike_cnt_next, int* overflow,
    int* trial_counts, unsigned int* frames, int use_frames, int bin,
    int n, int cap, unsigned long long seed, unsigned int step,
    float decay_v, float decay_g, float drive, float v0, float v_th, float v_rst, float kick)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) { *spike_cnt_next = 0; *cnt_stale = 0; }
    int count = *cnt_cur;
    // a crowded list is slower to gather than the whole array is to stream: then every neuron is visited (a neuron
    // outside the list is at a fixed point, so visiting it changes nothing and it is not carried)
    bool all = count * DENSE_FRACTION > n;
    int total = all ? n : count;
    const unsigned int lane = threadIdx.x & 31u;
    for (int base = blockIdx.x * blockDim.x; base < total; base += gridDim.x * blockDim.x) {
      int s = base + threadIdx.x;
      bool live = s < total;
      bool carry = false;
      int idx = 0;
      if (live) {
        idx = all ? s : list_cur[s];
        flags_cur[idx] = 0;
        float arr = arrivals[idx];
        if (arr != 0.0f) arrivals[idx] = 0.0f;
        short lf = left[idx];
        bool ready = (lf == 0);
        float v_old = v[idx], g_old = g[idx];
        float vv = v_old, gg = g_old;
        bool spike = false;
        if (ready) {
            relax(vv, gg, decay_v, decay_g, drive, v0);
            spike = vv > v_th;
            gg += arr;
        } else {
            left[idx] = lf - 1;
        }
        float pi = p[idx];
        bool kicked = false;
        if (pi > 0.0f) {
            kicked = true;
            if (uniform(seed, step, (unsigned int)idx) < pi) vv += kick;
        }
        carry = !ready || kicked;
        if (spike) {
            vv = v_rst;
            gg = 0.0f;
            left[idx] = rfc[idx];
            atomicAdd(&trial_counts[idx], 1);
            if (use_frames) atomicAdd(&frames[(size_t)bin * n + idx], 1u);
            if (!silenced[idx]) {
                int pos = atomicAdd(spike_cnt, 1);
                if (pos < cap) spike_list[pos] = idx; else atomicAdd(overflow, 1);
            }
            carry = true;
        }
        if (vv != v_old || gg != g_old) carry = true;
        if (vv != v_old) v[idx] = vv;
        if (gg != g_old) g[idx] = gg;
        carry = carry && atomicExch(&flags_nxt[idx], 1) == 0;
      }
      // one counter increment per warp instead of one per neuron
      unsigned int vote = __ballot_sync(0xffffffffu, carry);
      if (vote) {
          int basepos = 0;
          if (lane == 0) basepos = atomicAdd(cnt_nxt, __popc(vote));
          basepos = __shfl_sync(0xffffffffu, basepos, 0);
          if (carry) list_nxt[basepos + __popc(vote & ((1u << lane) - 1u))] = idx;
      }
    }
}

// Rebuilds the work lists from the state alone (when a chunk switches from the plain kernels to the active-set ones)
// and counts the neurons that move at tick `first`. build == 0 only counts. A neuron is listed for tick first + j when
// its ring slot holds a non-zero arrival (adding 0 changes nothing), and for tick `first` when it is refractory,
// Poisson-driven, or its free relaxation would change v or g or make it spike.
extern "C" __global__ void rebuild(
    const float* v, const float* g, const short* left, const float* p, const float* ring,
    int* lists, int* cnts, int* flags, int* moving,
    int n, int first, int build,
    float decay_v, float decay_g, float drive, float v0, float v_th)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bool carry = left[idx] != 0 || p[idx] > 0.0f;
    if (!carry) {
        float vv = v[idx], gg = g[idx];
        relax(vv, gg, decay_v, decay_g, drive, v0);
        carry = vv != v[idx] || gg != g[idx] || vv > v_th;
    }
    if (carry) atomicAdd(moving, 1);
    if (!build) return;
    int c0 = first % 19;
    if (carry && atomicExch(&flags[(size_t)c0 * n + idx], 1) == 0) lists[(size_t)c0 * n + atomicAdd(&cnts[c0], 1)] = idx;
    for (int j = 0; j < 18; j++) {
        if (ring[(size_t)((first + j) % 18) * n + idx] == 0.0f) continue;
        int c = (first + j) % 19;
        if (atomicExch(&flags[(size_t)c * n + idx], 1) == 0) lists[(size_t)c * n + atomicAdd(&cnts[c], 1)] = idx;
    }
}

// puts the Poisson-driven rows into the work list of the next tick (a stimulus may have just changed)
extern "C" __global__ void seed_stim(const int* rows, int m, int* list, int* cnt, int* flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    int idx = rows[i];
    if (atomicExch(&flags[idx], 1) == 0) list[atomicAdd(cnt, 1)] = idx;
}

extern "C" __global__ void deliver_sparse(
    const int* spike_list, const int* spike_cnt, const int* indptr, const int* indices, const float* weights,
    float* ring_dst, int* list_dst, int* cnt_dst, int* flags_dst, int cap, float w_syn)
{
    int count = min(*spike_cnt, cap);
    for (int s = blockIdx.x; s < count; s += gridDim.x) {
        int i = spike_list[s];
        for (int e = indptr[i] + threadIdx.x; e < indptr[i + 1]; e += blockDim.x) {
            int tgt = indices[e];
            atomicAdd(&ring_dst[tgt], weights[e] * w_syn);
            if (flags_dst[tgt] == 0 && atomicExch(&flags_dst[tgt], 1) == 0) list_dst[atomicAdd(cnt_dst, 1)] = tgt;
        }
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
        self.active_set = True  # single-trial runs and live sessions only touch neurons that can change (exact)
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
            self._update_sparse = self._module.get_function("update_sparse")
            self._deliver_sparse = self._module.get_function("deliver_sparse")
            self._seed_stim = self._module.get_function("seed_stim")
            self._rebuild = self._module.get_function("rebuild")
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
        st = _GPUState(self, cp, stim, silenced, trials, seed, n_bins if bin_steps else 0, bin_steps)
        st.run_epochs(epochs, n_steps)
        st.check_overflow()

        counts, frames = st.counts, st.frames
        trial_counts = cp.asnumpy(counts.T).astype(np.int32, order="C")
        total = counts.sum(axis=1, dtype=cp.int64)
        active = frames_out = None
        if st.use_frames:
            idx = cp.flatnonzero(total > 0)
            frames_out = cp.minimum(frames[:, idx].T, np.iinfo(np.uint16).max).astype(cp.uint16)
            active, frames_out = cp.asnumpy(idx).astype(np.int64), cp.asnumpy(frames_out)
        counts_host = cp.asnumpy(total)
        return SimResult(
            counts=counts_host, rate_hz=counts_host / (trials * duration_ms / 1000.0), trials=trials,
            duration_ms=float(duration_ms), trial_counts=trial_counts, bin_ms=bin_ms,
            active_rows=active, frames=frames_out,
        )

    def open_stream(self, seed: int = 0, silenced=None) -> "GPUBrainStream":
        """A live session with its state on the device (1 trial; same capacity checks as a run)."""
        return GPUBrainStream(self, seed, silenced)


class _GPUState:
    """Device arrays of one run or one live session (`n x trials`) and the tick loop over them.

    The loop and the two kernels are the ones `GPUSimulation` always used; `k` is the next absolute tick, which
    selects the ring slot, the cur/nxt spike counter and the RNG counter, so a session continues exactly where a
    previous `advance` stopped."""

    def __init__(self, sim: "GPUSimulation", cp, stim, silenced, trials: int, seed: int, n_bins: int = 0,
                 bin_steps: int | None = None) -> None:
        p, n, T = sim.params, sim.n, trials
        self.sparse = bool(sim.active_set and T == 1)
        self.sim, self.cp, self.k, self.bin_steps = sim, cp, 0, bin_steps
        self.dev = sim._ensure_resident(cp)
        f32, i32 = np.float32, np.int32
        self.v = cp.full((n, T), p.v0, dtype=cp.float32)
        self.g = cp.zeros((n, T), dtype=cp.float32)
        self.left = cp.zeros((n, T), dtype=cp.int16)
        self.rfc = cp.zeros(n, dtype=cp.int16)
        self.set_poisson_targets(stim)
        self.sil = cp.asarray(silenced.astype(np.uint8))
        ring = cp.zeros((ARRIVAL_SLOTS, n, T), dtype=cp.float32)
        self.ring = ring
        self.slots = [ring[s] for s in range(ARRIVAL_SLOTS)]
        self.counts = cp.zeros((n, T), dtype=cp.int32)
        self.use_frames = bool(bin_steps)
        self.frames = cp.zeros((n_bins, n), dtype=cp.uint32) if self.use_frames else cp.zeros(1, dtype=cp.uint32)
        self.spike_list = cp.zeros(sim.max_spikes, dtype=cp.int32)
        spike_cnt = cp.zeros(2, dtype=cp.int32)
        self.cnt = [spike_cnt[0:], spike_cnt[1:]]  # pointer views, built once
        self.overflow = cp.zeros(1, dtype=cp.int32)
        self.p_zero = cp.zeros(n, dtype=cp.float32)
        kick = p.w_syn * p.f_poi
        self.consts = (i32(n), i32(T), i32(sim.max_spikes), np.uint64(_mix(seed)))
        self.decay = (f32(sim._decay_v), f32(sim._decay_g), f32(sim._drive), f32(p.v0), f32(p.v_th), f32(p.v_rst),
                      f32(kick))
        self.grid_update = ((n * T + 255) // 256,)
        self.use_frames_arg = i32(1 if self.use_frames else 0)
        self.w_syn, self.cap, self.trials_arg = f32(p.w_syn), i32(sim.max_spikes), i32(T)
        if self.sparse:
            self.lists = cp.zeros((LIST_SLOTS, n), dtype=cp.int32)
            self.flags = cp.zeros((LIST_SLOTS, n), dtype=cp.int32)
            list_cnt = cp.zeros(LIST_SLOTS, dtype=cp.int32)
            self.list_cnt_all = list_cnt
            self.list_cnt = [list_cnt[s:] for s in range(LIST_SLOTS)]  # pointer views, built once
            self.list_rows = [self.lists[s] for s in range(LIST_SLOTS)]
            self.flag_rows = [self.flags[s] for s in range(LIST_SLOTS)]
            self.stim_for, self.stim_rows = None, None
            self.lists_valid = True  # all neurons start at their fixed point: empty lists are right for tick 0
            self.moving = cp.zeros(1, dtype=cp.int32)
            self.sparse_consts = (i32(n), i32(sim.max_spikes), np.uint64(_mix(seed)))

    def set_poisson_targets(self, stim) -> None:
        rfc_host = np.full(self.sim.n, self.sim._rfc - 1, dtype=np.int16)
        rfc_host[stim] = 0  # Poisson targets have no refractory period
        self.rfc[:] = self.cp.asarray(rfc_host)

    def rate_vector(self, rows, p_event):
        """Per-neuron event probability per tick on the device (zero outside `rows`)."""
        cp = self.cp
        p_dev = cp.zeros(self.sim.n, dtype=cp.float32)
        p_dev[cp.asarray(rows)] = cp.asarray(np.asarray(p_event, dtype=np.float32))
        return p_dev

    def _moving_now(self, first: int, build: bool) -> int:
        """Neurons that move at tick `first`; with `build`, also rebuilds the work lists from the state."""
        cp, sim, f32 = self.cp, self.sim, np.float32
        n = sim.n
        if build:
            self.list_cnt_all.fill(0)
            self.flags.fill(0)
        self.moving.fill(0)
        sim._rebuild(((n + 255) // 256,), (256,),
                     (self.v, self.g, self.left, self.p_dev_now, self.ring, self.lists, self.list_cnt_all, self.flags,
                      self.moving, np.int32(n), np.int32(first), np.int32(1 if build else 0),
                      self.decay[0], self.decay[1], self.decay[2], self.decay[3], self.decay[4]))
        return int(self.moving[0])

    def _choose_mode(self, first: int, p_dev) -> bool:
        """True: run this advance on the active-set kernels. Busy brains (a third of the neurons still relaxing) are
        cheaper on the plain kernels, whose cost does not depend on activity; switching rebuilds the lists from the
        state, so it is exact in both directions."""
        n = self.sim.n
        if self.lists_valid:
            busy = int(self.list_cnt[first % LIST_SLOTS][0])
            if busy * SWITCH_TO_DENSE <= n:
                return True
        self.p_dev_now = p_dev
        moving = self._moving_now(first, build=False)
        if moving * SWITCH_TO_SPARSE >= n:
            self.lists_valid = False
            return False
        self._moving_now(first, build=True)
        self.lists_valid = True
        return True

    def _advance_sparse(self, first: int, last: int, p_dev) -> None:
        """Active-set ticks. The work list of tick j is `lists[j % 19]`: the deliver of tick j-18 appends the arrivals
        scheduled for j, the update of tick j-1 appends the neurons that were still changing. The deliver of tick k
        therefore writes list `(k + 18) % 19 == (k - 1) % 19`, the one tick k-1 consumed and the update of tick k
        empties (count reset) just before that deliver runs; 19 lists, not 18, is what makes that order safe."""
        sim, i32, cp = self.sim, np.int32, self.cp
        dev = self.dev
        if self.stim_for is not p_dev:
            self.stim_rows, self.stim_for = cp.flatnonzero(p_dev).astype(cp.int32), p_dev
        m = int(self.stim_rows.size)
        if m:
            c = first % LIST_SLOTS
            sim._seed_stim(((m + 255) // 256,), (256,),
                           (self.stim_rows, i32(m), self.list_rows[c], self.list_cnt[c], self.flag_rows[c]))
        n_arg, cap, seed_arg = self.sparse_consts
        for k in range(first, last):
            cur, nxt, stale = k % LIST_SLOTS, (k + 1) % LIST_SLOTS, (k - 1) % LIST_SLOTS
            spk_cur, spk_nxt = k & 1, (k + 1) & 1
            bin_k = i32(k // self.bin_steps if self.use_frames else 0)
            sim._update_sparse(
                (SPARSE_BLOCKS,), (256,),
                (self.v, self.g, self.left, self.rfc, p_dev, self.sil, self.slots[k % ARRIVAL_SLOTS],
                 self.list_rows[cur], self.list_cnt[cur], self.flag_rows[cur],
                 self.list_rows[nxt], self.list_cnt[nxt], self.flag_rows[nxt], self.list_cnt[stale],
                 self.spike_list, self.cnt[spk_cur], self.cnt[spk_nxt], self.overflow, self.counts, self.frames,
                 self.use_frames_arg, bin_k, n_arg, cap, seed_arg, np.uint32(k), *self.decay),
            )
            dst = (k + ARRIVAL_SLOTS) % LIST_SLOTS
            sim._deliver_sparse(
                (SPARSE_DELIVER_BLOCKS,), (128,),
                (self.spike_list, self.cnt[spk_cur], dev["indptr"], dev["indices"], dev["weights"],
                 self.slots[(k + sim._delay) % ARRIVAL_SLOTS], self.list_rows[dst], self.list_cnt[dst],
                 self.flag_rows[dst], cap, self.w_syn),
            )
            if (k + 1) % CHECK_EVERY == 0 and int(self.overflow[0]):
                break  # check_overflow raises
        self.k = max(self.k, last)

    def advance(self, first: int, last: int, p_dev) -> None:
        if self.sparse and self._choose_mode(first, p_dev):
            return self._advance_sparse(first, last, p_dev)
        sim, i32 = self.sim, np.int32
        dev = self.dev
        for k in range(first, last):
            cur, nxt = k & 1, (k + 1) & 1
            bin_k = i32(k // self.bin_steps if self.use_frames else 0)
            sim._update(
                self.grid_update, (256,),
                (self.v, self.g, self.left, self.rfc, p_dev, self.sil, self.slots[k % ARRIVAL_SLOTS], self.spike_list,
                 self.cnt[cur], self.cnt[nxt], self.overflow, self.counts, self.frames, self.use_frames_arg, bin_k,
                 *self.consts, np.uint32(k), *self.decay),
            )
            sim._deliver(
                (DELIVER_BLOCKS,), (128,),
                (self.spike_list, self.cnt[cur], dev["indptr"], dev["indices"], dev["weights"],
                 self.slots[(k + sim._delay) % ARRIVAL_SLOTS], self.trials_arg, self.cap, self.w_syn),
            )
            if (k + 1) % CHECK_EVERY == 0 and int(self.overflow[0]):
                break  # check_overflow raises
        self.k = max(self.k, last)

    def run_epochs(self, epochs, n_steps: int) -> None:
        pos = 0
        for a, b, rows, p_ev in [*epochs, (n_steps, n_steps, None, None)]:
            if a > pos:
                self.advance(pos, a, self.p_zero)
            if b > a:
                self.advance(a, b, self.rate_vector(rows, p_ev))
            pos = max(pos, b)
            if int(self.overflow[0]):
                break

    def check_overflow(self) -> None:
        if int(self.overflow[0]):
            raise GPUOverflowError(
                f"more than {self.sim.max_spikes} non-silenced spikes in one 0.1 ms tick: the run is invalid "
                "(spikes would have been dropped); use the CPU backend or fewer trials")


class GPUBrainStream(BrainStream):
    """`BrainStream` with the state on the GPU: a few small arrays (~12 MiB on the whole brain) kept for the session.

    Counter-based Poisson draws make it bit-identical to a one-shot `GPUSimulation.run` with the same seed."""

    backend = "gpu"

    def __init__(self, sim: "GPUSimulation", seed: int = 0, silenced=None) -> None:
        self.sim = sim
        mask = np.zeros(sim.n, dtype=bool)
        mask[sim._rows(silenced, "silenced")] = True
        cp = sim._cupy()
        need = estimate_run_bytes(sim.n, sim.n_edges, 1, 0, sim.max_spikes)
        if need > sim.pool_limit_mb * 2**20:
            raise GPUCapacityError(f"a live session needs ~{need / 2**20:.0f} MiB, pool limit is {sim.pool_limit_mb:.0f} MiB")
        free = cp.cuda.runtime.memGetInfo()[0]
        if free < 256 * 2**20:
            raise GPUCapacityError(f"only {free / 2**20:.0f} MiB VRAM free")
        try:
            self._state = _GPUState(sim, cp, np.empty(0, dtype=np.int64), mask, 1, seed)
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise GPUCapacityError(f"out of GPU memory: {exc}") from None
        self._cp = cp
        self._p_dev = self._state.p_zero
        self._prev = cp.zeros(sim.n, dtype=cp.int32)
        self._rows, self._p_rows, self.stimulus = np.empty(0, dtype=np.int64), np.empty(0), None

    @property
    def t_ms(self) -> float:
        return self._state.k * self.sim.params.dt

    def _on_stimulus(self, rows: np.ndarray) -> None:
        st = self._state
        st.set_poisson_targets(rows)
        self._p_dev = st.rate_vector(rows, self._p_rows) if self.stimulus else st.p_zero

    def advance(self, ms: float) -> ChunkSpikes:
        sim, st, cp = self.sim, self._state, self._cp
        n_steps = _steps(ms, sim.params.dt, "advance ms")
        try:
            st.advance(st.k, st.k + n_steps, self._p_dev)
            st.check_overflow()
            now = st.counts[:, 0]
            delta = now - self._prev
            idx = cp.flatnonzero(delta)
            rows, counts = cp.asnumpy(idx).astype(np.int64), cp.asnumpy(delta[idx]).astype(np.int32)
            self._prev = now.copy()
        except cp.cuda.memory.OutOfMemoryError as exc:
            raise GPUCapacityError(f"out of GPU memory: {exc}") from None
        return ChunkSpikes(rows, counts, float(ms))

    def close(self) -> None:
        """Drop the session's device state and hand the pooled blocks back to the driver."""
        self._state = self._prev = self._p_dev = None
        self._cp.get_default_memory_pool().free_all_blocks()
