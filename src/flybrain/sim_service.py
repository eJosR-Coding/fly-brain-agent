"""Where the simulation weights live, how to build the cached CSR, and (below) the runtime service."""

import argparse
import os
import secrets
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from flybrain.atlas import load_atlas
from flybrain.simulation import LIFParams, Simulation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_DATA = ROOT / "data" / "raw" / "shiu"
DEFAULT_WEIGHTS = ROOT / "data" / "processed" / "sim_weights_783.npz"
DEFAULT_ANNOTATIONS = ROOT / "data" / "raw" / "flywire_neuron_annotations.tsv"
REFERENCE_REPO = "github.com/philshiu/Drosophila_brain_model (commit 91bdd1e, MIT)"


class SimulationUnavailable(RuntimeError):
    """The simulator cannot run here (weights missing); the message says how to fix it."""


MAX_RESULTS = 24  # activity recordings kept (LRU); a silencing screen keeps 1 baseline + up to 12 silenced runs
MAX_CHARTS = 12  # experiment charts kept (tiny dicts)

# Raw SimResult cache (deterministic given rows + params + seed + weights). Memory per entry on the
# real brain (n ~ 139k neurons): counts int64 1.1 MB + rate_hz float64 1.1 MB + trial_counts
# int32 (trials <= 10) 0.56 MB per trial = ~8 MB without frames; frames are sparse uint16 (active x
# <=100 bins x 2 B): ~0.2 MB typical, ~28 MB in a runaway cascade. 32 entries * ~36 MB worst = ~1.1 GB
# in theory, so a byte budget caps it too: whichever limit hits first evicts the least recently used.
MAX_CACHED_RUNS = 32
MAX_CACHE_BYTES = 300 * 1024 * 1024


def _result_bytes(res) -> int:
    return sum(a.nbytes for a in (res.counts, res.rate_hz, res.trial_counts, res.active_rows, res.frames)
               if a is not None)


class UnknownSimulationError(KeyError):
    """Bad or expired simulation id. str() is the plain message, not a quoted repr."""

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass(frozen=True)
class ActivityRecord:
    """Sparse activity of one run: only neurons that spiked, spike counts per time bin (uint8)."""

    id: str
    rows: np.ndarray  # (n_rows,) uint32 atlas rows, ascending
    frames: np.ndarray  # (n_rows, n_bins) uint8, counts summed over trials (clipped at 255)
    bin_ms: float
    duration_ms: float
    trials: int
    stimulus: dict
    active_count: int | None = None  # neurons above ACTIVE_HZ (the number the stimulate summary reports)
    compare_id: str | None = None  # silenced runs: the id of the baseline recording (same stimulus and seed)
    silenced_rows: np.ndarray | None = None  # silenced runs: atlas rows whose outgoing synapses were zeroed
    readout: dict | None = None  # {"label", "neurons", "hz": mean rate of the readout group per bin}

    @property
    def n_bins(self) -> int:
        return int(self.frames.shape[1])

    @property
    def n_rows(self) -> int:
        return int(self.frames.shape[0])


class SimulationService:
    """The simulator for this process: lazy weights, ONE run at a time, a small LRU of recordings.

    The whole brain simulator costs ~1.5 GB and the machine is shared with a browser, so loading
    is deferred to the first run and every run holds `exclusive()`.
    """

    def __init__(self, loader: Callable[[], Simulation], available: Callable[[], bool] = lambda: True,
                 max_results: int = MAX_RESULTS, max_cached_runs: int = MAX_CACHED_RUNS,
                 max_cache_bytes: int = MAX_CACHE_BYTES) -> None:
        self._loader, self._available = loader, available
        self._run_cache: OrderedDict[tuple, tuple] = OrderedDict()  # key -> (SimResult, nbytes); under _run_lock
        self._run_cache_bytes = 0
        self.max_cached_runs, self.max_cache_bytes = max_cached_runs, max_cache_bytes
        self.cache_hits = 0
        self._sim: Simulation | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._records: OrderedDict[str, ActivityRecord] = OrderedDict()
        self._expired: set[str] = set()
        self._record_lock = threading.Lock()
        self.max_results = max_results
        self._charts: OrderedDict[str, dict] = OrderedDict()
        self._listeners: list[Callable[[dict], None]] = []
        self._progress: dict | None = None

    @classmethod
    def default(cls, atlas, max_results: int = MAX_RESULTS) -> "SimulationService":
        return cls(lambda: load_simulation(atlas)[0], weights_available, max_results)

    @classmethod
    def from_simulation(cls, sim: Simulation, max_results: int = MAX_RESULTS, **cache_limits) -> "SimulationService":
        svc = cls(lambda: sim, lambda: True, max_results, **cache_limits)
        svc._sim = sim
        return svc

    @property
    def available(self) -> bool:
        return self._sim is not None or bool(self._available())

    def simulation(self) -> Simulation:
        """The simulator, loaded on first use (SimulationUnavailable says what is missing)."""
        with self._load_lock:
            if self._sim is None:
                self._sim = self._loader()
            return self._sim

    @contextmanager
    def exclusive(self):
        with self._run_lock:
            yield

    def run(self, stim_rows, rate_hz: float, duration_ms: float = 1000.0, trials: int = 1, seed: int = 0,
            silence_rows=None, bin_ms: float | None = None):
        """`Simulation.run` through the result cache; returns (SimResult, cache_hit).

        Call it inside `exclusive()` (the cache is guarded by that same lock). The key is the
        RESOLVED rows plus every parameter and the weights fingerprint, never selection ticket ids.
        Treat the returned arrays as read-only: they are shared with later hits.
        """
        sim = self.simulation()
        key = (
            sim.fingerprint, np.unique(np.asarray(list(stim_rows), dtype=np.int64)).tobytes(),
            np.unique(np.asarray(list(silence_rows) if silence_rows is not None else [], dtype=np.int64)).tobytes(),
            float(rate_hz), float(duration_ms), int(trials), int(seed), None if bin_ms is None else float(bin_ms),
        )
        if key in self._run_cache:
            self._run_cache.move_to_end(key)
            self.cache_hits += 1
            return self._run_cache[key][0], True
        res = sim.run(stim_rows, rate_hz, duration_ms, trials, seed=seed, silence_rows=silence_rows, bin_ms=bin_ms)
        size = _result_bytes(res)
        if size <= self.max_cache_bytes:
            self._run_cache[key] = (res, size)
            self._run_cache_bytes += size
            while len(self._run_cache) > self.max_cached_runs or self._run_cache_bytes > self.max_cache_bytes:
                _, (_, freed) = self._run_cache.popitem(last=False)
                self._run_cache_bytes -= freed
        return res, False

    def add(self, rows, frames, bin_ms: float, duration_ms: float, trials: int, stimulus: dict,
            active_count: int | None = None, compare_id: str | None = None, silenced_rows=None,
            readout: dict | None = None) -> str:
        with self._record_lock:
            while True:
                sim_id = f"sim_{secrets.token_hex(3)}"
                if sim_id not in self._records and sim_id not in self._expired:
                    break
            self._records[sim_id] = ActivityRecord(
                sim_id, np.asarray(rows, dtype="<u4"), np.asarray(frames, dtype=np.uint8),
                float(bin_ms), float(duration_ms), int(trials), dict(stimulus),
                None if active_count is None else int(active_count),
                compare_id, None if silenced_rows is None else np.asarray(silenced_rows, dtype="<u4"),
                None if readout is None else dict(readout),
            )
            while len(self._records) > self.max_results:
                self._evict_oldest()
            return sim_id

    # -- experiment charts and progress -----------------------------------------------------

    def add_chart(self, payload: dict) -> str:
        """Keep a chart payload (dose-response curve, screen ranking) for the viewer; returns its id."""
        with self._record_lock:
            while True:
                chart_id = f"chart_{secrets.token_hex(3)}"
                if chart_id not in self._charts:
                    break
            self._charts[chart_id] = {**payload, "id": chart_id}
            while len(self._charts) > MAX_CHARTS:
                self._charts.popitem(last=False)
            return chart_id

    def get_chart(self, chart_id: str) -> dict:
        with self._record_lock:
            if chart_id not in self._charts:
                raise UnknownSimulationError(f"unknown or expired chart {chart_id!r}; run the experiment again.")
            self._charts.move_to_end(chart_id)
            return self._charts[chart_id]

    def subscribe(self, listener: Callable[[dict], None]) -> Callable[[], None]:
        """Call `listener({experiment, done, total})` on every progress step (from the worker thread)."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def report_progress(self, experiment: str, done: int, total: int) -> None:
        self._progress = {"experiment": experiment, "done": done, "total": total}
        for listener in list(self._listeners):
            try:
                listener(dict(self._progress))
            except Exception:  # noqa: BLE001 - a dead stream must never break a running experiment
                pass

    def finish_progress(self) -> None:
        self._progress = None

    def progress_snapshot(self) -> dict | None:
        """The running experiment's {experiment, done, total}, or None when nothing long is running."""
        return None if self._progress is None else dict(self._progress)

    def _evict_oldest(self) -> None:
        """Drop the least recently used recording, skipping baselines a silenced run still points at.

        A baseline therefore lives exactly as long as the last run compared against it (the pair
        expires together) and `compare_id` can never dangle. The runs that point at a baseline
        are not baselines themselves, so there is always something else to evict.
        """
        needed = {r.compare_id for r in self._records.values() if r.compare_id is not None}
        victim = next((i for i in self._records if i not in needed), next(iter(self._records)))
        del self._records[victim]
        self._expired.add(victim)

    def get(self, sim_id: str) -> ActivityRecord:
        with self._record_lock:
            if sim_id in self._records:
                self._records.move_to_end(sim_id)
                return self._records[sim_id]
        if sim_id in self._expired:
            raise UnknownSimulationError(
                f"simulation {sim_id!r} expired (only the last {self.max_results} are kept); run the simulation again."
            )
        raise UnknownSimulationError(
            f"unknown simulation {sim_id!r}; ids come from the stimulate tool or POST /simulate."
        )


def sim_data_dir() -> Path:
    return Path(os.environ.get("FLYBRAIN_SIM_DATA", DEFAULT_SIM_DATA))


def connectivity_path() -> Path:
    return sim_data_dir() / "Connectivity_783.parquet"


def completeness_path() -> Path:
    return sim_data_dir() / "Completeness_783.csv"


def weights_cache_path() -> Path:
    return Path(os.environ.get("FLYBRAIN_SIM_WEIGHTS", DEFAULT_WEIGHTS))


def sources_present() -> bool:
    return connectivity_path().is_file() and completeness_path().is_file()


def weights_available() -> bool:
    """Can a simulator be built here? Either the cache or both source files must exist."""
    return weights_cache_path().is_file() or sources_present()


def load_simulation(atlas) -> tuple[Simulation, bool]:
    """Load the simulator (cache first, else build from the reference files and write the cache)."""
    con, comp, cache = connectivity_path(), completeness_path(), weights_cache_path()
    if not cache.is_file() and not (con.is_file() and comp.is_file()):
        raise SimulationUnavailable(
            f"simulation weights not found: expected {con.name} and {comp.name} in {sim_data_dir()} "
            f"(download them from {REFERENCE_REPO}) or a built cache at {cache}; "
            "then run flybrain-build-sim-weights"
        )
    if not (con.is_file() and comp.is_file()):
        # cache only: its signature needs the sources, so trust the file when they are gone
        try:
            with np.load(cache) as z:
                n = len(atlas.neurons)
                csr = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
                return Simulation(csr, LIFParams()), True
        except (OSError, ValueError, KeyError) as exc:
            raise SimulationUnavailable(f"cannot read the weights cache {cache}: {exc}") from None
    return Simulation.load_weights(con, comp, atlas, cache=cache)


def build_weights(connectivity, completeness, atlas, cache) -> dict:
    """Build (or confirm) the cached signed CSR. Returns {edges, cache_hit, seconds}."""
    if not Path(connectivity).is_file() or not Path(completeness).is_file():
        raise SimulationUnavailable(
            f"missing {Path(connectivity).name} / {Path(completeness).name}: download them from {REFERENCE_REPO} "
            "into data/raw/shiu/ (or point FLYBRAIN_SIM_DATA at the folder)"
        )
    t0 = time.perf_counter()
    sim, hit = Simulation.load_weights(connectivity, completeness, atlas, cache=cache)
    return {"edges": sim.n_edges, "cache_hit": hit, "seconds": round(time.perf_counter() - t0, 2),
            "dropped_edges": sim.dropped_edges, "dropped_neurons": sim.dropped_neurons}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the cached simulation weight matrix (sim_weights_783.npz).")
    p.add_argument("--annotations", default=os.environ.get("FLYBRAIN_ANNOTATIONS", DEFAULT_ANNOTATIONS))
    p.add_argument("--connectivity", default=None)
    p.add_argument("--completeness", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    atlas = load_atlas(Path(args.annotations))
    try:
        info = build_weights(
            args.connectivity or connectivity_path(), args.completeness or completeness_path(),
            atlas, args.out or weights_cache_path(),
        )
    except SimulationUnavailable as exc:
        print(f"flybrain-build-sim-weights: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(info)


if __name__ == "__main__":
    main()
