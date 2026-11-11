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


MAX_RESULTS = 10  # activity recordings kept (LRU), like selection tickets


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
                 max_results: int = MAX_RESULTS) -> None:
        self._loader, self._available = loader, available
        self._sim: Simulation | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._records: OrderedDict[str, ActivityRecord] = OrderedDict()
        self._expired: set[str] = set()
        self._record_lock = threading.Lock()
        self.max_results = max_results

    @classmethod
    def default(cls, atlas, max_results: int = MAX_RESULTS) -> "SimulationService":
        return cls(lambda: load_simulation(atlas)[0], weights_available, max_results)

    @classmethod
    def from_simulation(cls, sim: Simulation, max_results: int = MAX_RESULTS) -> "SimulationService":
        svc = cls(lambda: sim, lambda: True, max_results)
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
                self._expired.add(self._records.popitem(last=False)[0])
            return sim_id

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
