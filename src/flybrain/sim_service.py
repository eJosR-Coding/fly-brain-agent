"""Where the simulation weights live, how to build the cached CSR, and (below) the runtime service."""

import argparse
import os
import sys
import time
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
