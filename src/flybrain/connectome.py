"""Connectome core: who connects to whom, as sparse graphs over atlas rows. numpy/scipy only.

Everything here is STRUCTURAL: an edge means "this neuron makes at least `min_synapses` synapses
onto that one" (summed over neuropils). It says nothing about signal flow, sign or strength of
influence. Nodes are atlas rows (int32), edge weights are synapse counts.

Queries use frontier BFS over CSR slices (vectorised numpy, no per-node Python graph), and every
tie is broken by the lowest row so results are deterministic.
"""

from pathlib import Path

import numpy as np
from scipy import sparse

DEFAULT_MIN_SYNAPSES = 5  # filters single-synapse noise; ~2.7M of 15M pairs survive on FlyWire 783
DIRECTIONS = ("in", "out", "both")
METRICS = ("in", "out", "total")


class ConnectomeError(ValueError):
    """The connectome file is missing or does not match the atlas."""


def _rows(rows, n: int) -> np.ndarray:
    arr = np.unique(np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.int64))
    if len(arr) and (arr[0] < 0 or arr[-1] >= n):
        raise ValueError(f"row out of range 0..{n - 1}")
    return arr


def _sorted_desc(rows: np.ndarray, values: np.ndarray, top: int | None):
    """Sort by value descending, ties by row ascending; optionally keep the first `top`."""
    order = np.lexsort((rows, -values.astype(np.int64)))
    if top is not None:
        order = order[: max(top, 0)]
    return rows[order].astype(np.int64), values[order].astype(np.int64)


class Connectome:
    def __init__(self, out: sparse.csr_matrix, min_synapses: int) -> None:
        self.n = out.shape[0]
        self.min_synapses = min_synapses
        self._out = out
        self._in = out.T.tocsr()
        self.n_edges = int(out.nnz)
        self._out_strength = np.asarray(out.sum(axis=1)).ravel().astype(np.int64)
        self._in_strength = np.asarray(out.sum(axis=0)).ravel().astype(np.int64)

    @classmethod
    def from_edges(cls, pre, post, syn, n_neurons: int, min_synapses: int = DEFAULT_MIN_SYNAPSES) -> "Connectome":
        pre, post, syn = np.asarray(pre), np.asarray(post), np.asarray(syn)
        keep = syn >= min_synapses
        mat = sparse.csr_matrix(
            (syn[keep].astype(np.int32), (pre[keep].astype(np.int32), post[keep].astype(np.int32))),
            shape=(n_neurons, n_neurons),
        )
        mat.sum_duplicates()
        return cls(mat, min_synapses)

    @classmethod
    def load(cls, path, n_neurons: int, min_synapses: int = DEFAULT_MIN_SYNAPSES) -> "Connectome":
        """Read the npz written by flybrain-build-connectome and apply the synapse threshold."""
        path = Path(path)
        if not path.is_file():
            raise ConnectomeError(f"connectome file not found: {path}; run flybrain-build-connectome first")
        try:
            with np.load(path) as z:
                built_for = int(z["n_neurons"])
                if built_for != n_neurons:  # check before reading the big arrays
                    raise ConnectomeError(
                        f"connectome was built for {built_for} neurons but the atlas has {n_neurons}; "
                        "rebuild it with flybrain-build-connectome against the same annotations"
                    )
                pre, post, syn = z["pre"], z["post"], z["syn"]
        except ConnectomeError:
            raise
        except (OSError, KeyError, ValueError) as exc:
            raise ConnectomeError(f"could not read connectome {path}: {exc}") from None
        return cls.from_edges(pre, post, syn, n_neurons, min_synapses)

    # -- neighbors ---------------------------------------------------------------------

    def neighbors(self, rows, direction: str = "out", top: int | None = None):
        """Partners of a group with summed synapse counts, biggest first.

        direction "out": neurons the group projects to; "in": neurons projecting onto the group;
        "both": the sum. The input rows themselves are never listed. Returns (rows, synapses).
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}, got {direction!r}")
        src = _rows(rows, self.n)
        totals = np.zeros(self.n, dtype=np.int64)
        for mat, wanted in ((self._out, "out"), (self._in, "in")):
            if direction in (wanted, "both"):
                sub = mat[src]
                totals += np.bincount(sub.indices, weights=sub.data, minlength=self.n).astype(np.int64)
        totals[src] = 0
        partners = np.flatnonzero(totals)
        return _sorted_desc(partners, totals[partners], top)

    # -- BFS ---------------------------------------------------------------------------

    def _bfs(self, sources: np.ndarray, max_hops: int, removed: np.ndarray | None = None,
             stop_at: np.ndarray | None = None):
        """Multi-source BFS. Returns (dist, parent), -1 where unreached.

        Each level's frontier is sorted, so a node's parent is the lowest-row frontier node
        that reaches it: deterministic. With `stop_at`, stops after the first level that hits any.
        """
        indptr, indices = self._out.indptr, self._out.indices
        dist = np.full(self.n, -1, dtype=np.int32)
        parent = np.full(self.n, -1, dtype=np.int64)
        blocked = np.zeros(self.n, dtype=bool)
        if removed is not None and len(removed):
            blocked[removed] = True
        frontier = sources[~blocked[sources]]
        dist[frontier] = 0
        hop = 0
        while len(frontier) and hop < max_hops:
            if stop_at is not None and (dist[stop_at] >= 0).any():
                break
            starts, ends = indptr[frontier], indptr[frontier + 1]
            counts = ends - starts
            total = int(counts.sum())
            if total == 0:
                break
            offsets = np.repeat(starts - np.cumsum(counts) + counts, counts)
            nbrs = indices[offsets + np.arange(total)]
            who = np.repeat(frontier, counts)
            fresh = (dist[nbrs] < 0) & ~blocked[nbrs]
            nbrs, who = nbrs[fresh], who[fresh]
            if not len(nbrs):
                break
            uniq, first = np.unique(nbrs, return_index=True)  # first occurrence = lowest parent
            hop += 1
            dist[uniq] = hop
            parent[uniq] = who[first]
            frontier = uniq.astype(np.int64)
        return dist, parent

    def shortest_path(self, sources, targets, max_hops: int = 10):
        """Fewest-synapse-hops directed route from any source to any target, or None.

        Returns (hops, path_rows). Ties go to the lowest row at every step, so the same question
        always gives the same path. A source that is also a target is a 0-hop path.
        """
        src, tgt = _rows(sources, self.n), _rows(targets, self.n)
        if not len(src) or not len(tgt):
            return None
        dist, parent = self._bfs(src, max_hops, stop_at=tgt)
        hit = tgt[dist[tgt] >= 0]
        if not len(hit):
            return None
        best = int(hit[np.argmin(dist[hit])])  # nearest target; argmin keeps the lowest row on ties
        path = [best]
        while parent[path[-1]] >= 0:
            path.append(int(parent[path[-1]]))
        return int(dist[best]), path[::-1]

    def reachable_within(self, sources, max_hops: int, removed=None) -> set[int]:
        """Rows reachable from the sources in at most `max_hops` hops (sources included)."""
        rem = _rows(removed, self.n) if removed is not None else None
        dist, _ = self._bfs(_rows(sources, self.n), max_hops, rem)
        return set(np.flatnonzero(dist >= 0).tolist())

    def lesion_effect(self, sources, targets, removed, max_hops: int = 8) -> dict:
        """How many targets stay reachable from the sources once `removed` neurons are deleted."""
        tgt = _rows(targets, self.n)
        rem = _rows(removed, self.n)
        src = _rows(sources, self.n)
        before, _ = self._bfs(src, max_hops)
        after, _ = self._bfs(src, max_hops, rem)
        was, now = before[tgt] >= 0, after[tgt] >= 0
        lost_rows = tgt[was & ~now]
        return {
            "targets_total": int(len(tgt)),
            "targets_reachable_before": int(was.sum()),
            "targets_reachable_after": int(now.sum()),
            "lost": int(len(lost_rows)),
            "lost_rows": lost_rows,
            "sample_lost": lost_rows[:10].tolist(),
        }

    # -- hubs --------------------------------------------------------------------------

    def hubs(self, metric: str = "total", top: int = 10, candidate_rows=None):
        """Most connected neurons by summed synapse strength (in, out or total). (rows, values)."""
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {', '.join(METRICS)}, got {metric!r}")
        strength = {"in": self._in_strength, "out": self._out_strength,
                    "total": self._in_strength + self._out_strength}[metric]
        rows = _rows(candidate_rows, self.n) if candidate_rows is not None else np.arange(self.n)
        return _sorted_desc(rows, strength[rows], top)
