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
MAX_ALT_SOLVES = 60  # bound on the DAG passes spent looking for alternative routes


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

    # -- strongest shortest path -------------------------------------------------------

    def _layered_edges(self, dist: np.ndarray, depth: int, tgt_mask: np.ndarray):
        """Edges (u, v, w) that go from BFS layer d to d+1, one triple per layer d < depth."""
        indptr, indices, data = self._out.indptr, self._out.indices, self._out.data
        layers = []
        for d in range(depth):
            frontier = np.flatnonzero(dist == d)
            starts, ends = indptr[frontier], indptr[frontier + 1]
            counts = ends - starts
            total = int(counts.sum())
            pos = np.repeat(starts - np.cumsum(counts) + counts, counts) + np.arange(total)
            u = np.repeat(frontier, counts)
            v = indices[pos].astype(np.int64)
            w = data[pos].astype(np.int64)
            keep = dist[v] == d + 1
            if d + 1 == depth:
                keep &= tgt_mask[v]  # only the targets at the final layer end a shortest path
            layers.append((u[keep], v[keep], w[keep]))
        return layers

    def _prune(self, layers, tgt: np.ndarray, blocked: np.ndarray | None):
        """Keep only edges that lie on some source-to-target route (backward pass from targets)."""
        good = np.zeros(self.n, dtype=bool)
        good[tgt] = True
        pruned = [None] * len(layers)
        for d in range(len(layers) - 1, -1, -1):
            u, v, w = layers[d]
            keep = good[v]
            if blocked is not None:
                keep &= ~blocked[u] & ~blocked[v]
            u, v, w = u[keep], v[keep], w[keep]
            good[u] = True
            pruned[d] = (u, v, w)
        return pruned

    def _best_route(self, layers, src: np.ndarray, tgt: np.ndarray):
        """Best (bottleneck, total, rows, hop weights) over pruned layers, or None.

        Two exact passes: a widest-path DP (max of min) gives the best bottleneck; then, using
        only edges at least that strong, a max-sum DP gives the best total. Doing it in two
        passes matters: a (bottleneck, total) pair is not safe to optimize prefix by prefix.
        Ties go to the lowest row at every step.
        """
        width = np.zeros(self.n, dtype=np.int64)
        width[src] = np.iinfo(np.int64).max
        for u, v, w in layers:
            if len(u):
                np.maximum.at(width, v, np.minimum(width[u], w))
        bottleneck = int(width[tgt].max())
        if bottleneck <= 0:
            return None
        total = np.full(self.n, -1, dtype=np.int64)
        parent = np.full(self.n, -1, dtype=np.int64)
        parent_w = np.zeros(self.n, dtype=np.int64)
        total[src] = 0
        for u, v, w in layers:
            ok = (w >= bottleneck) & (total[u] >= 0)
            u, v, w = u[ok], v[ok], w[ok]
            if not len(u):
                continue
            cand = total[u] + w
            order = np.lexsort((u, -cand, v))  # per v: highest total, then lowest parent row
            v_s = v[order]
            first = np.concatenate(([True], v_s[1:] != v_s[:-1]))
            pick = order[first]
            total[v[pick]], parent[v[pick]], parent_w[v[pick]] = cand[pick], u[pick], w[pick]
        reached = tgt[total[tgt] >= 0]
        best = int(reached[np.argmax(total[reached])])  # first max = lowest row
        rows, weights = [best], []
        while parent[rows[-1]] >= 0:
            weights.append(int(parent_w[rows[-1]]))
            rows.append(int(parent[rows[-1]]))
        return bottleneck, int(total[best]), rows[::-1], weights[::-1]

    def strongest_paths(self, sources, targets, max_hops: int = 10, n_alternatives: int = 3):
        """Strongest of the shortest directed routes, how many shortest routes exist, and rivals.

        Among all routes with the minimum hop count, picks the one with the largest bottleneck
        (weakest hop, in synapses), then the largest total, then the lowest rows. Returns None
        if no target is within `max_hops`, else a dict: hops, path (rows), hop_synapses,
        min_hop_synapses (None at 0 hops), total_synapses, path_count (distinct shortest
        routes; float-exact up to 2**53) and alternatives (up to `n_alternatives` more routes,
        strongest first, each differing from the others in at least one intermediate neuron).
        """
        src, tgt = _rows(sources, self.n), _rows(targets, self.n)
        if not len(src) or not len(tgt):
            return None
        dist, _ = self._bfs(src, max_hops, stop_at=tgt)
        hit = tgt[dist[tgt] >= 0]
        if not len(hit):
            return None
        depth = int(dist[hit].min())
        tgt = hit[dist[hit] == depth]
        if depth == 0:
            both = np.intersect1d(src, tgt)
            return {"hops": 0, "path": [int(both[0])], "hop_synapses": [], "min_hop_synapses": None,
                    "total_synapses": 0, "path_count": int(len(both)), "alternatives": []}
        tgt_mask = np.zeros(self.n, dtype=bool)
        tgt_mask[tgt] = True
        layers = self._prune(self._layered_edges(dist, depth, tgt_mask), tgt, None)
        bottleneck, total, rows, weights = self._best_route(layers, src, tgt)
        count = np.zeros(self.n)
        count[src] = 1.0
        for u, v, _ in layers:
            np.add.at(count, v, count[u])
        result = {"hops": depth, "path": rows, "hop_synapses": weights, "min_hop_synapses": bottleneck,
                  "total_synapses": total, "path_count": int(count[tgt].sum()), "alternatives": []}
        if depth < 2 or n_alternatives <= 0:
            return result
        # best-first search over "these intermediates are banned" subproblems; a route's tuple of
        # intermediates is reported once, and every popped route spawns one child per intermediate
        first = (bottleneck, total, rows, weights)
        queue, tried, solves = [(first, frozenset())], {frozenset()}, 0
        seen: set[tuple] = set()
        while queue and len(result["alternatives"]) < n_alternatives and solves < MAX_ALT_SOLVES:
            queue.sort(key=lambda q: (-q[0][0], -q[0][1], q[0][2]))
            (bott, tot, route, wts), banned = queue.pop(0)
            if tuple(route[1:-1]) not in seen:
                seen.add(tuple(route[1:-1]))
                if banned:
                    result["alternatives"].append(
                        {"path": route, "hop_synapses": wts, "min_hop_synapses": bott, "total_synapses": tot})
            for x in route[1:-1]:
                child = banned | {x}
                if child in tried:
                    continue
                tried.add(child)
                solves += 1
                mask = np.zeros(self.n, dtype=bool)
                mask[list(child)] = True
                found = self._best_route(self._prune(layers, tgt, mask), src, tgt)
                if found is not None:
                    queue.append((found, child))
        return result

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
