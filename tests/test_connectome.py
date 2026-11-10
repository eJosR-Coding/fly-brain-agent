import numpy as np
import pytest

from flybrain.connectome import Connectome, ConnectomeError
from flybrain.connectome_build import write_npz

# Tiny structural graph on 12 neurons (rows 0..11):
#   chain      0 -> 1 -> 2 -> 3
#   branch     1 -> 4 -> 3           (a second 3-hop route 0-1-4-3; ties go to the lowest row)
#   cycle      5 -> 6 -> 7 -> 5
#   weak edge  0 -> 8  (2 synapses, below the default threshold of 5)
#   hub        9 -> 0, 1, 2, 5 (strong), and 10 -> 9
EDGES = [
    (0, 1, 10), (1, 2, 10), (2, 3, 10), (1, 4, 8), (4, 3, 8),
    (5, 6, 7), (6, 7, 7), (7, 5, 7),
    (0, 8, 2),
    (9, 0, 20), (9, 1, 20), (9, 2, 20), (9, 5, 20), (10, 9, 6),
]
N = 12


@pytest.fixture
def conn():
    pre, post, syn = (np.array(c, dtype=np.int32) for c in zip(*EDGES))
    return Connectome.from_edges(pre, post, syn, n_neurons=N, min_synapses=5)


def test_edges_below_the_threshold_are_dropped(conn):
    assert conn.n_edges == len(EDGES) - 1
    assert conn.min_synapses == 5
    low = Connectome.from_edges(*(np.array(c, dtype=np.int32) for c in zip(*EDGES)), n_neurons=N, min_synapses=1)
    assert low.n_edges == len(EDGES)


def test_neighbors_out_sorted_by_synapses_excluding_inputs(conn):
    rows, syn = conn.neighbors([0, 1], "out")
    assert dict(zip(rows.tolist(), syn.tolist())) == {2: 10, 4: 8}  # 0->1 and 1's own row excluded
    assert syn.tolist() == sorted(syn.tolist(), reverse=True)


def test_neighbors_in_and_both(conn):
    rows, syn = conn.neighbors([3], "in")
    assert dict(zip(rows.tolist(), syn.tolist())) == {2: 10, 4: 8}
    rows, syn = conn.neighbors([1], "both")
    assert dict(zip(rows.tolist(), syn.tolist())) == {0: 10, 9: 20, 2: 10, 4: 8}


def test_neighbors_top_truncates_and_ties_break_by_row(conn):
    rows, syn = conn.neighbors([9], "out", top=3)
    assert len(rows) == 3
    assert rows.tolist() == [0, 1, 2]  # all 20 synapses: lowest rows first


def test_neighbors_bad_direction(conn):
    with pytest.raises(ValueError, match="direction"):
        conn.neighbors([0], "sideways")


def test_shortest_path_is_deterministic_and_minimal(conn):
    hops, path = conn.shortest_path([0], [3])
    assert hops == 3 and path == [0, 1, 2, 3]  # not 0-1-4-3: tie goes to the lowest row


def test_shortest_path_multi_source_multi_target(conn):
    hops, path = conn.shortest_path([10, 0], [3, 7])
    assert hops == 3 and path[0] in (0, 10) and path[-1] in (3, 7)
    assert len(path) == hops + 1


def test_shortest_path_ignores_edges_below_threshold(conn):
    assert conn.shortest_path([0], [8]) is None


def test_shortest_path_respects_max_hops(conn):
    assert conn.shortest_path([0], [3], max_hops=2) is None
    assert conn.shortest_path([0], [3], max_hops=3)[0] == 3


def test_shortest_path_source_is_target_is_zero_hops(conn):
    assert conn.shortest_path([2], [2, 3]) == (0, [2])


def test_shortest_path_is_directed(conn):
    assert conn.shortest_path([3], [0]) is None


def test_shortest_path_survives_a_cycle(conn):
    assert conn.shortest_path([5], [7]) == (2, [5, 6, 7])


def test_reachable_within_counts_hops_and_skips_removed(conn):
    assert conn.reachable_within([0], 1) == {0, 1}
    assert conn.reachable_within([0], 3) == {0, 1, 2, 3, 4}
    assert conn.reachable_within([0], 3, removed=[1]) == {0}


def test_reachable_within_ignores_removed_sources(conn):
    assert conn.reachable_within([0, 5], 1, removed=[0]) == {5, 6}


def test_lesion_effect_reports_before_after_and_lost(conn):
    out = conn.lesion_effect([0], [3, 4], removed=[1], max_hops=5)
    assert out["targets_total"] == 2
    assert out["targets_reachable_before"] == 2
    assert out["targets_reachable_after"] == 0
    assert out["lost"] == 2
    assert sorted(out["lost_rows"].tolist()) == [3, 4]


def test_lesion_effect_with_a_redundant_route_loses_only_one(conn):
    out = conn.lesion_effect([0], [3], removed=[2], max_hops=5)  # 0-1-4-3 still works
    assert out["lost"] == 0 and out["targets_reachable_after"] == 1


def test_lesion_effect_removed_target_counts_as_lost(conn):
    out = conn.lesion_effect([0], [3, 2], removed=[2], max_hops=5)
    assert out["targets_reachable_before"] == 2 and out["targets_reachable_after"] == 1
    assert out["lost_rows"].tolist() == [2]


def test_hubs_by_metric(conn):
    rows, val = conn.hubs("out", top=1)
    assert rows.tolist() == [9] and val.tolist() == [80]
    rows, val = conn.hubs("in", top=2)
    assert rows.tolist()[0] in (1, 2)  # row 1 has in 10+20=30, row 2 has 10+20=30; lowest row wins
    assert rows.tolist()[0] == 1
    rows, val = conn.hubs("total", top=1)
    assert rows.tolist() == [9]


def test_hubs_restricted_to_candidates(conn):
    rows, val = conn.hubs("out", top=5, candidate_rows=[0, 1, 10])
    assert rows.tolist() == [1, 0, 10] and val.tolist() == [18, 10, 6]


def test_hubs_bad_metric(conn):
    with pytest.raises(ValueError, match="metric"):
        conn.hubs("betweenness", top=3)


def test_load_roundtrip_applies_the_threshold(tmp_path):
    pre, post, syn = (np.array(c, dtype=np.int32) for c in zip(*EDGES))
    path = tmp_path / "c.npz"
    write_npz(path, pre, post, syn, n_neurons=N, source="x", stats={"rows_in": 1, "dropped_unknown": 0, "dropped_self_loops": 0})
    assert Connectome.load(path, n_neurons=N).n_edges == len(EDGES) - 1
    assert Connectome.load(path, n_neurons=N, min_synapses=9).n_edges == 7


def test_load_missing_file_says_how_to_build_it(tmp_path):
    with pytest.raises(ConnectomeError, match="flybrain-build-connectome"):
        Connectome.load(tmp_path / "nope.npz", n_neurons=N)


def test_load_rejects_a_table_built_for_another_atlas(tmp_path):
    pre, post, syn = (np.array(c, dtype=np.int32) for c in zip(*EDGES))
    path = tmp_path / "c.npz"
    write_npz(path, pre, post, syn, n_neurons=N, source="x", stats={"rows_in": 1, "dropped_unknown": 0, "dropped_self_loops": 0})
    with pytest.raises(ConnectomeError, match="atlas"):
        Connectome.load(path, n_neurons=999)


# -- strongest shortest path -----------------------------------------------------------
# 0 -> T(3) in two hops through four middle neurons, plus a longer detour that must be ignored:
#   via 1: 50 then 5   (bottleneck 5,  total 55)
#   via 2: 20 then 20  (bottleneck 20, total 40)
#   via 6: 20 then 20  (bottleneck 20, total 40: ties with via 2, lowest row wins)
#   via 7: 20 then 30  (bottleneck 20, total 50: best)
#   detour 0 -> 8 -> 9 -> 3 would be 3 hops
STRONG_EDGES = [
    (0, 1, 50), (1, 3, 5), (0, 2, 20), (2, 3, 20), (0, 6, 20), (6, 3, 20), (0, 7, 20), (7, 3, 30),
    (0, 8, 90), (8, 9, 90), (9, 3, 90),
]


def _make(edges, n):
    pre, post, syn = (np.array(c, dtype=np.int32) for c in zip(*edges))
    return Connectome.from_edges(pre, post, syn, n_neurons=n, min_synapses=5)


@pytest.fixture
def strong():
    return _make(STRONG_EDGES, 10)


def test_strongest_path_maximizes_bottleneck_then_total_among_shortest(strong):
    res = strong.strongest_paths([0], [3])
    assert res["hops"] == 2
    assert res["path"] == [0, 7, 3]
    assert res["hop_synapses"] == [20, 30]
    assert res["min_hop_synapses"] == 20 and res["total_synapses"] == 50


def test_strongest_path_never_trades_hops_for_strength(strong):
    res = strong.strongest_paths([0], [3])
    assert 8 not in res["path"] and res["hops"] == 2  # the 90-synapse detour is 3 hops


def test_shortest_path_count_counts_distinct_shortest_routes(strong):
    assert strong.strongest_paths([0], [3])["path_count"] == 4


def test_alternatives_are_ranked_by_strength_and_differ_in_an_intermediate(strong):
    alts = strong.strongest_paths([0], [3])["alternatives"]
    assert [a["path"] for a in alts] == [[0, 2, 3], [0, 6, 3], [0, 1, 3]]
    assert [a["min_hop_synapses"] for a in alts] == [20, 20, 5]
    assert alts[0]["total_synapses"] == 40 and alts[0]["hop_synapses"] == [20, 20]


def test_alternatives_are_capped(strong):
    assert len(strong.strongest_paths([0], [3], n_alternatives=1)["alternatives"]) == 1
    assert strong.strongest_paths([0], [3], n_alternatives=0)["alternatives"] == []


def test_strongest_path_does_not_greedy_pick_the_best_prefix():
    # both routes reach 4 and then share the 6-synapse hop to 5: the bottleneck ties at 6, so
    # the higher total (via 2, prefix bottleneck 6 but total 56) must beat via 1 (10 / 20)
    conn = _make([(0, 1, 10), (1, 4, 10), (0, 2, 50), (2, 4, 6), (4, 5, 6)], 6)
    res = conn.strongest_paths([0], [5])
    assert res["path"] == [0, 2, 4, 5] and res["hop_synapses"] == [50, 6, 6]
    assert res["min_hop_synapses"] == 6 and res["total_synapses"] == 62
    assert res["path_count"] == 2


def test_strongest_path_multi_source_multi_target_picks_the_strongest_pair():
    conn = _make([(0, 2, 6), (1, 3, 40), (0, 3, 7)], 4)
    res = conn.strongest_paths([0, 1], [2, 3])
    assert res["path"] == [1, 3] and res["min_hop_synapses"] == 40
    assert res["path_count"] == 3


def test_strongest_path_none_when_unreachable_or_too_far(strong):
    assert strong.strongest_paths([3], [0]) is None
    assert strong.strongest_paths([0], [3], max_hops=1) is None


def test_strongest_path_source_is_target_is_zero_hops(strong):
    res = strong.strongest_paths([2], [2, 3])
    assert res["hops"] == 0 and res["path"] == [2] and res["hop_synapses"] == []
    assert res["min_hop_synapses"] is None and res["total_synapses"] == 0
    assert res["path_count"] == 1 and res["alternatives"] == []


def test_strongest_path_single_hop_has_no_alternatives():
    res = _make([(0, 1, 9)], 2).strongest_paths([0], [1])
    assert res["path"] == [0, 1] and res["hop_synapses"] == [9] and res["alternatives"] == []


def test_strongest_path_is_deterministic(strong):
    assert strong.strongest_paths([0], [3]) == strong.strongest_paths([0], [3])
