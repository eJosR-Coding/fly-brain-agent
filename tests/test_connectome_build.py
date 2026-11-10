import numpy as np
import pandas as pd

from flybrain.atlas import RowIndex
from flybrain.connectome_build import aggregate_pairs, write_npz


def _frame(rows):
    return pd.DataFrame(rows, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])


def test_synapses_are_summed_per_pair_across_neuropils(tiny_atlas):
    df = _frame([(1, 2, 3), (1, 2, 4), (2, 3, 6)])  # same pair twice (two neuropils)
    pre, post, syn, stats = aggregate_pairs(df, RowIndex(tiny_atlas))
    assert sorted(zip(pre.tolist(), post.tolist(), syn.tolist())) == [(0, 1, 7), (1, 2, 6)]
    assert stats["rows_in"] == 3 and stats["pairs_out"] == 2


def test_unknown_root_ids_are_dropped_and_counted(tiny_atlas):
    df = _frame([(1, 2, 5), (999, 2, 5), (1, 888, 5)])
    pre, post, syn, stats = aggregate_pairs(df, RowIndex(tiny_atlas))
    assert len(pre) == 1
    assert stats["dropped_unknown"] == 2


def test_self_loops_are_dropped_and_counted(tiny_atlas):
    df = _frame([(1, 1, 9), (1, 2, 5)])
    pre, post, syn, stats = aggregate_pairs(df, RowIndex(tiny_atlas))
    assert pre.tolist() == [0] and post.tolist() == [1]
    assert stats["dropped_self_loops"] == 1


def test_output_arrays_are_int32(tiny_atlas):
    pre, post, syn, _ = aggregate_pairs(_frame([(1, 2, 5)]), RowIndex(tiny_atlas))
    assert pre.dtype == post.dtype == syn.dtype == np.int32


def test_npz_round_trips_with_meta(tiny_atlas, tmp_path):
    pre, post, syn, stats = aggregate_pairs(_frame([(1, 2, 5)]), RowIndex(tiny_atlas))
    out = tmp_path / "c.npz"
    write_npz(out, pre, post, syn, n_neurons=len(tiny_atlas.neurons), source="x.feather", stats=stats)
    with np.load(out) as z:
        assert z["pre"].tolist() == [0] and z["syn"].tolist() == [5]
        assert int(z["n_neurons"]) == 63
        assert str(z["source"]) == "x.feather"
