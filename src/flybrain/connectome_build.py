"""Build step: FlyWire connection table (feather) -> compact (pre, post, syn) pair table in atlas rows.

Decisions (documented, not accidental):
- synapses are SUMMED per (pre, post) over all neuropils: one edge per neuron pair;
- pairs whose root_id is not in the atlas are DROPPED and counted (never guessed);
- self-loops (pre == post) are DROPPED and counted: they carry no information for neighbors,
  paths or lesions, and would only confuse "partners excluding the input";
- no synapse threshold here: the threshold is a load-time choice (Connectome.load).
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.atlas import RowIndex, load_atlas

PRE, POST, SYN = "pre_pt_root_id", "post_pt_root_id", "syn_count"
DEFAULT_OUT = "data/processed/connectome_783.npz"
DEFAULT_ANNOTATIONS = Path("data/raw/flywire_neuron_annotations.tsv")


def aggregate_pairs(df: pd.DataFrame, row_index: RowIndex):
    """Return (pre_rows, post_rows, syn, stats): int32 arrays, one entry per (pre, post) pair."""
    rows_in = len(df)
    pre = row_index.positions(df[PRE].to_numpy())
    post = row_index.positions(df[POST].to_numpy())
    syn = df[SYN].to_numpy(dtype="int64")
    known = (pre >= 0) & (post >= 0)
    dropped_unknown = int((~known).sum())
    pre, post, syn = pre[known], post[known], syn[known]
    loops = pre == post
    dropped_self_loops = int(loops.sum())
    pre, post, syn = pre[~loops], post[~loops], syn[~loops]

    n = len(row_index)
    key = pre.astype("int64") * n + post
    order = np.argsort(key, kind="stable")
    key, syn = key[order], syn[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.array([], dtype="int64")
    sums = np.add.reduceat(syn, starts) if len(key) else syn
    uniq = key[starts]
    out_pre = (uniq // n).astype(np.int32)
    out_post = (uniq % n).astype(np.int32)
    out_syn = sums.astype(np.int32)
    stats = {
        "rows_in": rows_in,
        "pairs_out": int(len(out_pre)),
        "dropped_unknown": dropped_unknown,
        "dropped_self_loops": dropped_self_loops,
    }
    return out_pre, out_post, out_syn, stats


def write_npz(path, pre, post, syn, n_neurons: int, source: str, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pre=pre, post=post, syn=syn,
        n_neurons=np.int64(n_neurons),
        source=np.str_(source),
        built_at=np.str_(datetime.now(timezone.utc).isoformat(timespec="seconds")),
        rows_in=np.int64(stats["rows_in"]),
        dropped_unknown=np.int64(stats["dropped_unknown"]),
        dropped_self_loops=np.int64(stats["dropped_self_loops"]),
    )


def build(source, annotations, out) -> dict:
    t0 = time.perf_counter()
    atlas = load_atlas(annotations)
    df = pd.read_feather(source, columns=[PRE, POST, SYN])
    pre, post, syn, stats = aggregate_pairs(df, RowIndex(atlas))
    del df
    write_npz(out, pre, post, syn, len(atlas.neurons), str(source), stats)
    stats["bytes"] = Path(out).stat().st_size
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the compact connectome pair table.")
    p.add_argument("--source", default=os.environ.get("FLYBRAIN_CONNECTIONS_SOURCE"),
                   help="FlyWire proofread_connections feather (or env FLYBRAIN_CONNECTIONS_SOURCE)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--annotations", default=os.environ.get("FLYBRAIN_ANNOTATIONS", DEFAULT_ANNOTATIONS))
    args = p.parse_args(argv)
    if not args.source:
        p.error("--source is required (or set FLYBRAIN_CONNECTIONS_SOURCE)")
    stats = build(args.source, args.annotations, args.out)
    print(
        f"rows in {stats['rows_in']:,}; pairs out {stats['pairs_out']:,}; "
        f"dropped unknown-id rows {stats['dropped_unknown']:,}; self-loops {stats['dropped_self_loops']:,}; "
        f"file {stats['bytes'] / 1e6:.1f} MB; {stats['seconds']} s -> {args.out}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
