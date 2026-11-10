"""Compare our LIF simulator with the reference Brian2 model (Shiu et al. 2024). Never run by pytest.

Three steps, so the heavy reference stays in its own isolated environment (Brian2 is NOT a
project dependency):

  # 1. reference (run with a python that has brian2, pandas, pyarrow, joblib, numpy)
  python scripts/validate_against_reference.py reference --shiu-dir data/raw/shiu \\
      --out /tmp/ref1.parquet --trials 1 --n-proc 1   # repeat with ref2.parquet, ... (~80 s and ~3 GB each)
  # 2. ours (project env)
  uv run python scripts/validate_against_reference.py ours --shiu-dir data/raw/shiu \\
      --annotations data/raw/flywire_neuron_annotations.tsv --out /tmp/ours.npz --trials 30
  # 3. compare (project env)
  uv run python scripts/validate_against_reference.py compare --reference /tmp/ref1.parquet /tmp/ref2.parquet \\
      --ours /tmp/ours.npz

Stimulus: the 21 sugar-sensing GRN root ids of the reference example.ipynb at 100 Hz for 1000 ms.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265
RATE_HZ, DURATION_MS = 100.0, 1000.0


def run_reference(args) -> None:
    shiu = Path(args.shiu_dir).resolve()
    sys.path.insert(0, str(shiu))  # the reference model.py lives next to the data
    import pandas as pd
    from brian2 import Hz

    import model

    comp = set(pd.read_csv(shiu / "Completeness_783.csv", index_col=0).index)
    stim = [i for i in SUGAR if i in comp]  # the reference raises KeyError on ids absent from v783
    print(f"stimulus ids absent from the v783 completeness table (skipped): {sorted(set(SUGAR) - set(stim))}")
    params = dict(model.default_params, n_run=args.trials, r_poi=RATE_HZ * Hz)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.run_exp("sugar", stim, str(out.parent), shiu / "Completeness_783.csv",
                  shiu / "Connectivity_783.parquet", params=params, n_proc=args.n_proc,
                  force_overwrite=True)
    (out.parent / "sugar.parquet").replace(out)
    df = pd.read_parquet(out)
    print(f"reference: {args.trials} trials, n_proc={args.n_proc}, {time.time() - t0:.0f} s, {len(df):,} spikes")


def run_ours(args) -> None:
    from flybrain.atlas import RowIndex, load_atlas
    from flybrain.simulation import Simulation

    shiu = Path(args.shiu_dir)
    atlas = load_atlas(args.annotations)
    sim = Simulation.from_files(shiu / "Connectivity_783.parquet", shiu / "Completeness_783.csv", atlas)
    rows = RowIndex(atlas).positions(np.array(SUGAR))
    missing = [i for i, r in zip(SUGAR, rows) if r < 0]
    t0 = time.time()
    res = sim.run(rows[rows >= 0], RATE_HZ, duration_ms=DURATION_MS, trials=args.trials, seed=args.seed)
    secs = time.time() - t0
    np.savez(args.out, counts=res.counts, root_ids=atlas.neurons["root_id"].to_numpy(),
             trials=args.trials, seconds=secs, missing=np.array(missing, dtype=np.int64))
    print(f"ours: {args.trials} trials, {secs:.0f} s ({secs / args.trials:.1f} s per simulated second), "
          f"{int(res.counts.sum()):,} spikes, stimulus ids missing from atlas: {missing}")


def _rates(ids, counts, trials) -> dict:
    return {int(i): c / trials for i, c in zip(ids, counts) if c > 0}


def top_overlap(a: dict, b: dict, k: int) -> int:
    top = lambda d: set(sorted(d, key=lambda i: (-d[i], i))[:k])  # noqa: E731
    return len(top(a) & top(b))


def run_compare(args) -> None:
    import pandas as pd
    from scipy import stats

    # one parquet per reference process: Brian2 keeps growing across trials in one process, so the
    # reference is run as several single-trial invocations whose files are pooled here
    ref = pd.concat([pd.read_parquet(f).assign(src=i) for i, f in enumerate(args.reference)])
    trials_ref = int(ref.groupby("src")["trial"].nunique().sum())
    per = ref.groupby("flywire_id").size()
    r_ref = {int(i): n / trials_ref for i, n in per.items()}  # Hz: 1 s trials
    z = np.load(args.ours)
    trials_ours = int(z["trials"])
    r_ours = _rates(z["root_ids"], z["counts"], trials_ours)

    print(f"trials: reference {trials_ref}, ours {trials_ours}; ours took {float(z['seconds']):.0f} s")
    print(f"stimulus ids missing from our atlas: {z['missing'].tolist()}")
    print(f"MN9 mean rate Hz: reference {r_ref.get(MN9, 0):.2f}, ours {r_ours.get(MN9, 0):.2f}")
    both = sorted(set(r_ref) | set(r_ours))
    x = np.array([r_ref.get(i, 0.0) for i in both])
    y = np.array([r_ours.get(i, 0.0) for i in both])
    print(f"neurons active in either: {len(both)} (reference {len(r_ref)}, ours {len(r_ours)})")
    print(f"Pearson {stats.pearsonr(x, y)[0]:.3f}, Spearman {stats.spearmanr(x, y)[0]:.3f}")
    print(f"top-20 overlap {top_overlap(r_ref, r_ours, 20)}/20, top-100 overlap {top_overlap(r_ref, r_ours, 100)}/100")
    print(f"rate > 0: reference {len(r_ref)}, ours {len(r_ours)}; "
          f"rate > 1 Hz: reference {sum(v > 1 for v in r_ref.values())}, ours {sum(v > 1 for v in r_ours.values())}")
    print(f"total spikes per trial: reference {sum(r_ref.values()):.0f}, ours {sum(r_ours.values()):.0f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference", help="run the Brian2 reference (needs brian2)")
    r.add_argument("--shiu-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--trials", type=int, default=5)
    r.add_argument("--n-proc", type=int, default=2)
    o = sub.add_parser("ours", help="run our simulator")
    o.add_argument("--shiu-dir", required=True)
    o.add_argument("--annotations", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--trials", type=int, default=30)
    o.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare", help="compare the two result files")
    c.add_argument("--reference", required=True, nargs="+", help="one or more reference parquet files")
    c.add_argument("--ours", required=True)
    args = p.parse_args(argv)
    {"reference": run_reference, "ours": run_ours, "compare": run_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
