# Feature: Level 1 experiments (whole-brain LIF simulation)

## Objective
Simulate activity spreading through the real connectome, replicating the published whole-brain LIF model (Shiu et al., Nature 634, 210-219, 2024, DOI 10.1038/s41586-024-07763-9; code github.com/philshiu/Drosophila_brain_model, MIT), so the agent can "give the fly sugar" and we can check it against the paper's reference implementation.

## Verified so far (2026-11-10)
- Model (repo model.py): dv/dt = (v0 - v + g)/t_mbr, dg/dt = -g/tau; spike when v > v_th -> v = v_rst, g = 0; v0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms, refractory 2.2 ms (0 for Poisson-driven), delay 1.8 ms, w = (Excitatory x Connectivity) * w_syn, w_syn = 0.275 mV; Poisson input per stimulated neuron (default 150 Hz, example uses 100 Hz) adding w_syn * 250 to v; t_run 1000 ms, 30 trials; readout = mean rate per neuron.
- Repo ships `Connectivity_783.parquet` (100.8 MB; Presynaptic_Index, Postsynaptic_Index, Excitatory x Connectivity) and `Completeness_783.csv` (3.3 MB; index -> root id). Paper used v630 by default.
- example.ipynb: 21 sugar GRN root ids (20/21 present in our v783 atlas: types LB3c 9, LB3d 5, LB4b 4, LB3b 2, all sensory) and MN9 = 720575940660219265 (present in v783, cell_type CB0701, motor).
- Not verified: paper Methods/Results numbers (Nature/bioRxiv blocked), the exact sign rule behind the parquet column, the "activated" threshold.

## Design
- Data: download the two v783 files from the repo (MIT; record commit SHA + license) into data/raw/shiu/ (gitignored); map their indices to our atlas rows via root ids.
- Core `flybrain.simulation`: exact LIF as above in numpy/scipy sparse (dt = 0.1 ms, delay ring buffer, spike-driven sparse updates), seeded; `run(stimulate_rows, rate_hz, duration_ms, trials, silence_rows)` -> spike counts per neuron per trial + optional time-binned counts for the viewer.
- Validation: run the reference Brian2 code (same 783 files, same stimulus: sugar ids at 100 Hz, 1000 ms) in an isolated env for a few trials, compare per-neuron mean rates with ours (correlation over neurons that fire; MN9 rate; set of top-20 responders). Keep the machine responsive: limit parallelism (the user's PC struggles under load).
- Later tasks: MCP tool `stimulate` (+ silence), API endpoint for activity frames, viewer animation.

## Constraints
- Strict TDD on the core with tiny synthetic networks (analytic checks: subthreshold decay, single spike timing, refractory, delay, inhibition).
- Deps in their own `chore(deps)` commit; Brian2 only in an isolated validation env (not a project dependency).
- Honesty: label results "LIF model (Shiu et al. 2024 parameters), not the living fly"; no neuromodulation/plasticity; no VNC.

## TDD
- Mode: strict (source: global user config). Runner: `uv run pytest`.

## Tasks
- [x] S1 Data download + mapping, LIF core, validation against the reference Brian2 implementation. Route: delegated writer. Done 2026-11-10: validation agrees within stochastic noise.
- [x] S2 MCP tools `preset_selection` + `stimulate` (with silencing), API activity endpoints. Route: delegated writer. Done 2026-11-10.
- [x] S3 Viewer: activity wave animation and a "give sugar" chip. Route: delegated writer. Done 2026-11-10.

## Progress / evidence
### S1 data facts (reference repo commit 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960, MIT, 2024-09-14; files in data/raw/shiu/, gitignored)
- `Completeness_783.csv`: 138,639 root ids (index column, all Completed=True). `Presynaptic_Index`/`Postsynaptic_Index` are positions in that CSV's row order (checked: `Presynaptic_ID == ids[Presynaptic_Index]` for all 15,091,983 edges).
- 138,625 of those map to our atlas (139,248 neurons; 14 theirs missing from ours, 623 ours missing from theirs). 15,090,883 edges have both ends in our atlas (1,100 dropped). No self-loops, no zero weights.
- Edges: 9,059,302 excitatory, 6,032,681 negative (inhibitory); sum |w| = 54,492,922 synapses; net sum w = 10,496,512. `Excitatory x Connectivity` = `Excitatory` (+1/-1) x `Connectivity` (synapse count, 1..2405) exactly.
- Sign rule: `Excitatory` is constant per presynaptic neuron (0 of 138,005 mixed). Against our atlas `top_nt` (per presynaptic neuron, + / -): acetylcholine 84,162 / 1,719; dopamine 5,897 / 3; octopamine 192 / 1; serotonin 2,127 / 9; GABA 1,348 / 17,731; glutamate 2,941 / 21,863. So roughly: acetylcholine and monoamines excitatory, GABA and glutamate inhibitory, with a 2-12% per-neuron disagreement that our `top_nt` cannot explain (the reference sign is the one used; our atlas `top_nt` is not used by the simulator).
- Reference gotchas: the example's 21 sugar ids come from v630; `720575940620900446` is absent from the v783 completeness table (reference `run_exp` raises KeyError on it) and from our atlas, so both runs stimulate the same 20 ids. Brian2 reference with 15M synapses needs ~3 GB RAM per process and one process's memory grows across trials (6 trials in one process hit `bad_alloc` under a 7 GB cap), so it was run as separate single-trial processes.

### S1 validation (sugar GRNs, 20 ids, 100 Hz, 1000 ms; reference = Brian2 2.11.0 default clock dt = 0.1 ms, method linear)
| Metric | Reference (6 trials) | Ours (30 trials) |
|---|---|---|
| MN9 (720575940660219265) mean rate | 61.67 Hz | 61.67 Hz |
| Neurons with rate > 0 | 376 | 395 (the extra 19 all <= 0.47 Hz, i.e. at most 14 spikes in 30 trials; the reference missed them with 6 trials) |
| Neurons with rate > 1 Hz | 341 | 342 |
| Total spikes per trial | 8,978 | 9,003 |
| Pearson / Spearman (395 neurons active in either) | 0.999 / 0.997 | |
| Top-20 / top-100 overlap | 18 / 20, 99 / 100 | |
Noise floor: our own 30-trial run vs a 6-trial run with another seed gives the same top-20 18/20, top-100 99/100, Pearson 0.999, so the remaining differences are stochastic. Tiny-network spike trains (Poisson every step, delay, refractory, inhibition) match Brian2 exactly and are pinned in `tests/test_simulation.py`.
Semantics matched after calibrating on a 4-neuron Brian2 net: per tick update, threshold, arrivals + Poisson, reset last (an arrival or kick on a spiking tick is wiped); refractory neurons ignore synaptic arrivals (g stays clamped) and are ready again 22 ticks after the spike.
Runtimes: reference 82 s for the first trial (includes Cython compile), then ~8 s per trial, ~3 GB RSS; ours ~4.8 s per simulated second (6 trials in 29 s; 30 trials in 188 s while the machine was heavily loaded), ~1.5 GB RSS, 3 s to load the files and build the CSR.
Commits: `feat(simulation): leaky integrate-and-fire whole-brain simulator`, `feat(simulation): validation script against the reference Brian2 model`. TDD: tests written first, RED = `ModuleNotFoundError: No module named 'flybrain.simulation'`; first GREEN run had 1 failure (refractory ticks and arrival gating vs the Brian2 trace), fixed after tracing Brian2 state. `uv run pytest -q`: 315 passed (298 before).
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

- S1 review (2026-11-10): main..7624d83 (553 lines, medium), granted, approved, acknowledged. Follow-ups for S2: bound the time-binned allocation (bins x neurons can be huge; simulation.py:154) and add a from_files test on a tiny parquet/csv fixture (simulation.py:105-125); cache the CSR as npz.
- Cleanup: removed the isolated Brian2 env (/tmp/claude-1000/shiu-ref, 369 MB) and 4 orphaned wait loops left by the validation run.

### S2 evidence (branch feat/simulation-tools)
- Follow-ups closed: time bins are sparse (`active_rows` + `frames` uint16, <= 200 bins, a finer request raises), `from_files` has a test on a tiny parquet + csv, the CSR is cached in `data/processed/sim_weights_783.npz` (`flybrain-build-sim-weights`; real data: 2.41 s without cache, 0.19 s with it, 121 MB).
- RED evidence: sparse/cache tests `6 failed` (AttributeError `active_rows` / `load_weights`); `tests/test_sim_weights.py` ImportError `flybrain.sim_service`; `tests/test_stimulate.py` ImportError `SimulationService`; `tests/test_api_simulation.py` `18 failed` (`create_app() got an unexpected keyword argument 'simulation'`). Then GREEN; `uv run pytest -q`: 359 passed (315 on main).
- Presets in `src/flybrain/data/presets.json` (sugar: 21 ids, 20 in v783; mn9), each with its source (commit 91bdd1e, MIT).
- Real data, no LLM, one run at a time (`stimulate` on the real atlas and weights): `sugar` 20 neurons, `mn9` 1 neuron. Run 1 (100 Hz, 1000 ms, 3 trials, readout mn9): 13.1 s including the first weight load, 336 neurons above 1 Hz, MN9 60.3 Hz (validated 61.7), payload 1,485 chars, 373 recorded rows x 100 bins = 38.8 KB of frames. Only one non-sensory neuron in the top 10: CB0248 (central, 107.3 Hz). Run 2 (silence CB0248 + baseline): 25.7 s, payload 1,766 chars, MN9 60.3 -> 63.7 Hz (+3.3 Hz, +5.5%), active 336 -> 340: no evidence that CB0248 gates MN9, and a 3-trial difference of this size is within noise (same seed for baseline and silenced run).
- Design notes: `runtime_s` includes the lazy weight load on the first call; the tool seeds are fixed (seed 0) so a repeated call returns the same numbers; the activity recording keeps every neuron that spiked (not only > 1 Hz), counts clipped to 255.

### S3 evidence
- Pure logic in `web/src/activity.ts` (frames decode with size validation, brightness per bin with a decaying trail, summed frame for reduced motion, playback state, texts, wire parsers, `loadActivity`) with `web/test/activity.test.ts`; vitest went 102 -> 133 tests. RED: the activity suite failed to import before the module existed; the chat/status additions went RED first (3 failures). Exception: the tiny `scatterBrightness` helper was written together with its test.
- Scene: an additive activity layer (indexed Points sharing the position buffer, per-point `aBright` attribute), the base cloud dims to 40% while it plays; timeline (play/pause, thin range bar, `t = N ms`) above the legend; "give sugar" chip; chat `activity` events play too; legend `activity · N neurons · MN9 60 Hz`; reduced motion shows the summed activity as one still frame; all style tokens in `style.css`.
- Visual check (headless Chromium 1440x900, swiftshader, built app on a throwaway preview with a temporary config outside the repo, backend on port 8020 without a model, servers stopped afterwards): captures `~/Downloads/fly-brain-capturas/v3-azucar-1.png` (t = 120 ms) and `v3-azucar-2.png` (t = 800 ms). The sugar-driven neurons show as bright flecks (mostly the subesophageal region and the central brain) over the dimmed cloud, the activity differs between the two times, and the legend, timeline and chips are legible. No console errors. The effect is subtle at this zoom because only 336 of 139,248 neurons are active; a bigger point size (token `--activity-size-scale: 6`) and dimming (`--activity-dim: 0.4`) were tuned from the first capture.

- S2/S3 review (2026-11-10): 4 slices (346 / 494 HIGH 4 lenses / 255 / 867 lines), all granted, approved, acknowledged. Advisory follow-ups: weight-cache signature is weak (simulation.py:155; should hash sources), load_simulation untested (sim_service.py:61), duplicate summary helpers and a hardcoded 1 Hz threshold in tools.py.

## Next step
Ship the branch (PR). Ideas: a "silence" chip in the viewer, a per-region summary of the activity, and a simulated-vs-structural comparison tool.
