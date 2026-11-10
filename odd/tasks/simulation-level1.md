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
- [ ] S1 Data download + mapping, LIF core, validation against the reference Brian2 implementation. Route: delegated writer.
- [ ] S2 MCP tool `stimulate` (+ `silence`), API activity endpoint. Route: delegated writer.
- [ ] S3 Viewer: activity wave animation and a "give sugar" chip. Route: delegated writer.

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

## Next step
S1.
