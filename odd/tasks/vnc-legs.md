# VNC legs: the front leg follows a connectome VNC model driven by the brain's walking command neuron

## Objective
After ANY brain simulation, the fly body's front left leg moves when (and only when) the simulated brain drives the walking
command descending neuron DNg100 strongly enough: brain DNg100 rate -> precomputed ventral nerve cord (VNC) rate-model
response -> leg motor neuron traces -> one hip joint. The user can also activate DNg100 directly (a `dng100` preset usable by
the agent through `stimulate`/`fly`, and a chip). Requested 2026-11-17 ("por ambos ... deberia interactuar con cualquier
estimulo independientemente").

## Why / evidence
- VNC model: Pugliese et al. 2026, Nature, "Connectome simulations identify a central pattern generator for fly walking"
  (https://www.nature.com/articles/s41586-026-11195-y), code https://github.com/smpuglie/Pugliese_cpg_2025 (MIT), commit
  10e7661, bundled MANC T1 data (`data/manc t1 connectome data/W_20250813_DNtoMN_unsorted.csv`).
- Reproduced locally (2026-11-17, CPU, 4 replicates): DNg100 step 250 -> 2-4 rhythmic front-left leg MNs at ~10-11 Hz
  (paper: 2-10 MNs, ~7-15 Hz); silencing E1 (IN17A001) or E2 (INXXX466) abolishes the rhythm in 4/4; silencing I1
  (IN16B036) does not (paper: I1 is redundant). DNg62 (driven in our brain by jo_ce 200 Hz) gives tonic, non-rhythmic
  output (0 rhythmic MNs at 65/150/250); DNg97 inconsistent (0 and 5 rhythmic in 2 reps).
- Stimulus threshold: 150 -> nothing; 250 -> rhythm (finer sweep 175/200/225/350/500 in progress).
- Our brain (FlyWire v783 LIF) contains DNg100 (2), DNb08 (4), DNg97 (2), DNg74 (4), DNg62 (2). No packaged sensory
  preset drives DNg100 meaningfully (max 1.8 Hz with water 200 Hz).
- The MN -> movement assignment comes from the paper's table (`motor module`, `step contribution` columns): the rhythmic
  MNs are coxa promotors (module "coxa swing", phase "swing").

## Honesty rules (hard)
- Two different models: the brain is the Shiu et al. 2024 spiking LIF model (FlyWire, female); the VNC is a rate model on
  the MANC (male) connectome. Linking them by cell type name (DNg100) and the scale from brain Hz to VNC input are OUR
  choices and are labelled as such. The VNC response is PRECOMPUTED at fixed input levels (no live VNC simulation).
- Only the front left leg, only the hip (coxa) swing; one joint for the whole leg (the rig has no joint chains). No gait,
  no other legs, no claim of walking behaviour. Never "the fly walks/feels/decides".
- With today's sensory presets the legs stay still: that is the model's result, shown honestly.
- No forbidden words (die, dead, death, overdose, poison, kill, suffer, pain, feel) in captions/notes.

## Tasks
- [ ] V1 data: script to precompute VNC traces at several DNg100 input levels from the external Pugliese checkout
  (not a project dependency) + packaged `src/flybrain/data/vnc_dng100.json` with provenance (repo, commit, config, seed,
  data sha, levels, per-level traces of the coxa promotor/remotor MNs and the rhythm frequency), MIT attribution.
- [ ] V2 rig: `leg_lf_l` joint (all `lf_*` parts, pivot at the `lf_coxa` origin, axis for fore-aft swing, "geometry (our
  choice)") in the asset build, regenerated `fly-body.json` (bins byte-identical), build tests.
- [ ] V3 backend: `dng100` stimulus preset; after every stimulate/fly run, measure brain DNg100 (left) mean rate, map to a
  VNC level (our scale; below threshold -> no leg readout), attach named readouts (e.g. `vnc_coxa_swing_lf`) to the
  recording; tests.
- [ ] V4 web: MAPPING rows for the VNC readouts (the angle follows the trace; label "VNC rate model (Pugliese et al.
  2026), linked by us"), caption, a chip for DNg100; tests.
- [ ] V5 verification: renders of the leg through canvas -> image at several times; live check by the user.

## Checks
TDD strict (project config). Runners: `uv run pytest -q` (893 + 1 skipped on main 830d7f6), `pnpm -C web test` (690),
`pnpm -C web build`. Tool-def budget unchanged (no new agent tool). Delivery: ask-on-risk; chain strategy as before
(stacked PRs to main) if over ~400 authored lines.

## Progress
- 2026-11-17 created on branch feat/vnc-legs after research, local reproduction and code mapping. Route: V1 inline
  (scratch compute + one data file), V2-V4 delegated writer (2+ non-trivial files).
