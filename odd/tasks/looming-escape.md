# Looming escape experiment

## Objective
Test whether the connectome-driven LIF model tells a looming object from a receding one or from
uniform dimming, reading the giant fiber (DNp01), and show it visually.

## Phase 0 evidence (2026-11-11, tools on the real connectome)
- 2 DNp01; direct inputs from 99/210 LPLC2 and 89/207 LC4 (>= 5 synapses).
- R1-6 -> LPLC2 in 3 hops (L3 > Tm20); R1-6 -> DNp01 in 4 hops via LC4.
- Lesioning all LPLC2 loses 0% of R1-6 -> DNp01 reachability (LC4 is a parallel route).
- DNp01 also receives from 8 JO-A5 (Johnston's organ, hearing).
- Shortest routes skip T4/T5 (motion detectors), which in the real fly give LPLC2 its looming selectivity.

## Tasks
- [x] T1 time-varying stimulus: per-neuron rate schedule in the simulator (piecewise-constant
      rate per time window), exposed through the service; constant stimulation unchanged.
      (route: delegated writer)
      Done in d64c8bc: `StimulusSchedule` + `Simulation.run_schedule` + `SimulationService.run_schedule`
      (cached, own key tag). Full suite 688 passed; new tests/test_stimulus_schedule.py (constant
      run pinned, one-segment schedule bit-identical to `run`). Real brain, 500 ms, 400 neurons:
      constant 2.5 s vs 4-segment growing schedule 3.1 s.
- [x] T2 eye map: pick R1-6 inside a disc on the eye from their 3D positions; disc grows/shrinks.
      (route: delegated writer) Done in d9835be: src/flybrain/eye.py (`EyeMap` PCA 2D, `disc`,
      `eye_map(atlas, side)`, `looming_radii`, `disc_schedule`, `schedule_drive`); tests/test_eye.py 9 passed.
      Process note: the module was written before its tests (no observed RED for T2).
- [x] T3 looming experiment tool: looming vs receding vs dimming with equal total drive, DNp01
      readout, stats; plus lesion LPLC2+LC4 control.
      (route: delegated writer) Done in the looming_escape commit: src/flybrain/looming.py, MCP tool,
      added to EXPERIMENT_TOOLS (deadline extension), tool-defs budget 8400 -> 8850 (+547 chars).
      RED: tests/test_looming.py failed at collection (no flybrain.looming); GREEN: 12 passed; full
      suite 709 passed. Output 2151 B on the real run.
      REAL RUN (defaults, 5 trials, right eye, 4029 R1-6, 47 s): NULL RESULT. DNp01, LPLC2 and LC4 all
      0.0 Hz in all four conditions (all p = 1). R1-6 weights: 71.2% excitatory, 21.4% inhibitory
      (rest no outgoing). Diagnostics: constant 300 Hz on the whole right eye for 300 ms still gives
      LPLC2/LC4/DNp01 = 0 Hz (L3 4.2 Hz, Tm20 0.01 Hz), while driving LPLC2 at 100 Hz gives DNp01
      139 Hz and LC4 at 100 Hz gives 85 Hz: the readout works, the signal dies in the optic lobe.
- [x] T3b detector input: `looming_escape(input="detectors"|"photoreceptors")`, default detectors. Detectors path
      builds the same PCA EyeMap from LPLC2+LC4 of the chosen side and drives the rows inside the growing disc
      (looming), reversed (receding), all of them at the equal-drive rate (dimming); control = looming on the same
      detectors in SHUFFLED order (same rows per window, random, fixed seed). Photoreceptor path unchanged
      (control = LPLC2+LC4 silenced). Output has `input` + `input_note`. (route: delegated writer)
      Done in 2ce8a6a. RED: 20 tests failed (TypeError, unexpected kw `input`); GREEN: full suite 717 passed.
      Tool-defs budget 8850 -> 8890 (+1 param `input`, looming_escape 592 chars).
- [ ] T4 visual: disc projected on the eye, wave, DNp01 trace.

- [ ] T5 hybrid with a graded optic-lobe model (Lappalainen et al. 2024): verify code/weights availability and
      the column-to-FlyWire mapping, then feed its output into the LIF at mid-level neurons.

## Findings
- Photoreceptor null result (kept as a documented model limit): R1-6 -> DNp01, LPLC2 and LC4 all 0 Hz in all
  four conditions (all p = 1); R1-6 at 300 Hz gives L3 ~7.7 Hz, LPLC2 0, DNp01 0 (verified independently).
  The signal dies in the optic lobe, not at the readout.
- Likely cause (literature): the Shiu LIF treats every neuron as spiking, but lamina/medulla neurons are graded,
  and FlyWire's transmitter predictor has no histamine class, so R1-6 signs are guessed.
  - Shiu et al. 2024 validated only taste and grooming; the model ignores non-spiking neurons and treats all as
    spiking: https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/
  - Eckstein et al. 2024 transmitter predictor has no histamine class: https://pmc.ncbi.nlm.nih.gov/articles/PMC11106717/
  - Lamina/medulla neurons are graded (Borst, Haag & Mauss 2020): https://link.springer.com/10.1007/s00359-019-01375-9
  - Lappalainen et al. 2024 graded optic-lobe model works for motion: https://www.biorxiv.org/content/10.1101/2023.03.11.532232v1.full
  - Eon Systems pipes a Lappalainen-style visual model into the Shiu LIF at mid-level neurons:
    https://eon.systems/updates/embodied-brain-emulation
- Readout works: driving LPLC2 directly gives DNp01 139 Hz; LC4 gives 85 Hz.
- Decision: keep the null result as a documented limit, and add `input="detectors"` (inject the disc at
  LPLC2+LC4, "starts halfway", like Eon's hybrid).
- Detector run (right side), 5 trials, 500 ms, 100 Hz, 10 steps, 44 s:
  DNp01 looming 24.8 Hz [23.2, 26.4], receding 24.8 [23.2, 26.4], dimming 32.8 [28.3, 37.3], shuffled control
  22.6 [20.9, 24.3]. Looming vs receding: d 0, p 1 (identical mean). vs dimming: -8.0 Hz, p_adj 0.017, d -2.9
  (dimming drives MORE). vs shuffled control: +2.2 Hz, p 0.030, p_adj 0.060, d 1.7 (not significant after Holm).
  Looming DNp01 trace rises only in the last 3 of 10 windows (48, 76, 124 Hz); receding peaks early.
  Reading: the model integrates total input, so with equal drive DNp01 mean rate is timing-blind (looming =
  receding) and more spread-out drive (dimming) wins; no looming selectivity. Mean-rate readout cannot show
  timing selectivity: a peak/last-window readout or a latency metric would be needed.

## Checks
TDD strict. Runners: `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`.

## Progress
- Created 2026-11-11 after phase 0.
