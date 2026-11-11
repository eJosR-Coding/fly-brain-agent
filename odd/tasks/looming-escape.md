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
- [ ] T4 visual: disc projected on the eye, wave, DNp01 trace.

## Checks
TDD strict. Runners: `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`.

## Progress
- Created 2026-11-11 after phase 0.
