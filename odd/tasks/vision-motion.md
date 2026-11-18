# Vision step 1: motion-detector input so the connectome computes loom selectivity

## Objective
Give the brain model a visual input one layer earlier than today: drive the direction-selective motion detectors T4a-d / T5a-d
with rates computed from the stimulus (local motion at each neuron's retinotopic position, each subtype with its published
preferred direction), and let the FlyWire connectome compute LPLC2 / LC4 and the escape DNs. Test whether DNp01 becomes
selective for looming over receding, dimming and a control. Requested 2026-11-17 ("arranca con vision").

## Evidence (2026-11-17)
- Current input drives LPLC2+LC4 directly: DNp01 ~25 Hz for looming AND receding/control, dimming higher (not selective).
- Connectome v783 (data/processed/connectome_783.npz): T4a-d/T5a-d -> LPLC2 ~52k synapses onto ~205 of 210 LPLC2;
  LPLC2 -> DNp01 1,080 syn; LC4 -> DNp01 805; T5a -> LC4 914; HSE/HSS -> DNa02 67.
- LPLC2 loom selectivity comes from radial-motion opponency on T4/T5 input (Klapoetke et al. 2017,
  https://pubmed.ncbi.nlm.nih.gov/29120418/; Zhou et al. 2022 eLife 11:e72067).
- Annotation counts: T4a-d and T5a-d ~1,460-1,710 each, LPLC2 210, LC4 104, HS 6, VS 26.

## Honesty
The motion front end (how a stimulus becomes T4/T5 rates, retinotopic placement) is OUR model choice and is labelled so;
the selectivity result is whatever the connectome gives. Spiking LIF vs graded T4/T5 is a known limitation.

## Tasks
- [x] V1 motion front end + `input="motion"` in the looming experiment; tests; run the 4 conditions; report.

- [x] V2 exact retinotopy in code: T4/T5 visual directions from data/raw/flywire_v783_column_assignment.csv.gz (Codex
  'Visual Neuron Columns', Matsliah et al. 2024; p,q = Zhao et al. 2025 eyemap lattice; hex y = dorsal, x = front, both eyes);
  fallback to the PCA map with a note when the file is missing; tests with a tiny fixture.
- [x] V3 escape reaches the body: looming recordings carry the DNp01 readout under the name `dnp01` so the existing body
  mapping (dnp01 -> wings, exploratory) plays; a chip runs the validated configuration (motion, right eye, 300 Hz).

## Validation results (scratch, 2026-11-18)
- V3 exact map (3 seeds, 300 Hz, right stimulus, DNp01 Hz loom/recede/dim/control): polarity both 9.9 / 0.7 / 0 / 3.5;
  dark 4.9 / 0.1 / 0 / 0.9; reversed preferred directions 0.0 / 11.8; permuted positions 3.4 / 5.4; Zhao per-column
  PDs same as exact. Left stimulus: DNp01 0 everywhere (LPLC2 still loom-ordered 1.0 / 0.4): left LPLC2 receive 182
  T4/T5 synapses per cell vs 315 on the right; direct detector drive fires both DNp01 (readout works).

## V2/V3 progress (2026-11-18)
- V2: src/flybrain/retina.py. A plain lattice (hex y up, x front, scaled) is ~19-20 deg median off the Zhao-based exact map
  (p90 36-39), so instead a cubic fit (20 coefficients, no per-column data) of the lens map in (p/20, q/20): 1.0 deg median,
  2.0 p90 vs the exact map. Missing column file -> PCA map with a note. Rerun (code path, seed 0, 5 trials, 500 ms, both,
  300 Hz, right): DNp01 9.4 / 0.8 / 0 / 3.8 (scratch exact, 3 seeds: 9.9 / 0.7 / 0 / 3.5).
- V3: looming recordings carry readouts=[{name: "dnp01"}]; POST /experiments/looming (defaults = validated config);
  chip "looming disc (escape)" plays the looming recording (web/src/looming.ts). Agent tool unchanged (polarity not exposed).

## Checks
TDD strict. `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`.

## V1 results (2026-11-17, 5 trials, real model)
Front end: src/flybrain/motion.py (+ EyeMap.head_frame). Rate = base x cos(preferred vs radial edge motion) in the moving band.
DNp01 Hz (looming / receding / dimming / control):
- motion, 100 Hz, dark polarity: 0.2 / 0.0 / 0.0 / 0.0 (nothing propagates)
- motion, 100 Hz, both:           1.4 / 1.4 / 0.0 / 0.0
- motion, 300 Hz, dark:           1.4 / 0.8 / 0.0 / 0.0 (looming > dimming, control; receding n.s.)
- motion, 300 Hz, both:           4.0 / 1.2 / 0.0 / 2.0 (looming > all, p_adj<0.003); LPLC2 1.5/0.8/0/0.5, DNp04 1.6/0/0/0.2, LC4 0
- detectors (100 Hz):             24.8 / 24.8 / 32.8 / 22.6 (not selective, as before)
Reading: weak but looming-ordered responses; the drive is tiny (DNp01 <= 4 Hz vs ~25 Hz with direct injection), so this is
a front-end-amplitude limited result, not a validated selectivity claim. See the report for artifacts.
