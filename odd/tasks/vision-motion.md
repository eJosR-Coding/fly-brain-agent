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
