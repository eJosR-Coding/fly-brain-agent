# Brain-body loop: descending-neuron activity drives a physics body (Eon-style, labelled)

## Objective
The fly reacts with real physics to what its simulated brain does: descending-neuron (DN) rates from a brain recording are
translated (OUR mapping, labelled on screen) into the NeuroMechFly v2 / flygym 2.1.1 walking controller's descending drive,
the physics runs on demand on the server, and the viewer replays the resulting body motion with the brain recording.
Requested 2026-11-18 ("si pues eso deberia funcionar para que reaccione"), after comparing with Eon Systems (same brain
model + NeuroMechFly; mapping and code not published).

## Honesty
- The brain model and the body are published; the DN -> drive mapping is ours (scales, thresholds, smoothing) and says so.
- Triggers that work today: looming (right eye, validated in-model loom selectivity of DNp01), sugar -> MN9, and direct
  DN activation (like optogenetics: DNg100 walk, DNa02 turn). No sensory preset drives walking DNs: not faked.
- flygym walks with a preprogrammed tripod CPG (fixed 12 Hz); no takeoff model: escape = stop + wing raise (illustrative).

## Tasks
- [x] B1 pure DN -> drive mapping: forward from DNg100 (with a threshold), turn from DNa01/DNa02 left-right difference,
      escape (stop + wings) from DNp01; per-bin from a recording's named readouts; tests.
- [x] B2 body runner: flygym as an OPTIONAL extra (`body`), runs the controller with a time-varying drive, returns a
      fly-walk/1 clip incl. thorax path; refused clearly when not installed; tests (pure parts; runner gated).
- [ ] B3 API + presets: direct-activation presets (dng100 walk, dna02 left/right turn) with recordings that carry the DN
      readouts; POST route recording -> body clip; agent access within the tool budget.
- [ ] B4 viewer: play the body clip with the brain recording, caption with the mapping disclaimer; chips.

## Progress
B1/B2 done (uncommitted): src/flybrain/body_drive.py, src/flybrain/body_runner.py (shared exporter logic; the script regenerates the
walking asset byte-identically), optional extra `body` (flygym==2.1.1, uv.lock updated), tests/test_body_drive.py (14) and
tests/test_body_runner.py (7 pure + 5 physics, gated). Physics ~8 s wall per simulated second. Next: B3.

## Checks
TDD strict. `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`. Base: feat/walk-06-player (PRs #62-#67, unmerged).
