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
      CORRECTION (B3): DNa01/DNa02 are one neuron per SIDE, so turn = left minus right within each type (dna02_left/right,
      dna01_left/right; DNa02 gain 1.0, DNa01 0.4 = our weights); ipsilateral direction per Rayshubskiy et al. eLife 102230.
- [x] B2 body runner: flygym as an OPTIONAL extra (`body`), runs the controller with a time-varying drive, returns a
      fly-walk/1 clip incl. thorax path; refused clearly when not installed; tests (pure parts; runner gated).
- [x] B3 API + presets: direct-activation presets (dng100 walk, dna02 left/right turn) with recordings that carry the DN
      readouts; POST route recording -> body clip; agent access within the tool budget.
- [x] B4 viewer: play the body clip with the brain recording, caption with the mapping disclaimer; chips.
      Chips 'walk (DNg100)', 'turn left (DNa02 L)', 'turn right (DNa02 R)' run the preset (dng100 50 Hz, dna02 200 Hz, 1 s, 3 trials),
      then POST /body/from-simulation/{id} ('computing body physics ~N s'), then recording and body start together (clip driven by
      the player clock). Other runs: a 'move body' chip (never automatic: ~8 s compute per simulated second); an all-zero drive
      shows 'no movement command from the brain (descending neurons below threshold)' and plays nothing; 503 shows the install hint.
      Body scene: root transform from thorax_world (whole fly moves and yaws), faint world-fixed floor grid, smooth camera follow
      (no auto-orbit while following), escape frames raise the wings. Visual result in a browser NOT yet verified.

## Progress
B1/B2 done (uncommitted): src/flybrain/body_drive.py, src/flybrain/body_runner.py (shared exporter logic; the script regenerates the
walking asset byte-identically), optional extra `body` (flygym==2.1.1, uv.lock updated), tests/test_body_drive.py (14) and
tests/test_body_runner.py (7 pure + 5 physics, gated). Physics ~8 s wall per simulated second. 
B3 done (uncommitted): presets dng100 / dna02_left / dna02_right (role stimulus, exploratory, 'direct activation'),
src/flybrain/body_readouts.py (DN traces computed on demand from any recording's frames), body_service.py, route
POST /body/from-simulation/{id} (<= 3000 ms, 503 without the extra), tests; `fly` description names the new presets within
the tool-def budget. Real end-to-end: dng100 200 Hz -> walks ~11 mm/s; dna02_left/right turn the expected way; sugar: no walk. B4 done (uncommitted): web/src/bodyWorld.ts, bodyClipPlayer.ts, bodyFlow.ts (+ bodyClip/body/bodyMount/main wiring), tests.

## Checks
TDD strict. `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`. Base: feat/walk-06-player (PRs #62-#67, unmerged).
