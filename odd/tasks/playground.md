# Playground: a glass-cube world the user drives through the chat, with the full brain live

## Objective
No buttons: the user writes "dale azúcar", "dale café", "muéstrale este video", "sube la temperatura" and the agent
changes a live world. The fly lives inside a transparent glass cube with a floor; objects (sugar or coffee cubes from a
dispenser) appear inside; when the fly touches one, the matching taste neurons are activated and the full brain responds
live. Requested 2026-11-18.

## Real time
- Brain: full FlyWire LIF (139k neurons) on the GPU, advanced in 20-50 ms windows (measured ~2x real time, 1 trial).
- Body: KINEMATIC in live mode (the physics-recorded flygym gait, phase advanced and blended by the brain's forward/turn
  drive; escape = stop + wings). Labelled "recorded physics gait, blended live". Full physics stays as an optional
  high-fidelity mode (slower, labelled).
- Always show the real-time factor; never fake real time.

## Honesty
- Sensory paths: sugar -> sugar GRNs (validated MN9), coffee -> caffeine-sensitive bitter GRNs (to verify with sources),
  video -> T4/T5 motion front end (exact columns; right eye reaches DNp01), temperature -> thermosensory neurons (to measure).
- World rules are labelled as such (glass walls stop the fly; objects fall). The brain is never replaced by rules: no
  scripted "walk to food". If the brain does not react, show it.

## Tasks
- [x] P1 kinematic body for the live session (gait phase + blending + root motion inside the cube; tests).
      Progress P1 (branch wip/pg-p1, backend): `scripts/build_motion_library.py` (body extra) records flygym physics over pace {0, .33, .67, 1} x
      turn {-1, -.5, 0, .5, 1} (drive = body_drive.walking_drive) -> `web/public/fly-motion-library.json` (462 KB, 20 clips, 25 phase samples x 48
      bodies int16/base64, measured speed / lateral / yaw rate per clip, standing pose, escape = stand + wings flag; phase 0 = LF swing onset;
      seams checked). `src/flybrain/body_kinematic.py` `KinematicBody` (pure numpy: bilinear clip blend at equal phase, 100 ms stand ramp,
      root motion from the blended measured speed / yaw rate, optional labelled `bounds` world rule). LiveSession `body_mode`
      kinematic (default) | physics in `started` + every chunk (+ `body_caption`, `world`); /ws/live start option `body_mode`.
      Measured live (GPU brain + kinematic body, 50 ms chunks): brain ~22 ms + body ~0.7 ms = ~23 ms per chunk (x2.15 real time after the
      GPU warm-up). Kinematic vs physics (3 s, same drive): speed within 1%, yaw rate within 2% on turning clips. Not done: spin-in-place clips
      (the opposite-direction CPG is not periodic: recorded, rejected), viewer (P2).
- [ ] P2 glass cube sandbox in the viewer (transparent walls, floor, fly inside, objects rendered; camera).
      Progress P2 (branch wip/pg-p2, uncommitted; includes the live viewer client G3b): web/src/bodyCube.ts (pure geometry; 50 mm
      interior, 10x10 floor cells, 24 mm follow span), bodySandbox.ts (faint depth-tested glass faces + theme-line edges + floor
      grid + objects layer), bodyObjects.ts (`setObjects([{id, kind: sugar|coffee, pos_mm, size_mm}])` validation), body.ts /
      bodyMount.ts `setSandbox(on)` / `setObjects(list)`; live client liveFrames / livePlayback / liveSession / liveClient (jitter
      buffer, own time base, close codes, no auto-reconnect), vite `/api` proxy `ws: true`; dev entry = "live (dev)" chip +
      `window.flyLive` until P4. Web 940 tests, build and tsc clean. Not yet seen in a real browser/GPU or against a real socket.
      Walls fix (branch wip/pg-walls, uncommitted): the SERVER owns the arena (`live.arena_spec`, 50 mm glass cube centred on the
      start; bounds_mm [xmin,xmax,ymin,ymax], fly frame x forward / y left) and sends it in `started.arena`; kinematic mode enforces it
      in KinematicBody (world.blocked + world.walls), physics mode clamps the REPORTED root (labelled display clamp). The viewer draws
      the cube from `arena` (fallback constant; x -> viewer Z, y -> viewer X) and shows "stopped by the glass (world rule)". A start
      right after a client stop waits (<= 3 s) only for a TEARING-DOWN session; a running one still gets 4409 at once.
- [x] P3 world state + contact: objects, contact detection (proboscis/tarsi near an object) -> taste stimulus in the live brain.
- [x] P4 agent tools: world actions (add sugar/coffee cubes, clear) within the tool budget; the chat drives the live world.
- [ ] P5 caffeine neurons: identify caffeine-sensitive gustatory neurons in FlyWire with sources.
- [ ] P6 later: video input (link/mp4 -> motion -> T4/T5) and temperature (thermosensory neurons, measured first).

- [ ] CL1 closed-loop vision: a general fly eye (per-column view directions from the exact column map; luminance of the
      arena scene per column per frame; Hassenstein-Reichardt motion per subtype direction -> T4/T5 rates) feeding the live
      brain every chunk; optional labelled background noise; report what the fly does by itself (DN activity, movement).
      Progress CL1 (branch wip/pg-cl1, uncommitted; DEPENDS on feat/vision-motion: its eye.py head_frame/from_xy, retina.py, motion.py,
      looming.py and their tests are carried as a working-tree copy and must come from that branch's merge): `src/flybrain/fly_eye.py`
      (pure numpy): per-column unit view directions (exact column map, eye axis 65 deg lateral = our choice), `ArenaScene` ray casting
      (floor soft checker 4 mm period, light background through the glass, 12 cube edges as dark 1.5 mm lines, optional dark boxes, analytic
      receptor blur), `Hrc` (Weber contrast, ON=T4 / OFF=T5 half-wave split, 40 ms low-pass delay on the two upstream flank columns,
      opponent, rectified; static or uniform change = exactly 0), drive -> rate = 300 Hz * (1 - exp(-hr/0.02)) (scale = the validated
      looming 300 Hz; gain our choice), `VisionFrontEnd` (pose of the last chunk -> rows + rates; 1 chunk latency; ~4.5 ms per chunk, both
      eyes, ~800 columns each). `BrainStream.set_rates(rows, rates_hz)` (CPU + GPU; run() goldens untouched). LiveSession `vision`
      (+ build_live_session / ws start options `vision: "arena"|null`, `background_hz`): per-chunk combined rates, `senses` block + caption in
      every message, `wall_ms.vision`. Default OFF. Real GPU experiment (kinematic body, 50 ms chunks): at rest nothing; the walking fly's own
      optic flow drives thousands of T4/T5 cells at up to 300 Hz and the loop sustains walking/turning/DNp01 that the vision-off control lacks;
      DNp01 first fires ~20 mm from the wall (floor flow), not from the edges (uniform floor: no DNp01). Numbers in the CL1 report.
      CL1 calibration (branch wip/pg-cal): the rate scale k = `fly_eye.MAX_RATE_HZ` is the ONLY free parameter and is now 100 Hz
      (`MAX_RATE_CALIBRATED = True`; was an uncalibrated 300 Hz, at which floor flow fires DNp01). Detector delay tau 53 ms
      (`HRC_OPTIMAL_TF_HZ` = 3 Hz, our choice inside a published bracket; the previous 40 ms peaked at 4 Hz). Criteria fixed before the sweep:
      LOOM = a dark disc approaching the right eye (r/v 20 and 40 ms, 5 to 75 deg half-angle over 0.8 s, 12 brain seeds) fires the ipsilateral
      DNp01 (>= 1 spike in the 1 s window) in >= 80% of trials; NO-GO = receding disc, uniform dimming, and straight walking at 14 mm/s over the
      checkered floor (x -20 to +20 mm, 2.9 s, scripted pose) fire neither DNp01 in any trial (0/12 = at most ~22% at 95%). Result:
      k = 25 fails loom (0/12); k = 50 loom 0.83; k = 75-200 loom 1.00 and all NO-GO 0/12 (the calibrated range, centre ~120 Hz); k = 300 floor 0.75.
      The full table (k x stimulus -> DNp01 R/L spikes, LPLC2, LC4, driven cells) is in the CL1 calibration report and in the docstring/comment at
      `fly_eye.MAX_RATE_HZ`. Literature: T4/T5 are graded (no published firing rate in Hz), LPLC2 does not respond to wide-field translation
      (Klapoetke 2017); whether walking-speed floor flow drives the real GF was NOT FOUND.
- [ ] CL2 touch: antennal mechanosensory input when the head touches the glass (leg mechanosensors live in the VNC, not
      in this brain-only model: stated).
- [ ] Flight roadmap: F0 research flybody flight (weights, license, geometry); F1 takeoff from DNp01 (validated loom path);
      F2 kinematic flight library; F3 steering/landing only with evidence or labelled as our choice.

## Checks
TDD strict. `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`. Depends on G3a (live session backend).
