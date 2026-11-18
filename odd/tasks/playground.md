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
- [ ] P3 world state + contact: objects, contact detection (proboscis/tarsi near an object) -> taste stimulus in the live brain.
- [ ] P4 agent tools: world actions (add sugar/coffee cubes, clear) within the tool budget; the chat drives the live world.
- [ ] P5 caffeine neurons: identify caffeine-sensitive gustatory neurons in FlyWire with sources.
- [ ] P6 later: video input (link/mp4 -> motion -> T4/T5) and temperature (thermosensory neurons, measured first).

## Checks
TDD strict. `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`. Depends on G3a (live session backend).
