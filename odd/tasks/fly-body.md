# Fly body: zoom out of the brain and see the fly

## Objective
In the web viewer the fly body (3D, white lines on black, the app's style) is the center stage and the live brain
is a smaller panel top-right (swappable). The first idea was a zoom in/out with the brain inside the head at real scale;
the user proposed the two-panel layout instead (simpler, no lazy loading). The user asked for this on
2026-11-16 ("el siguiente paso es darle cuerpo": a zoom in/out where zoomed out you see the body). The user
rejected graphical widgets earlier (Terrarium panel); this one is the user's own idea, keep it minimal and let the
fly itself be the interface.

## Honesty rules
- The body is a static NeuroMechFly v2 micro-CT mesh of an adult female (Apache-2.0, credit it). It is NOT animated
  by the brain yet; do not imply the body reacts to the simulated neurons until a later task does it.
- Scale is approximate and says so; the brain is placed in the head from real units (atlas micrometers, mesh
  millimeters x 1000).
- Never claim the body is connectome-driven. Eon Systems' demo (March 2026) chose its neuron-to-body mappings by hand;
  any future coupling here is our own choice and is labelled so.

## Phase 0 evidence (2026-11-16, two read-only mappers)
- Viewer boot loads /brain/meta, /brain/points (139,248 x 12 B = 1.67 MB), regions (139 KB), nt (139 KB), landmarks.
  JS + GPU for the whole brain is only about 18-20 MB, so unloading the points saves little memory; the real gain
  of lazy loading would be boot time. No lazy loading or partial dispose exists: createScene takes the Brain
  eagerly and one position buffer is shared by base, highlight and activity layers, paths, labels, pick, project.
- Camera: FIT 1.55, PerspectiveCamera(45, 1, maxDim/100, maxDim*20), no min/max distance, the render callback
  already computes `dist = camera.position.distanceTo(controls.target)` each frame (the LOD hook). Positions are raw
  FlyWire micrometers; `flip` group (scale.y = -1) and `world` offset by -brain.center; net p = (x-cx, -(y-cy), z-cz).
- Web tests are pure-function style (no WebGL mocking, no scene.ts tests): put decisions in a pure module.
- Body asset: flygym 2.1.1 wheel (PyPI, 2.7 MB), Apache-2.0 (no separate asset license found, an assumption that it
  covers the meshes), `flygym/assets/model/neuromechfly/meshes/simplified_max2000faces/*.stl`: 46 parts, 71,852
  triangles, c_head/c_thorax/c_abdomen*/c_haustellum/c_rostrum, l_eye/l_arista/l_wing/l_haltere, legs lf/lm/lh x
  coxa..tarsus5. Only the LEFT side is shipped (mirror for the right). Units metres (thorax ~1.05 mm long).
  OPEN: whether the STLs share one body frame or need the MJCF/rigging transforms (not verified).
- Real size: body ~2.5-3 mm (secondary source), brain ~0.6 mm wide (JRC2018 template box 627 x 293 x 173 um).

## Tasks
- [x] T1 asset pipeline (route: delegated writer). `src/flybrain/fly_body_build.py` + `scripts/build_fly_body.py`
      (numpy only, deterministic, refuses a wheel whose sha256 differs), tests/test_fly_body_build.py (26 tests),
      assets `web/public/fly-body.bin` (387,144 B = 32,262 segments x 12 B, uint16 per axis), `fly-body.json`,
      `LICENSE-neuromechfly.txt` (Apache-2.0), README build paragraph and credit line.
      FRAME QUESTION RESOLVED: the STLs do NOT share a frame; each is body-local. Placed with rigging.yaml (pos in mm,
      quat) over the manual neutral pose; joint convention cross-checked on six generated pose files (feet agree
      within 50-80 um; not independently proven). Mirror axis y. Final fly: 69 parts (fused trochanter+femur),
      length ~2,611 um (real ~2.5-3 mm), width ~980, head 753 um (829 with eyes), head center [762.9, 0, 1363.2]
      um in NeuroMechFly world frame (x forward, y left, z up; origin NOT the fly's centre). Midline artifact
      (internal mid-wall, ~5,000 false edges) removed; feature-edge angle 40 deg (30 deg gave 516 KB). Standing
      pose not levelled (about 5 deg tilt). RED: ModuleNotFoundError; GREEN: 714 passed (parent re-ran). Parent
      rendered three line views from the committed .bin and judged it reads as a fly (side, top, front); the
      trace is busy with fine lines. Not verified: how it looks in Three.js; license scope of the meshes is an
      assumption (Apache-2.0 of the wheel, no separate asset license found).
- [ ] T2b brain PREVIEW (user idea 2026-11-16): a small 2D picture of the brain drawn with the same oblique angle as the
      3D home view and depth-shaded so it reads as 3D. Backend `GET /brain/preview` returns a small uint8 grid (about
      20 KB, pure numpy, no new dependency, cacheable); the browser draws it on a canvas in the theme colors. It
      needs NO point positions, so /brain/points is no longer fetched at boot. It is a static picture (no activity).
- [ ] T2a brain ON DEMAND: all-or-nothing. boot() stops awaiting the brain; the real Three.js scene (existing
      createScene, unchanged inside) is created when the user zooms (wheel) or clicks the preview, or when a chat
      highlight / path / activity event arrives (auto-open, the latest pending event is replayed once the scene
      exists), with a loading state. charts/player/inspector are wired when the scene appears. Once opened it stays
      loaded (no unload yet). Open triggers are pure functions with tests.
- [ ] T3 `web/src/body.ts`: its own small scene+renderer for the fly body (LineSegments from fly-body.bin, theme
      colors, auto-rotate, orbit, dispose); pure helpers (decode, camera framing) tested.
- [ ] T2c layout: the body takes the CENTER stage; the brain (preview first, the real scene once opened) is an INSET panel
      top-right with a hide/show toggle (remembered in localStorage, try/catch; hidden = not rendered, state kept)
      and a swap control that exchanges the inset and the center sizes (full-size brain one click away).
- [ ] T4 (dropped) partial lazy loading of the point layers and distance crossfade: all-or-nothing on-demand replaces it.
- [ ] T5 (later, separate decision) animate the body from the brain readouts, labelled as our own mapping.
Order: T1 (asset, running) -> T2b -> T2a -> T3 -> T2c.

## Checks
TDD strict (source: project config). Runners: `uv run pytest -q` (baseline 688 on main), `pnpm -C web test`
(baseline 409), `pnpm -C web build`. Visual checking needs a browser: the parent cannot see it, so the user tries it.
Delivery: ask-on-risk; branch feat/fly-body from main; no push or PR until asked.
Engram mirror `odd/fly-body/tasks`: PENDING (mem_save is blocked by a hook that rewrites its input).

## Progress
- 2026-11-16 created after phase 0. T1 done. Next step: T2b.
