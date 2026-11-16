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
- [ ] T1 asset pipeline: `scripts/build_fly_body.py` (numpy only, reproducible) turns the NMF STL parts into a compact
      line-segment file `web/public/fly-body.bin` + `fly-body.json` (parts/groups, units um, head center, bounds,
      source and attribution) plus the Apache-2.0 LICENSE/NOTICE text. Resolve the frame question first. Feature
      edges only, quantized, target <= ~400 KB. Tests for parse, edges, quantize, mirror, and the committed asset.
- [ ] T2 layout (user idea 2026-11-16, replaces the zoom/LOD crossfade): the fly body takes the CENTER stage where the
      neurons are today; the existing 3D brain scene moves into an INSET panel top-right of the stage (still the
      same live Three.js scene, orbit/pick/highlight/activity unchanged). A swap control (click the inset) exchanges
      the two sizes, so the full-size brain stays one click away. Verify first how createScene/resize, labels,
      inspector picking, charts and the left controls behave when the brain container is small.
- [ ] T3 `web/src/body.ts`: its own small scene+renderer for the fly body (LineSegments from fly-body.bin, theme
      colors, auto-rotate, orbit, dispose); pure helpers (decode, camera framing, swap/layout decision) tested.
- [ ] T4 (dropped) lazy loading of the point cloud and distance crossfade: not needed with two views.
- [ ] T5 (later, separate decision) animate the body from the brain readouts, labelled as our own mapping.

## Checks
TDD strict (source: project config). Runners: `uv run pytest -q` (baseline 688 on main), `pnpm -C web test`
(baseline 409), `pnpm -C web build`. Visual checking needs a browser: the parent cannot see it, so the user tries it.
Delivery: ask-on-risk; branch feat/fly-body from main; no push or PR until asked.
Engram mirror `odd/fly-body/tasks`: PENDING (mem_save is blocked by a hook that rewrites its input).

## Progress
- 2026-11-16 created after phase 0. Next step: T1.
