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
- [x] T2b brain PREVIEW (user idea 2026-11-16; route: delegated writer): a small 2D picture of the brain drawn with the same oblique angle as the
      3D home view (azimuth 16, elevation 10 in ONE Python place, asserted by a test) and depth-shaded. `GET /brain/preview`
      = 4-byte header + uint8 grid, 256x124 = 31,748 B on the real atlas, 16,734 non-empty cells, first request 12.5 ms,
      then 1.2 ms (cached in app.state), Cache-Control 1 h. src/flybrain/brain_preview.py (pure numpy) +
      web/src/brainPreview.ts (parse, RGBA from theme.highlightHex with alpha = brightness, mountPreview). Wired in boot():
      shown as a placeholder before `await loadBrain`, removed after createScene and in the error path; pointer-events
      none, fade-in unless reduced motion. /brain/points and the real scene are untouched (T2a does that).
      RED: collection errors (modules missing). GREEN: pytest 723, web 418 (27 files), build ok (parent re-ran all three).
      Parent viewed the PNG decoded from the real endpoint: two optic lobes and a textured central brain, depth shading
      reads as relief. Not verified: how it looks in the real page (flip, 10% inset, alignment with the 3D home view).
- [x] T2a brain ON DEMAND (route: delegated writer + parent fixes). boot() no longer awaits the brain: the 2D preview is the
      default stage with a focusable control (role=button, hint "zoom or click to load the neurons"); the real scene
      (existing createScene) is created by `gate.ensure()` on wheel zoom-in, click, Enter/Space, a chat highlight/path/
      activity event, RUN/VIEW/HIGHLIGHT chips, and replays only the latest pending event once ready. New pure
      `web/src/brainGate.ts` (idle|loading|ready|failed, shared load, retry after failure) and `web/src/previewOpen.ts`.
      RED: two new test files failed on missing modules. GREEN: web 436 (29 files), pytest 723, build ok (parent
      re-ran). Untested glue in main.ts (no WebGL in this repo): loadScene wiring, the openBrain call sites, the
      onHighlight/onPath/onActivity integration with pathToken/shown replay, pointCount re-validation.
      PARENT BROWSER CHECK (headless Firefox, throwaway profile, a wrapper page delaying the load event 6 s): the
      page shows the preview with the hint and no 3D scene; FOUND AND FIXED: (1) the header said "FlyWire v783 ·
      loading" forever -> `loadBrainMeta()` (small meta JSON) fills "139,248 neurons" at boot, 2 tests RED -> GREEN;
      (2) the preview looked pixelated and grey -> compared 4 variants on the real atlas, kept 384 cells / gamma
      0.6 (70,276 B; 512 grids get speckled), 2 size tests updated on purpose. Verified by screenshot: the preview and
      the header. NOT verified: clicking/zooming opens the real scene, the loading state, retry, replay of pending
      events (headless Firefox cannot click; unit tests cover the gate and the control only).
      Caveat: /brain/preview is cached for 1 h (like /brain/points), so a browser keeps the old image until a hard
      reload; a stale-cache concern after any change to the picture (ETag or a version query would fix it).
      Caveat from the writer: if loadScene throws after createScene (e.g. in createPlayer) a retry would build a
      second scene (judged rare).
- [x] T2d collapsible agent chat (user recommendation 2026-11-16; route: delegated writer). `web/src/panelState.ts` (localStorage
      key flybrain.chat.collapsed, try/catch, default shown) + `web/src/chatToggle.ts` (class chat-collapsed on body,
      `inert` on the chat while collapsed, aria-expanded/aria-controls/labels, focus moves to the visible control, a
      "hide" button in #chat-head and a vertical "AGENT" tab on the right edge while collapsed). CSS: collapsed =
      `--panel-width: 0px` (stage and status re-anchor by themselves), chat visibility hidden + 180 ms translate, never
      display:none or remounted; hidden on screens <= 720 px (the chat is a bottom block there). RED: 2 test files
      failed on missing modules. GREEN: web 447 (31 files), build ok, pytest 723 (parent re-ran). Parent browser check
      (headless Firefox with the preference seeded by a temporary page, removed): chat gone, stage full width, status
      block re-anchored, "AGENT" tab visible. FOUND AND FIXED: in the wide collapsed stage the preview grew over the
      control rows -> inset 12% 8% 26% 8%. NOT verified: the click itself, the transition, ESC/focus behavior.
      Known: the preview still looks pixelated when upscaled ~4x on a wide stage (blur tried and reverted: no visible gain).
- [x] T3 `web/src/bodyData.ts` + `body.ts` + `bodyMount.ts` (route: delegated writer). Pure decode of fly-body.bin/json to mm and to
      Three axes (three = (y_fly, z_fly, x_fly), handedness kept: face on +Z, fly's left on viewer-right +X), own
      WebGL scene (transparent, theme.highlightHex lines at 0.6, OrbitControls, 3/4 front-left default, slow auto-
      rotate via orbit.ts, off under reduced motion, on-demand scheduler, ResizeObserver, full dispose) and a caption
      "static model (NeuroMechFly v2 micro-CT mesh, Apache-2.0), not driven by the brain". Visible ONLY with the URL
      flag `?body=1` (in that mode the brain is skipped; chat events that need the brain would still call
      gate.ensure() and draw the brain over the body: documented in main.ts). RED: bodyData tests failed on the
      missing import; GREEN: web 461 (32 files), build ok, pytest 723 (parent re-ran).
      PARENT RENDER CHECK: headless Firefox does NOT capture WebGL (a plain red-cleared WebGL canvas also screenshots
      empty; rAF works, also in iframes), so the first screenshot (only the caption) proved nothing. Workaround: a
      temporary debug page (removed) built the real createBodyScene with preserveDrawingBuffer patched in temporarily
      (reverted), copied the canvas to an <img> with toDataURL: the fly renders correctly (white lines on black,
      3/4 view, head left, wing over the back, six legs). Data diagnostics: 32,262 segments, 193,572 floats, 0 NaN,
      bounds x -1.53..1.53, y -0.09..1.72, z -2.32..1.34 mm, theme color #ffffff, canvas 1898x968.
      POLISH (to do in T2c): the fly fills only ~43% of the height and sits right of center (framing uses the bbox
      bounding sphere); the writer left a redundant assertion and an unused variable in bodyData.test.ts.
      Also: this method cannot verify the REAL brain scene's WebGL either.
- [x] T2c layout (route: delegated writer). The body is the DEFAULT center stage (the `?body=1` flag and its early return are
      gone); the brain (preview first, the real scene once opened) is an inset top-right under #status (340 px, 16:10,
      1 px --line border) with "swap" and "hide" buttons; a tab shows it again. New pure `layoutState.ts` (key
      flybrain.layout, try/catch, malformed values fall back to the default) and `layoutPanes.ts` (`#pane-body`,
      `#pane-brain`, `#pane-show`; swap and hide only change classes/aria/inert, no DOM is moved and no scene is
      recreated; hidden = visibility:hidden, size unchanged). The real brain scene now mounts in `layout.brainHost`
      (createScene already used its container for canvas, labels, ResizeObserver and pick; the fix was in main.ts);
      landmark labels are hidden by CSS while the brain is the inset. Swapping to the brain opens it (gate.ensure()).
      Narrow screens: inset 150 px at top 96 px, caption hidden. Framing: `centroid`, `boundingRadius` and an exact
      `viewFitDistance` (replaces frameDistance; a test over the committed asset gets max |ndc| 0.70 at the default
      3/4 view) with the orbit target at the vertex centroid. RED: modules missing / viewFitDistance not a function.
      GREEN: web 497 (34 files), build ok, pytest 723 (parent re-ran). Parent browser check (headless Firefox, clean
      profile, 1898x968): the inset with BRAIN header, SWAP and HIDE, the preview and the hint at top-right; the center
      pane shows only its caption because the body is WebGL (not captured). NOT verified: any WebGL look (body in the
      center, real brain in the inset), picking in the small pane, swapping with the real scene open, the 128 px offset
      against every health state, the narrow layout on a phone, the clicks themselves.
- [x] T3b hidden-line look (user feedback 2026-11-16: the transparent wireframe read as a tangle of "nerves / connections").
      Route: delegated writer + parent tuning. Build script now also writes `web/public/fly-body-mesh.bin` (welded triangle
      mesh of all parts, same assembly/mirror/quantization as the lines, mid-plane faces dropped; 54,861 vertices, 109,130
      triangles, uint16 indices, 983,946 B) and adds ONLY a `mesh` key to fly-body.json; fly-body.bin is byte-identical
      (sha256 0810bb68...b3c2), the build is deterministic (two runs identical). Viewer: `parseBodyMesh`, `loadBody` fetches
      the mesh (any failure -> one console.warn and the lines-only body as before), body.ts draws a depth-only occluder
      (colorWrite false, depthWrite true, DoubleSide, renderOrder -1) before the lines (depthTest on, depthWrite off).
      RED: ImportError (MeshPart) / 11 web failures; GREEN: pytest 731, web 508 (34 files), build ok (parent re-ran).
      PARENT RENDER CHECK (canvas -> toDataURL image): the tangle is gone and the fly reads as a body (eye, thorax, wing,
      legs). First render (polygon offset 1/1, opacity 0.75) was too faint and broken by z-fighting (WebGL polygon offset
      moves only filled polygons, never lines); tuned to offset 4/4 and opacity 0.9 by eye. Still sparse on the abdomen
      side in this 3/4 view (the folded wings cover it). Not verified: the real page, per-frame cost of 109k triangles on
      a weak GPU, other azimuths while it auto-rotates.
- [ ] T4 (dropped) partial lazy loading of the point layers and distance crossfade: all-or-nothing on-demand replaces it.
- [ ] T5 (later, separate decision) animate the body from the brain readouts, labelled as our own mapping.
Order: T1 (asset, running) -> T2b -> T2a -> T3 -> T2c.

## Checks
TDD strict (source: project config). Runners: `uv run pytest -q` (baseline 688 on main), `pnpm -C web test`
(baseline 409), `pnpm -C web build`. Visual checking needs a browser: the parent cannot see it, so the user tries it.
Delivery: ask-on-risk; branch feat/fly-body from main; no push or PR until asked.
Engram mirror `odd/fly-body/tasks`: PENDING (mem_save is blocked by a hook that rewrites its input).

## Reviews
- T1 slice (base main -> d5a67ad, 1524 lines, risk medium: scripts/build_fly_body.py): consent granted, consolidated
  native review APPROVED, acknowledged (lineage review-68336c01a9aaca1e, authority burned). 4 non-blocking findings
  (dev-time build script, the committed asset is validated by tests): R3-001 fly_body_build.py:357-358 .bin and .json
  written separately (not atomic, an interruption leaves an inconsistent pair); R3-002 :195-196 encode_segments edge
  case; R3-003 tests/test_fly_body_build.py:276-337 (suggestion); R3-004 :110-112 parse_stl strictness (suggestion).
  Open follow-ups. Reviewed boundary is now d5a67ad.
- T2b+T2a slice (base d5a67ad -> 6dc522b, 909 lines, risk medium): consent granted, consolidated native review
  APPROVED, acknowledged (lineage review-3a747dd3521e020c, authority burned). 1 non-blocking WARNING, web/src/main.ts:385-390:
  `brainScene` is assigned right after createScene and the preview/control are removed before the player, inspector
  and chips are wired; if that wiring throws, a retry would build a SECOND scene and the page would have lost its
  preview. Fix idea: wire first, remove the preview last, and in the catch dispose the scene and reset brainScene.
  Open follow-up (untestable glue without WebGL). Reviewed boundary is now 6dc522b.
- T2d+T3 slice (base 6dc522b -> 2e306a5, 691 lines, risk medium: web/src/body.ts): consent granted, consolidated native
  review APPROVED, acknowledged (lineage review-2ee83b8ffb851600, authority burned). 4 non-blocking findings:
  R3-001 bodyMount.ts:28-33 if createBodyScene throws after building the canvas, stage.remove() leaves the WebGL
  context undisposed (small leak); R3-002/R3-003 chatToggle.ts:44-48 and :39-41 focus handling around inert/hidden
  controls (focus moves on every state change); R3-004 bodyData.ts:50-55 (suggestion). Open follow-ups.
  Reviewed boundary is now 2e306a5.
- T2c slice (base 2e306a5 -> be52349, 696 lines, risk medium): consent granted, consolidated native review APPROVED,
  acknowledged (lineage review-86b70c3f59e0b78f, authority burned). 2 non-blocking findings: R3-001 web/src/main.ts:437-440
  (WARNING; see the lines: the layout/body/preview mounting order in boot()); R3-002 bodyData.ts:131-148 (suggestion,
  viewFitDistance numerics). Open follow-ups. Reviewed boundary is now be52349.

## Progress
- 2026-11-16 created after phase 0. T1, T2b, T2a, T2d, T3, T2c done. Next: the user tries it in a real browser; T5 (animate the body from the brain readouts) is a separate decision.
