# Fly animation: the body reacts to the simulated brain

## Objective
When the agent (`fly`, `stimulate`) or a chip runs a simulation, the 3D fly body moves in sync with the playback, driven by
the REAL readout time series of the simulated neurons through an explicit, fixed mapping that is OUR choice and says so on
screen. The user asked for this on 2026-11-16 ("si los ocho pasos hagamos").

## Honesty rules (hard)
- It is a model of neurons; the body is a static micro-CT mesh posed by us. Never say the body is connectome-driven or that the
  fly "feels", dies or suffers. Intensity comes from the real readout; the movement itself is an illustration.
- Only "sugar -> MN9 -> proboscis extension" has paper backing (Shiu et al. 2024). aBN1 -> antennal grooming is a validated
  READOUT of the model but the antenna motion is illustrative; DNp01 -> wing raise is exploratory. The caption and the
  mapping table carry these labels.
- Scales are FIXED constants (never normalized per run: a low dose must look weaker than a high dose). Chosen from measured
  readouts and recorded: MN9 63.3 Hz at sugar 100 Hz (4.0 Hz with bitter), aBN1 25.3 Hz at jo_ce 100 Hz.
- Honor prefers-reduced-motion (static final pose). No WebGL screenshot is possible in the parent's headless Firefox: render
  checks use the canvas -> toDataURL image trick.

## CONTRACTS between the parallel pieces (do not deviate; ask the parent if a change is needed)
### Joint ids (step 3 produces them, step 5 consumes them, steps 4 and 6 apply them)
`proboscis` (rostrum + haustellum, extension), `antenna_l`, `antenna_r` (pedicel + funiculus + arista, sweep), `wing_l`, `wing_r`
(raise), `haltere_l`, `haltere_r` (not driven yet), and `leg_<lf|lm|lh>_<l|r>` (not driven yet; may be absent). Each joint in
`web/public/fly-body.json` under a new key `rig`:
`{"units":"um","frame":"fly frame as in the asset: x anterior, y left, z dorsal","joints":[{"id","parts":[part names],"pivot_um":[x,y,z],"axis":[x,y,z] (unit vector),"rest_deg":0,"range_deg":[min,max],"source":"rigging.yaml"|"geometry (our choice)"}]}`.
Joint motion = rotation by `angle_deg` about `axis` through `pivot_um`, applied to every vertex of the listed parts.
Where the rigging files have no joint for a part (likely proboscis, antennae, wings) the pivot and axis are chosen from geometry
and `source` says "geometry (our choice)".
### Mesh and line ranges per part (step 3 produces, step 4 consumes)
`fly-body.json` `mesh` gains per-part `triangle_start`/`triangle_count` (triangles are already concatenated in part order) and
the existing `parts[].segment_start/segment_count` stay; the mesh `index_dtype` rules are unchanged. The committed line asset
`fly-body.bin` stays byte-identical; the mesh asset may be re-ordered ONLY if the new order is deterministic and documented.
### Recording meta (step 1 produces, step 5/6 consume)
`GET /simulations/{id}` keeps its existing `readout` field (the first readout, unchanged) and ADDS `readouts`: a list of
`{ "name": <preset name or null>, "label", "neurons", "rows", "hz": [per-bin mean Hz] }` for EVERY watched readout of the run
(`fly`: all watched presets in the doll's order; `stimulate`: the single readout, name null). Web `Activity` gets an optional
`readouts` with the same shape.
### Time (step 2 produces, step 6 consumes)
The player exposes `subscribeTime(cb) -> unsubscribe`; `cb(ms | null)` receives exactly the values `onTime` gets (null when the
playhead hides). Existing constructor option `onTime` keeps working.
### Pose function (step 5 produces, step 6 consumes)
Pure `poseFor(readoutsHz: Record<string, number>, tMs: number) -> Record<jointId, number /* degrees */>` plus the exported
mapping table `MAPPING` (readout name -> joint ids, FIXED full-scale Hz, max angle, validation label, kind "extension" |
"oscillation"). First version: `mn9` -> `proboscis` extension (validated; full scale 60 Hz); `abn1` -> `antenna_l`/`antenna_r`
oscillation at a fixed 6 Hz whose amplitude follows the rate (illustrative; full scale 30 Hz); `dnp01` -> `wing_l`/`wing_r` raise
(exploratory; full scale 100 Hz). Unknown readouts are ignored; missing readouts give 0.

### Body scene API (step 4 produces, step 6b wiring consumes)
`BodyScene { invalidate(); dispose(); setPose(angles: Partial<Record<JointId, number /* degrees */>>): void; jointIds(): JointId[] }`;
`MountedBody { remove(): void; setPose(angles): void }` (`setPose` is safe BEFORE the scene exists: the latest pose is stored and
applied on creation). `setPose({})` = rest pose. Unknown joint ids are ignored. A positive angle follows the `rig` axis by the
right-hand rule about its `pivot_um`.
### Body driver (step 6a produces, 6b wiring consumes)
Pure `web/src/bodyDriver.ts`: `createBodyDriver({ setPose, smoothBins?: number })` -> `{ setActivity(act: {readouts?: NamedReadout[]; binMs: number} | null): void; onTime(ms: number | null): void; dispose(): void }`.
`onTime(ms)` samples every named readout with `sampleReadout`, builds `readoutsHz` by readout NAME (readouts with a null name are
ignored), calls `poseFor(readoutsHz, ms)` and `setPose(pose)`; `onTime(null)` or no activity -> `setPose({})` (rest). Redundant identical
poses are not re-sent. After `dispose` nothing is sent.

## Tasks (dependency order)
- [x] S1 backend (commit on feat/anim-01; 878 py) backend: all readout traces per recording (`fly` stores every watched readout; meta `readouts`; web `Activity.readouts`). Needs: nothing.
- [x] S2 player (feat/anim-02; 595 web) `subscribeTime`. Needs: nothing.
- [x] S3 asset (feat/anim-03; all joints are 'geometry (our choice)': the rigging has no axes; legs skipped): `rig` joints and per-part mesh ranges from the build script. Needs: the asset pipeline (merged).
- [x] S5 pure mapping (feat/anim-04; 29 tests) module `bodyPose.ts` (`MAPPING`, `poseFor`) with tests. Needs: the contracts above (S1 and S3 only by name).
- [x] S4 viewer (feat/anim-05; 649 web at its own branch): body drawn as separate part groups with per-group occluder meshes, joints applied as rotations. Needs: S3.
- [x] S6a driver (feat/anim-06; 22 tests; chain tip 671 web, 891 py + 1 skipped). [x] S6b wiring (feat/anim-07) wiring: playback clock -> readout sample -> `poseFor` -> joint rotations each frame; reduced motion = final pose. Needs: S2, S4, S5 (and S1 for real data).
- [x] S7 experience (feat/anim-07; honest caption): runs from `fly`, chips or the agent play on the body automatically; caption with the honesty text; the body is the main pane or a clear cue shows when it is the inset. Needs: S6.
- [x] S8 verification (parent render check below; user browser check pending): frame captures through canvas -> image at several times and doses (sugar low/high, sugar+bitter, jo_ce), unit tests of the mapping, and the user's browser check. Needs: S7.

## Checks
TDD strict (source: project config). Runners: `uv run pytest -q` (baseline 875 on main), `pnpm -C web test` (586), `pnpm -C web build`.
Delivery: stacked PRs to main (the user's choice), merged with merge commits, one PR per step or small group, branches NOT deleted
by `gh pr merge` (it closes the child PR). Engram mirror `odd/fly-animation/tasks`: PENDING (mem_save blocked by a hook).

## Parent render check of S4 (2026-11-16; canvas -> toDataURL, 3/4 view)
Rest pose looks identical to the old single-mesh drawing. Wings at 60 deg rise clearly; the proboscis at 60 deg shows new lines at the
front of the head (it was hidden inside the head at rest); antennae at +-30 deg change subtly from this angle (the mapping's maxDeg
could be raised if more presence is wanted: an illustration choice). Every joint has geometry: proboscis 1026 segs / 1884 tris,
antennae 1190 / 4526 each, wings 890 / 1999, halteres 468 / 824.

## S8 render check with REAL recordings (2026-11-16; API on the chain, driver + scene, canvas -> image)
- Found and fixed: `stimulate` recorded its preset readout with name null, so the sugar chip would have left the body at rest.
  It now records the preset name (commit 9863131; tests added). 893 py + 1 skipped.
- sugar 100 Hz, readout mn9 (recorded peak 133 Hz): proboscis 0 deg at t=0, 50 at 300 ms, 60 at 450 ms, 33 at 900 ms (follows the trace).
- sugar 20 Hz: MN9 stays at 0 Hz, the body stays at rest (low dose is weaker, honestly).
- jo_ce 100 Hz, readout abn1: antennae oscillate (-8.2 deg at 100 ms, 0 at 125 ms); the motion is subtle at this camera angle.
- Not verified: the live browser playback with the real player (no WebGL in the parent's headless run).

## Progress
- 2026-11-16 created. Wave 1 DONE (S1, S2, S3, S5; chain feat/anim-01..04 on main: 891 py + 1 skipped, 625 web). Wave 2: S4 (body groups) and S6a (driver) in parallel; wave 3: S6b wiring + S7 UX; then S8.
