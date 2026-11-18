# Fly behaviors: the simulated brain decides, a physics body executes (walk, fly, escape, feed, groom)

## Objective
The fly reacts to whatever the user asks (feed, groom, escape, walk, move somewhere, fly): the whole-brain model decides WHICH
behavior from the descending neurons (DNs) it activates, and a published physics body model executes HOW the body moves.
Every link is validated against published data, and every link that is our choice is labelled. Requested 2026-11-17:
"todo lo que falta modelalo ... caminar volar escapar reaccionar ... valida lectura y verifica el modelo el movimiento real
... ese camino grande pero de a poco".

## Architecture (target)
1. Brain (existing): Shiu et al. 2024 LIF on FlyWire v783. Output = DN activity per bin.
2. Behavior decoder: DN readouts -> behavior commands, each with evidence: MN9 (feeding, validated), aBN1 (grooming,
   validated readout), DNp01 giant fiber (escape), DNg100 (walking; VNC model Pugliese et al. 2026), turning/backward DNs,
   flight DNs (to be identified with evidence).
3. Motor execution: published physics models (NeuroMechFly v2 / flygym: walking with CPG and a descending command; flybody,
   Vaxenburg et al. Nature 2025: walking and flight with trained controllers) and the VNC rate model where it applies.
   Learned controllers are NOT connectome-derived: labelled "executed by a trained controller".
4. Viewer: our body mesh replays the simulator's joint angles (needs a hierarchical leg rig).
5. Verification: movement vs real-fly data (stepping 7-15 Hz, tripod coordination, takeoff latency, flight wingbeat).

## Honesty rules (hard)
- Brain decides only through DN readouts that evidence links to a behavior; body motion from a trained controller is an
  execution model, not the connectome. No "the fly feels/wants/decides consciously". No forbidden words in captions.
- A behavior that the brain model does not trigger from a stimulus is shown as not triggered (no faking).
- Every phase ends with a validation gate; a failed gate is reported, not hidden.

## Phases (each small, with its own gate)
- [x] P0 feasibility (scratch only, no project change):
  - [x] P0a physics: install the pinned flygym 2.1.1 (+ MuJoCo) in an isolated env; run walking headless on CPU; measure
        speed; export joint angles; check stepping frequency and tripod coordination against published values.
  - [x] P0b behaviors: evidence for flight and escape DNs and their presence in FlyWire v783; flybody availability
        (license, pretrained flight/walk policies, CPU runtime); escape takeoff models.
- [ ] P1 decoder: validate each DN trigger in OUR brain model (which stimuli drive which DNs), with tests.
- [ ] P2a physics walking replay in the viewer (branch feat/physics-walk; user chose easiest first):
  - [x] W1 data: `scripts/export_flygym_walk.py` (runs in an external flygym 2.1.1 env, not a project dependency) writes
        `web/public/fly-walk.json`: per-body world transforms of all leg bodies from MuJoCo (thorax-relative, so the fly walks
        in place), the same bodies' transforms at the neutral pose, 200 Hz, a seamless loop, provenance. Gate: our assembled
        mesh's body origins match MuJoCo's neutral-pose origins (tolerance stated).
  - [x] W2 viewer: per-leg-body groups; each frame applies M_body(t) * M_body(neutral)^-1 to that body's lines and occluder;
        pure transform module with tests; existing joints (proboscis, antennae, wings) keep working.
  - [x] W3 experience + verification: a chip "physics walk (demo)" with an honest caption (flygym tripod CPG controller,
        not yet driven by the brain); renders at several phases; tripod check (LF+RM+LH vs RF+LM+RH in antiphase).
  - P2a evidence (2026-11-17, uncommitted on feat/physics-walk; parent still to check in the browser):
    W1: `scripts/export_flygym_walk.py` -> `web/public/fly-walk.json` (373 KB, 200 frames @200 Hz = 12 cycles @12 Hz = 1 s,
    48 leg bodies, thorax-relative, um ints + quat x10000). Gate: our body origins vs MuJoCo neutral-pose origins, max 0.84 um,
    median 0.51 um (rounding only; reference = flygym model in the axis order of pose/_manual_specs/neutral.yaml; flygym's own
    walking-model keyframe is an IK fit up to 86.5 um off, so it is NOT used as the neutral). Loop seam 6.5 um / 0.66 deg
    (ordinary step 453 um). W2: `bodyClip.ts`, leg groups + `setClipFrame` in `body.ts`, `MountedBody.setClipFrame`.
    W3: `bodyWalk.ts` + chip "physics walk (demo)". Tripod/12 Hz verified from the export (`tests/test_fly_walk_asset.py`).
    Not verified: real WebGL render (parent's visual check).
- [ ] P2 walking end to end: brain DN -> descending command -> physics walking -> joint angles -> body replay.
- [ ] P3 escape (looming -> DNp01 -> takeoff), P4 flight, P5 turning / moving to a target.
- [ ] Each phase: verification against real-fly numbers + user browser check.

## Related
- `odd/tasks/vnc-legs.md` (branch feat/vnc-legs): the VNC rate-model front-leg rhythm; folded in as an evidence source
  for P1/P2 (DNg100 threshold: 150 -> nothing, 250 -> rhythm; female datasets FANC/BANC bundled in the Pugliese repo).

## Checks
TDD strict (project config) for project code. Runners: `uv run pytest -q`, `pnpm -C web test`, `pnpm -C web build`.
Heavy simulators stay out of the project's dependencies until a phase proves they are needed (precomputed or optional).

## Findings so far (2026-11-17)
- P0a physics (verified, scratch): flygym 2.1.1 + MuJoCo 3.16.0 walk headless on CPU, ~16 s wall per simulated second;
  tripod gait (LF+RM+LH vs RF+LM+RH), 12 Hz fixed CPG frequency, ~14 mm/s straight; 2-value descending drive [L,R]
  (0.4/1.2 -> ~160 deg/s spin). Joint angles exported (66 DoFs, 200 Hz, same meshes/rigging as our viewer).
  Real flies: up to ~15 Hz stepping, 5-30 mm/s (Mendes et al. 2013, eLife; to re-verify).
- P0b evidence: turning DNa01/DNa02 (Rayshubskiy, eLife 102230), backward MDN (Feng et al. 2020), escape DNp01 giant fiber
  (von Reyn 2014) + DNp02/04/06/11, landing DNp09 (weak). Flight DNs: thin evidence. flybody (Apache-2.0) walks/flies but
  uses its own geometry; CPU ~7x (walk) / ~42x (flight) slower than real time; pretrained weights location unverified.
  All these DN types exist in our FlyWire v783 table (DNg12/DNg02 only as split subtypes; BDN2/aDN names absent).
- P1 screen in OUR brain: gustatory/antennal presets (sugar, bitter, water, jo_ce at 100/200 Hz) drive none of the
  walking/turning/escape DNs (all <= 2.2 Hz). Visual detector input (looming_escape, LPLC2+LC4 driven directly, 100 Hz,
  500 ms, 5 trials) drives the escape DNs: DNp01 ~25 Hz, DNp04 ~15 Hz, DNp06 ~7 Hz, DNa01/02 ~2 Hz, MDN 0 — but NOT
  selectively: receding and shuffled control give the same, dimming more (DNp01 33 Hz). Real flies discriminate looming.
  The photoreceptor path does not propagate in this LIF model (graded visual neurons; Shiu et al. 2024 limits), so visual
  behaviors need input at detector level (our choice) or a visual-system model.
- Bottleneck: sensory input (vision, odor, wind) reaching the DNs, not only the VNC.

## Progress
- 2026-11-17 created. P0a and P0b launched in parallel (delegated, scratch only).
