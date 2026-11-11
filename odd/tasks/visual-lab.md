# Feature: visual lab (see every result in 3D)

## Objective
Every agent result should be visible in the 3D viewer, not only read: inspect any neuron by clicking it, see paths as glowing cables, see the direction of activity (latency), see what changes when neurons are silenced (difference map), and see spikes/rates as charts. Then the experiment bench (dose-response, silencing screen, statistics) builds on these views.

## Tasks
- [x] VL1 Click-to-inspect: raycast picking on the point cloud, neuron card (type, class, side, NT, position, top in/out partners). Route: delegated writer. Evidence: commits `feat(api): neuron details endpoint for the viewer`, `feat(web): click a neuron to inspect it`; pytest 472, vitest 288; capture v7-ficha-neurona.png.
- [x] VL2 Paths as cables: find_path results drawn as line segments between the path neurons, thickness by hop synapses, animated flow along the signal direction; SSE `path` event. Route: delegated writer. Evidence: commits `feat(api): path events for find_path results`, `feat(web): draw paths as glowing cables`; pytest 482, vitest 302; captures v7-camino-cables.png (R1-6 > L3 > MTe07 > KCg-s1, 19/5/7 syn, real data) and a reduced-motion variant.
- [x] VL3 Latency map: first-spike time per neuron, static map toggled with the wave. Route: delegated writer. Evidence: commit `feat(web): latency map of the activity wave`; RED: `Cannot find module '../src/latency'` + 4 failing activity tests, then GREEN; capture v8-latencia.png.
- [x] VL4 Difference map: baseline and silenced recordings kept (`compare_id`, `silenced_rows`, readout trace in meta and `activity` event; `silence_preset` in `POST /simulate`), diff mode with rings (lost), dots (gained), crosses (silenced). Route: delegated writer. Evidence: commits `feat(api): keep baseline and silenced recordings for comparison`, `feat(web): difference map between baseline and silenced runs`; RED: 8 failing pytest, then GREEN; capture v8-diferencias.png (silencing CB0248: 36 lost, 263 gained, MN9 60 -> 71 Hz).
- [x] VL5 Charts: raster of the 30 busiest neurons + readout trace (baseline solid vs silenced dashed), playhead, tooltips, `GET /simulations/{id}/rows`. Dose-response curves stay with EB1. Route: delegated writer. Evidence: commit `feat(web): raster and readout charts`; RED: `smooth is not a function`, module-missing for charts; captures v8-graficos.png, v8-cables-finos.png (thinner cables, commit `fix(web): thinner path cables`).
- [ ] EB1-3 Experiment bench tools: dose-response sweep, silencing screen, statistics with confidence intervals. Route: delegated writer.

## Constraints
- Strictly black and white, Penpot-derived tokens in style.css; readable (12px mono min, contrast >= 4.5:1); reduced motion respected.
- Strict TDD for pure logic (vitest, pytest). Model output untrusted (cards and labels via textContent only).
- Real data only; illustrations labeled. GitHub Flow; Claude merges after checks.

## Progress / evidence
- After VL5: pytest 492, vitest 347. Not done: a viewer button that triggers a compare run (the API supports it; the diff is reachable from chat answers); charts hidden on phone widths.
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).
