# Feature: visual lab (see every result in 3D)

## Objective
Every agent result should be visible in the 3D viewer, not only read: inspect any neuron by clicking it, see paths as glowing cables, see the direction of activity (latency), see what changes when neurons are silenced (difference map), and see spikes/rates as charts. Then the experiment bench (dose-response, silencing screen, statistics) builds on these views.

## Tasks
- [ ] VL1 Click-to-inspect: raycast picking on the point cloud, neuron card (type, class, side, NT, position, top in/out partners). Route: delegated writer.
- [ ] VL2 Paths as cables: find_path results drawn as line segments between the path neurons, thickness by hop synapses, animated flow along the signal direction; SSE `path` event. Route: delegated writer.
- [ ] VL3 Latency map: color activity by first-spike time (early -> late), toggle with the wave. Route: delegated writer.
- [ ] VL4 Difference map: silenced vs baseline (lost / gained activity). Route: delegated writer.
- [ ] VL5 Charts: raster of top neurons + readout trace; later dose-response curves. Route: delegated writer.
- [ ] EB1-3 Experiment bench tools: dose-response sweep, silencing screen, statistics with confidence intervals. Route: delegated writer.

## Constraints
- Strictly black and white, Penpot-derived tokens in style.css; readable (12px mono min, contrast >= 4.5:1); reduced motion respected.
- Strict TDD for pure logic (vitest, pytest). Model output untrusted (cards and labels via textContent only).
- Real data only; illustrations labeled. GitHub Flow; Claude merges after checks.

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).
