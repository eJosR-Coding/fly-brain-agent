# Feature: Level 0 experiments (connectome structure, no simulation)

## Objective
Let the agent (and the viewer) answer structural questions with the real FlyWire wiring: who talks to whom, the path between two groups of neurons, what breaks if a group is removed, and the most connected hubs.

## Problem / why
The project only has neuron positions. Every experiment the user wants (paths, stimulation, lesions, dopamine) needs the connections. Level 0 is 100% real data (no dynamics model), so it is the solid first step.

## Data (verified 2026-11-10)
- Source: frut-fly `data/raw/proofread_connections_783.feather` (16,847,997 rows: pre, post, neuropil, syn_count, nt probabilities; 54,492,922 synapses).
- Aggregated per (pre, post): 15,091,983 pairs; >= 5 synapses: 2,700,513; >= 10: 1,066,822.
- Build step: `scripts/build_connectome.py` aggregates pairs, maps root_ids to atlas rows (drops unknown ids and reports how many), saves a compact `data/processed/connectome_783.npz` (pre_row int32, post_row int32, syn uint16/int32) — gitignored.

## Design
- Core `flybrain.connectome`: load the npz into scipy CSR matrices (out-edges and in-edges) with a runtime `min_synapses` threshold (default 5).
- Queries (pure, tested on a tiny synthetic graph):
  - `neighbors(rows, direction=in|out|both, min_synapses, top)` -> partner rows + synapse totals, grouped summary by cell_type / super_class.
  - `shortest_path(source_rows, target_rows, min_synapses, max_hops)` -> hop count and one concrete path (list of rows), or none.
  - `lesion_reachability(source_rows, target_rows, removed_rows, min_synapses, max_hops)` -> how many targets are reachable before vs after removing a group.
  - `hubs(metric=in|out|total, top, filters)`.
- MCP tools (ACI rules from notes 01-05): `neighbors`, `find_path`, `lesion_effect`, `top_hubs`. They take/return selection tickets, small summaries, notes when values are clamped, and guiding errors. Results that are neuron groups return a new ticket so the viewer can highlight them.
- The viewer needs no change for highlights (tickets already highlight). Optional: path order shown in the chat.

## Constraints
- Strict TDD (`uv run pytest`), fixture graphs only in tests (never the 850 MB file).
- scipy added in its own `chore(deps)` commit.
- Memory: keep CSR int32; report load time and RAM in evidence.
- Honesty: structural answers only; say "structural path", not "signal flow"; sign (excitatory/inhibitory) not used at Level 0.

## TDD
- Mode: strict (source: global user config). Runner: `uv run pytest`.

## Tasks
- [ ] L0-1 Build script + connectome core (load, threshold, neighbors, shortest path, lesion reachability, hubs). Route: delegated writer.
- [ ] L0-2 MCP tools + API/agent wiring + real-data spot checks + README. Route: delegated writer.

## Acceptance criteria
- Build script produces the npz from the real file; load < ~5 s; numbers reported.
- Tools return small payloads (< ~2k chars) and tickets; path query between two known groups returns a sensible hop count on real data.
- `uv run pytest` and `pnpm -C web test` green.

## Progress / evidence
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

## Next step
L0-1.
