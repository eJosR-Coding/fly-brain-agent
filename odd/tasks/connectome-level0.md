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
- [x] L0-1 Build script + connectome core (load, threshold, neighbors, shortest path, lesion reachability, hubs). Route: delegated writer.
- [x] L0-2 MCP tools + API/agent wiring + real-data spot checks + README. Route: delegated writer.

## Acceptance criteria
- Build script produces the npz from the real file; load < ~5 s; numbers reported.
- Tools return small payloads (< ~2k chars) and tickets; path query between two known groups returns a sensible hop count on real data.
- `uv run pytest` and `pnpm -C web test` green.

## Progress / evidence
- Decisions: self-loops dropped (counted); synapses summed over neuropils; threshold applied at load (default 5); build lives in `flybrain.connectome_build` (entry point `flybrain-build-connectome`, wrapper `scripts/build_connectome.py`); pyarrow added (pandas cannot read feather without it).
- RED evidence: build tests collection error `ModuleNotFoundError: flybrain.connectome_build`; core `ModuleNotFoundError: flybrain.connectome`; tools 17 failed (`module 'flybrain.tools' has no attribute 'neighbors'`); MCP/API 9 failed (new tools/`conn`/`connectome_path` missing). Each went GREEN (final `uv run pytest -q`: 283 passed; `pnpm -C web test`: 99 passed).
- Real build: 16,847,997 rows in -> 15,090,883 pairs; dropped 1,117 rows with unknown ids, 0 self-loops; 43.6 MB; 14.4 s. Edges at >= 5: 2,700,429.
- Load: 0.70 s, peak RSS 228 -> 486 MB (+258 MB incl. both CSR directions).
- Spot checks (real data, no LLM; all < 140 ms, payloads 350-1250 chars): neighbors(out) of PAM (307 neurons): 179 partners, 11,119 synapses, top = MBON06/MBON03/MBON05 and KC subtypes (KCab-p 75); neighbors(in) of PAM: 996 partners, mostly KCs, plus MBON05, SMP114, APL, DPM. find_path R1-6 (8,452) -> KC (5,177): 3 hops, R1-6 -> L3 -> MTe07 -> KCg-s1 (27 ms). R1-6 -> descending: 3 hops via L1 -> MTe01b -> DNp53. top_hubs total: CT1 (149,712 / 148,605), APL (127,080), LPi15, Li33, DPM. top_hubs out among KC: KCg-s2 475, KCg-s1 304. lesion EPG on PEN -> PFL (hops 4/6): 50 -> 50 reachable, lost 0 (redundant routes); removing PEN on EPG -> PFL (hops 2/3/4): 50 -> 50, lost 0; removing MTe07 on R1-6 -> KC (4 hops): 301 -> 300 reachable of 5,177 KCs, lost 1.
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).
- Engram mirror: pending (mem_save blocked by hook/auto-mode in this session).

- Review (2026-11-10): G1 ad0db28..dc2650a (516), G2 dc2650a..24fcd10 (396), G3 24fcd10..e4360e5 (272, HIGH, 4 lenses), all granted, approved, acknowledged. Follow-ups: a corrupt/non-zip npz raises BadZipFile at load and the optional-load catch is too narrow (connectome.py:67-71, mcp_server.py:29) -> should degrade to 'no connectome' with one stderr line; top_hubs returns a ticket even when empty (tools.py:345).
- Parent spot check: find_path R1-6 -> KC re-run independently: 3 hops (R1-6 -> L3 -> MTe07 -> KCg-s1), load 0.75 s.

## Next step
Parent: read-only review of the spot checks; optional Level 1 (stimulation) design. Not pushed.

## Path strength
Problem (user-reported via the MCP tool): `find_path` returned one arbitrary fewest-hops path with no hint of how strong each hop is or whether stronger or alternative routes exist.

Decision: `Connectome.strongest_paths` replaces the BFS-parent path in `find_path`. Among all minimum-hop routes it maximizes the bottleneck (weakest hop, in synapses), then the total synapses, then the lowest rows. No new parameter; `max_hops` and the structural note stay (the note now says strength = synapse count, a proxy for weight, not sign or efficacy). `shortest_path` is kept for other callers.

Algorithm (O(E) over the layered DAG, numpy/CSR only): BFS gives distances; keep edges from layer d to d+1; backward pass from the targets prunes edges that cannot reach one; a widest-path DP (max of min) gives the best bottleneck; a second max-sum DP over edges at least that strong picks the best total (two passes because (bottleneck, total) is not safe to optimize prefix by prefix; a test pins this). Route count = sum DP over the pruned DAG (float64, reported capped at 1,000,000 with `shortest_path_count_capped`). Alternatives = up to 3 more routes, strongest first, found by best-first search over "ban this intermediate neuron" subproblems (bounded to 60 DAG passes), each distinct in at least one intermediate neuron. Payload fields: `path[].synapses_from_previous`, `min_hop_synapses`, `total_synapses`, `shortest_path_count`, `alternative_routes[] = {chain (cell types), min_hop_synapses}`.

RED evidence: `tests/test_connectome.py` 11 failed, 23 passed (`AttributeError: 'Connectome' object has no attribute 'strongest_paths'`); `tests/test_connectome_tools.py` 3 failed, 18 passed (`KeyError: 'shortest_path_count'`, missing `synapses_from_previous`); `tests/test_mcp_server.py` 1 failed (docstring lacked `synapses_from_previous` etc.). All GREEN afterwards; `uv run pytest -q`: 298 passed (was 283).

Real data (no LLM; load 0.88 s; `find_path` call 42-56 ms; payloads 962 and 1,076 chars):
- R1-6 -> KC: 3 hops, R1-6 -> L3 (19) -> MTe07 (5) -> KCg-s1 (7); bottleneck 5, total 31; `shortest_path_count` 2; one alternative with the same cell-type chain (a different neuron pair). Identical rows to the old answer: the old path happened to be the strongest, and the route is weak (bottleneck 5 = at the threshold, so structurally present but barely).
- R1-6 -> descending: 3 hops, R1-6 -> L1 (36) -> L5 (131) -> DNc01 (11); bottleneck 11, total 178; `shortest_path_count` 341; alternatives `R1-6 > L2 > L5 > DNc01` (11), `R1-6 > L1 > L5 > DNc01` (10), `R1-6 > L2 > Tm4 > DNp11` (10). The old answer (R1-6 -> L1 -> MTe01b -> DNp53) had hops 8 / 5 / 5, bottleneck 5: the strongest path differs and is more than twice as strong at the bottleneck.

Known limits: alternatives are distinct by neuron, so their cell-type chain can equal the main one; with several sources/targets a 1-hop answer has no alternatives; only intermediate neurons are varied.
