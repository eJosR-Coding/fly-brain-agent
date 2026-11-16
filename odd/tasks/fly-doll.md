# Fly doll: talk to the simulated fly and change its conditions from the chat

## Objective
The chat agent behaves like a simulator with a doll: "give sugar", "now also bitter", "remove the bitter",
"reset". The doll keeps its current conditions between messages, combines several stimuli with different
rates in one run, and reports the reaction in behavior terms (feeding, grooming, escape) plus what changed
versus the previous run. No graphical widgets (user rejected the Terrarium UI on 2026-11-16): everything lives in
the agent tools and the chat answer.

## Authorized scope
User approved on 2026-11-16 the core: (1) doll state, (2) combined stimuli with per-stimulus rates,
(6) behavior-named readouts. Later, not yet approved: more sense presets, timeline, global model knobs.

## Honesty rules (hard)
- It is a model of neurons (LIF, Shiu et al. 2024): no body, no physiology. Never say the fly dies, is harmed or
  overdoses; runaway activity is "the model saturates" (an artifact), the tool already warns about it.
- Temperature, hunger and sleep are NOT model inputs. "Conditions" means which neurons are driven, at what
  rate, which are silenced, and which readouts are watched.
- Only validated mappings are called validated (MN9 proboscis, sugar/bitter/water GRNs, antennal JO -> grooming
  aBN1 per Shiu). Everything else says "exploratory".
- Do not invent neuron ids: take them from the primary source and verify them against the v783 atlas.

## Phase 0 evidence (2026-11-16)
- Agent tools today (main): search_neurons, count_neurons, describe_neuron, list_selection, neighbors, find_path,
  lesion_effect, top_hubs, preset_selection, stimulate (now also `preset=`), dose_response, silencing_screen.
- `Simulation.run_schedule` exists (cached via SimulationService.run_schedule); overlapping segments ADD rates for
  the same neuron; segments over different neurons are independent. It is not exposed to the agent.
- Atlas lookups by cell_type: aBN1 0, aBN2 0, MN9 0 (no name), DNp01 2 (giant fiber, left+right), DNg11 6.
  Presets hold root_ids from Shiu's notebook (sugar 21, mn9 1). Shiu's figures.ipynb has groups sugar_N, water_N,
  bitter, IR94e, JON_CE, JON_F, JON_D_m and readouts MN9, DN1, DN2, aBN1.
- Measured earlier: pure sugar GRNs reproduce Shiu (37 active at 10 Hz, 430 at 200 Hz); sugar 100 Hz + bitter
  100 Hz gives 0/24 proboscis motor neurons (bitter blocks sugar); the chat takes 2 model rounds for "dale azucar".
- The deployed chat server is one process: server-side doll state is shared by every conversation (single-user
  local app); documented, not solved.

## Tasks
- [x] T1 data: presets from Shiu's figures.ipynb (commit 91bdd1e7), ids parsed with ast.literal_eval (no notebook
      code run), verified against the v783 atlas. (route: delegated writer.) Added bitter (20/20 in atlas, all
      sub_class bitter), water (18/18), jo_ce (69/69, all wind_gravity sub_class; readout abn1), abn1 (id
      720575940630907434), dnp01 (2, from the atlas, exploratory). Fields role/meaning/validation on every preset;
      preset_selection and the stimulate `preset` block carry them. RED: 12 failed, 9 passed. GREEN: 709 passed
      (parent re-ran, and cross-checked the ids against the atlas). Not given a default readout: bitter, water.
      Notes: annotation drift v630 -> v783: of Shiu's 20 sugar ids present, 9 are annotated sugar and 5 high_salt;
      his water set holds 7 annotated sugar. Shiu's sets stay the reference for reproducing the paper.
- [x] T2 doll tool `fly` (route: delegated writer). New src/flybrain/doll.py (pure state, atomic apply: reset ->
      remove -> set), tools.fly (one StimulusSchedule segment per stimulus, own rate; silence rows unioned; shared
      helper `_activity_lists` extracted, `stimulate` output unchanged), one Doll per MCP server (shared by all
      conversations, lost on restart), prompt guidance. RED: collection ModuleNotFoundError; GREEN: 741 passed
      (parent re-ran). Tool-defs budget 8500 -> 9650 (+1150; fly definition 1164 chars); prompt now ~1792 of the
      1800-char cap (about 8 chars of headroom: the next prompt change must trim something).
      REAL BRAIN (500 ms, 3 trials): sugar 100 -> 342 active, MN9 63.3 Hz (6.9 s); + bitter 100 -> 299 active, MN9
      4.0 Hz (-59.3, 6.5 s; the earlier 0/24 was the 24-neuron proboscis group, MN9 is one neuron); remove bitter
      -> back to 342 / 63.3 (cache hit); reset -> ran false; jo_ce 100 -> 219 active, abn1 (grooming) 25.3 Hz.
      No runaway warnings. Caveat: after a reset the first run reports "removed sugar" vs the last run.
- [x] T3 chat evaluation (real /chat path, scripted conversations; script in the session scratchpad).
      deepseek-flash, 5 messages: fly used each time, 2 rounds per message, 3.6-12.3 s (6.5 s of it is the first
      simulation, repeats come from the cache); sugar -> + bitter: MN9 63.3 -> 4.0 Hz and back on removal.
      Found and FIXED (3 commits, tests RED -> GREEN, suite 749): (1) "dale viento" took 3 rounds/17.9 s because
      nothing said jo_ce = antenna wind/touch -> tool description now says so; (2) after a reset the next run was
      compared with the pre-reset run -> reset clears the baseline; (3) the answer quoted field names -> prompt.
      Then groq gpt-oss-120b used `stimulate` instead of `fly` for "dale azucar" (two competing prompt rules,
      English-only trigger words) -> one rule: presets + give/dale/add/remove/silence/reset -> fly; stimulate only
      for a selection_id. Prompt 1696 of 1800 chars. After the fix both models call fly. groq also hit a provider
      429 rate limit (not ours).
      OPEN, user decision: the doll is shared by every conversation of the server, and this was observed: a new
      groq chat inherited sugar+bitter from the previous deepseek chat, so "add bitter" showed no change.

## Checks
TDD strict (source: project config). Runners: `uv run pytest -q` (baseline 697 on feat/stimulate-presets),
`pnpm -C web test`, `pnpm -C web build`.
Delivery: ask-on-risk; stack of local branches main -> feat/stimulate-presets -> feat/fly-doll; no push or PR
yet, the chain strategy question is asked when the user wants to open PRs. Reviewed boundary: 81d6058 on
feat/stimulate-presets (approved 2026-11-16; follow-up fix 3ff6e7d is under budget).
Engram mirror `odd/fly-doll/tasks`: PENDING (mem_save is blocked by a hook that rewrites its input).

## Reviews
- Slice 81d6058 -> 92c0d8a (T1 + T2, 976 lines, risk high): consent granted, native 4-lens review APPROVED,
  acknowledged (lineage review-72902dc79d59908f, authority burned). 9 non-blocking findings, 3 WARNINGs:
  shared doll across conversations (mcp_server.py:57; documented, kept: single-user local app), conditions committed
  before the run, and no lock. The last two FIXED in the follow-up commit (rollback on a failed run + doll.lock held
  for the whole change; 3 new tests RED -> GREEN, suite 744). Open suggestions (style): bounded input of
  stimuli/silence/readouts lists (tools.py:700-712), duplicated activity-rows code (tools.py:545-549), import order
  (tools.py:10-14), magic seed and DOLL_TOP (tools.py:662-663), role check reads inverted (tools.py:567-569).
  Reviewed boundary is now 92c0d8a.

## Progress
- 2026-11-16 created after phase 0. T1-T3 done. Next: decide how to scope the doll (per conversation / idle reset), then the pending native review of fc188ba (high risk, consent needed).
