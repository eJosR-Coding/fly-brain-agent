# fly-brain-agent

Explore the whole fruit-fly brain (FlyWire v783, ~139k neurons) in 3D and talk to it through an LLM agent that uses tools.

## Architecture

Dependencies point down. The core knows nothing about the web or LLMs.

```
frontend   Vite + Three.js         3D brain + chat
backend    FastAPI                 brain data (binary) + chat stream
agent      LangGraph               any chat model via init_chat_model
tools      MCP server (FastMCP)    search, count, describe neurons
core       flybrain.atlas          positions in µm, cell types, neurotransmitters
```

## Status

| Layer | State |
|---|---|
| core: atlas | done |
| tools: MCP server | done |
| agent: LangGraph loop | in progress (T1 graph done) |
| backend, frontend | planned |

## Agent

`flybrain.agent` is a hand-built LangGraph loop (no `create_agent`): the model node asks, `ToolNode` executes, and we repeat until the model answers without tool calls. Tool errors (unknown tool, bad args) go back to the model as messages instead of raising. `ask()` sets `recursion_limit` as a hard stop and returns `[hard stop] reached N steps without a final answer` instead of crashing. No CLI yet (next: MCP wiring + `flybrain-chat`).

## Core

```python
from flybrain.atlas import load_atlas

atlas = load_atlas("data/raw/flywire_neuron_annotations.tsv")
atlas.find(cell_type="PEN", side="left", limit=5)  # prefix match, filters are AND-ed
atlas.count_by("top_nt", super_class="central")    # {"acetylcholine": ..., "unknown": ...}
atlas.describe(720575940628857210)                  # one neuron as a plain dict (positions in um)
```

## Tools (MCP)

| Tool | What it does |
|---|---|
| `search_neurons` | filter by cell_type / nt / super_class / side, returns a summary + `selection_id` |
| `count_neurons` | counts grouped by a field (top N + other) |
| `describe_neuron` | one neuron by root_id |
| `list_selection` | page through a ticket (max 50 per call) |

Ticket idea: a search can match 86k neurons, so tools never return full lists. Matches stay
server-side under a `selection_id` ("sel_3f9a2c", random so a stale ticket fails loudly after a restart) that later tools (highlight, lesion) act on.

Run (stdio): `uv run flybrain-mcp`. Annotations path comes from `FLYBRAIN_ANNOTATIONS`
(default `data/raw/flywire_neuron_annotations.tsv`).

Register in Claude Code (syntax checked against `claude mcp add --help`; not run here):

```bash
claude mcp add flybrain -- uv run --directory /path/to/fly-brain-agent flybrain-mcp
```

## Data

Not versioned. Place the FlyWire neuron annotations (Schlegel et al. 2024, v783) at:

```
data/raw/flywire_neuron_annotations.tsv
```

Source: https://github.com/flyconnectome/flywire_annotations

## Development

```bash
uv sync
uv run pytest
```
