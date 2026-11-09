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
| tools: MCP server | planned |
| agent, backend, frontend | planned |

## Core

```python
from flybrain.atlas import load_atlas

atlas = load_atlas("data/raw/flywire_neuron_annotations.tsv")
atlas.find(cell_type="PEN", side="left", limit=5)  # prefix match, filters are AND-ed
atlas.count_by("top_nt", super_class="central")    # {"acetylcholine": ..., "unknown": ...}
atlas.describe(720575940628857210)                  # one neuron as a plain dict (positions in um)
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
