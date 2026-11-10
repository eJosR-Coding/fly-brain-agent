<p align="center">
  <img src="assets/cover.png" alt="fly-brain-agent: a minimal geometric fruit fly drawn in white lines on black" width="640">
</p>

# fly-brain-agent

/flaɪ breɪn ˈeɪ.dʒənt/
noun

1. an LLM agent that lets you talk to a whole fruit-fly brain, neuron by neuron.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.13+-3776AB?logo=python&logoColor=white)
[![uv](https://img.shields.io/badge/managed%20with-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![MCP](https://img.shields.io/badge/tools-MCP-000000)](https://modelcontextprotocol.io)
[![LangGraph](https://img.shields.io/badge/agent-LangGraph-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![FlyWire](https://img.shields.io/badge/data-FlyWire%20v783-6A5ACD)](https://flywire.ai)
![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-agent%20works%2C%203D%20viewer%20next-orange)

**Ask a question in plain language. An agent searches 139,248 real neurons and answers.**

```text
$ uv run --env-file .env flybrain-chat --model google_genai:gemini-3.5-flash \
    "How many dopamine neurons are there, and in which super_class?"

There are 5,909 dopamine neurons:            (real run, answer trimmed)
  central 5,705 · sensory 129 · optic 26 · visual_centrifugal 21 · descending 12 · ...
```

The FlyWire project mapped every neuron of an adult fruit-fly brain. This repo turns that
atlas into **tools an LLM can use**: search neurons by type, neurotransmitter or side,
count them, describe one, page through results. The agent picks the tools, we run them,
and the answer comes back grounded in real data instead of the model's memory.

You choose the model. The same agent runs on Gemini, Claude, GPT or a local model, and the
same tools work from Claude Code or any other MCP client without changing a line.

## Highlights

- **The whole brain, not a toy.** FlyWire v783: 139,248 neurons with positions in µm,
  cell types and neurotransmitters.
- **Tools over MCP.** A standalone [Model Context Protocol](https://modelcontextprotocol.io)
  server. Plug it into Claude Code, our agent, or anything that speaks MCP.
- **Any model.** LangChain's `init_chat_model`: `--model google_genai:...`,
  `anthropic:...`, `openai:...`.
- **Selection tickets, not token floods.** A search that matches 86k neurons returns a
  short summary plus a ticket like `sel_3f9a2c` (about 1.3k characters instead of millions
  of tokens). Later tools act on the ticket.
- **A loop with a brake.** A hand-built LangGraph loop (no black-box `create_agent`) with
  a hard step limit, so a confused model can't spin forever.
- **Errors that teach.** Unknown tickets or tools come back as messages the model can read
  and fix itself, not stack traces.
- **Keys stay out of history.** Provider keys live in a gitignored `.env` loaded with
  `uv run --env-file`.

## Architecture

Dependencies point inward. The core knows nothing about MCP, the web or LLMs.

```text
frontend   Vite + Three.js         3D brain + chat                        (planned)
backend    FastAPI                 brain data + chat stream               (planned)
agent      LangGraph               any chat model, MCP client             done
tools      MCP server (FastMCP)    search, count, describe, page          done
core       flybrain.atlas          positions in µm, cell types, NTs       done
```

```text
 you ──► flybrain-chat ──► LLM (any provider)
                 │            │ "call count_neurons(nt=dopamine)"
                 ▼            ▼
           LangGraph loop ── MCP ──► flybrain MCP server ──► atlas (139k neurons)
                 ▲                                │
                 └──────── tool result ◄──────────┘   repeat until the model answers
```

## Quick start

```bash
git clone https://github.com/eJosR-Coding/fly-brain-agent.git
cd fly-brain-agent
uv sync
```

Download the FlyWire neuron annotations (see [Data](#data)) to
`data/raw/flywire_neuron_annotations.tsv`, then add a key:

```bash
cp .env.example .env && chmod 600 .env   # then put your key in .env
uv run --env-file .env flybrain-chat --model google_genai:gemini-3.5-flash "Find PEN neurons on the left side"
```

Options: `--model provider:model` (default `anthropic:claude-sonnet-5-5`, or set
`FLYBRAIN_MODEL`), `--max-steps N`. Other providers: `uv add langchain-openai`, then
`--model openai:<model>`.

## Tools (MCP)

| Tool | What it does |
|---|---|
| `search_neurons` | filter by cell_type / nt / super_class / side; returns a summary + `selection_id` |
| `count_neurons` | counts grouped by a field (top N + other) |
| `describe_neuron` | one neuron by root_id |
| `list_selection` | page through a ticket (max 50 per call) |

Run the server on its own (stdio): `uv run flybrain-mcp`. The annotations path comes from
`FLYBRAIN_ANNOTATIONS` (default `data/raw/flywire_neuron_annotations.tsv`).

Use it from Claude Code:

```bash
claude mcp add flybrain -- uv run --directory /path/to/fly-brain-agent flybrain-mcp
```

## How the agent works

`flybrain.agent` is an explicit LangGraph loop: the model node asks, `ToolNode` executes,
and we repeat until the model answers without tool calls. `ask()` sets `recursion_limit` as
a hard stop and returns `[hard stop] reached N steps without a final answer` instead of
crashing.

The agent is an MCP client: `flybrain.agent_mcp` launches the server over stdio and keeps
**one session per conversation**. That matters: tickets live in the server process, and the
adapter's default `get_tools()` opens a new session (a fresh server) per tool call, which
would lose them. A test pins this behavior.

## Core (no LLM needed)

```python
from flybrain.atlas import load_atlas

atlas = load_atlas("data/raw/flywire_neuron_annotations.tsv")
atlas.find(cell_type="PEN", side="left", limit=5)  # prefix match, filters are AND-ed
atlas.count_by("top_nt", super_class="central")    # {"acetylcholine": ..., "unknown": ...}
atlas.describe(720575940628857210)                  # one neuron as a plain dict (µm)
```

## Roadmap

- [x] Atlas core: positions, types, neurotransmitters
- [x] MCP server with selection tickets
- [x] LangGraph agent + `flybrain-chat` CLI, tested end to end with Gemini
- [ ] FastAPI backend: binary brain points + streamed chat
- [ ] Three.js viewer: the brain in 3D, neurons light up as the agent works
- [ ] Tools that change the brain: lesion a selection, scale a neurotransmitter's gain
- [ ] Simulation: run activity through the connectome and watch it respond

## Data

Not included in this repo. Place the FlyWire neuron annotations (Schlegel et al. 2024,
materialization v783) at:

```text
data/raw/flywire_neuron_annotations.tsv
```

Source: <https://github.com/flyconnectome/flywire_annotations>. The data belongs to the
FlyWire consortium and its authors; follow their terms and cite their papers if you use it.

## Development

```bash
uv sync
uv run pytest        # 40 tests, no network or API key needed (fake chat model)
```

Work is tracked per feature in [`odd/tasks/`](odd/tasks/), with test-first commits.

## License

Code: [MIT](LICENSE). FlyWire data is not covered by this license (see [Data](#data)).
