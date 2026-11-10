<p align="center">
  <img src="assets/cover.png" alt="fly-brain-agent: a minimal fruit fly in white lines on black, with a status panel reading LangGraph OK and MCP server (FastMCP) OK" width="800">
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
![Tests](https://img.shields.io/badge/tests-273%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-agent%20and%203D%20viewer%20work-orange)

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
backend    FastAPI                 brain data + chat stream (SSE)         done
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
`FLYBRAIN_MODEL`), `--max-steps N` (max tool rounds, default 6: `1` allows one tool round
plus the final answer). Other providers: `uv add langchain-openai`, then
`--model openai:<model>`.

## Tools (MCP)

| Tool | What it does |
|---|---|
| `search_neurons` | filter by cell_type / nt / super_class / side; returns a summary + `selection_id` |
| `count_neurons` | counts grouped by a field (top N + other) |
| `describe_neuron` | one neuron by root_id |
| `list_selection` | page through a ticket (max 50 per call) |

Behavior worth knowing:

- Filters are trimmed; an empty or whitespace-only filter means "no filter". Values over 200
  characters are rejected. `side` is `left`, `right`, `center` or `na`.
- Out-of-range paging never errors: `offset`, `limit` (max 50) and `count_neurons` `top`
  (max 100) are clamped and the answer carries a `note` saying what changed
  (e.g. `limit capped to 50`, `offset 900 is past the end (total 42)`).
- A search with no match lists the valid values when `nt`, `super_class` or `side` is off.
- Tickets live in server memory, capped at 100 (`FLYBRAIN_MAX_SELECTIONS`), least recently
  used evicted first. An expired ticket says which search to run again.
- If the annotations file is bad or missing, the server prints one line to stderr and exits
  with code 1.

Run the server on its own (stdio): `uv run flybrain-mcp`. The annotations path comes from
`FLYBRAIN_ANNOTATIONS` (default `data/raw/flywire_neuron_annotations.tsv`).

Use it from Claude Code:

```bash
claude mcp add flybrain -- uv run --directory /path/to/fly-brain-agent flybrain-mcp
```

## How the agent works

`flybrain.agent` is an explicit LangGraph loop: the model node asks, `ToolNode` executes,
and we repeat until the model answers without tool calls. `ask()` caps the tool rounds
(`--max-steps`) and returns `[hard stop] reached N tool rounds without a final answer`
instead of crashing. A malformed tool call goes back to the model so it can retry, an empty
reply is reported as `[no answer] ...`, and if the MCP server dies mid-run the run stops with
`the flybrain server stopped: <cause>` rather than letting the model improvise.

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

## Web API

A FastAPI backend for a browser viewer. It loads the atlas once, hosts the MCP server
in-process (in-memory session, so the agent still only talks to tools over MCP) and shares
the same selection store, which lets it turn a ticket into row indices for highlighting.

```bash
uv run --env-file .env flybrain-web --model google_genai:gemini-3.5-flash   # 127.0.0.1:8000
```

Options: `--host`, `--port`, `--model` (or `FLYBRAIN_MODEL`), `--max-steps`. CORS allows
`http://localhost:5173` (Vite) by default; set `FLYBRAIN_CORS_ORIGINS` to a comma-separated
list to change it. `--help` needs no model or key.

Each chat turn has a time limit (`FLYBRAIN_TURN_TIMEOUT`, default 90 s). Provider SDKs can
retry overload errors (HTTP 503) with backoff for minutes; when the limit hits, the stream
sends an `error` event and then `done` instead of going silent. A cut turn (limit, hard stop,
model error) leaves the conversation usable: tool calls it never answered are closed with a
"cancelled" tool result, so the next message on the same `conversation_id` works. The value
must be a positive number; anything else stops startup with a one-line error.

| Endpoint | What it returns |
|---|---|
| `GET /health` | `{"status": "ok", "neurons": N}` |
| `GET /brain/meta` | the binary contract below, as JSON |
| `GET /brain/points` | all positions, `application/octet-stream` (~1.67 MB) |
| `POST /chat` | `text/event-stream`, body `{"message", "conversation_id"?}` |
| `GET /selections/{id}/indices` | row indices of a ticket, uint32 LE; 404 if unknown or expired |

**Binary contract.** `/brain/points` is `count * 12` bytes: float32, little-endian, `x y z`
interleaved, in micrometers, in atlas row order. Decode with
`new Float32Array(buffer)` in the browser or `np.frombuffer(body, "<f4").reshape(-1, 3)`.
A highlight is a list of row indices into that same array (the browser never sees root_ids).

**Chat stream.** `message` is trimmed and must be 1 to 2000 characters (else 422).
Send the same `conversation_id` to keep multi-turn memory (the server makes one if you
omit it). Events, in order:

| Event | Data |
|---|---|
| `meta` | `{conversation_id}` (always first) |
| `tool_call` | `{tool, args}` (long argument values are truncated) |
| `highlight` | `{selection_id, total, indices}`; above 5000 neurons `indices` is `null` and `fetch` gives the URL to download them as binary |
| `answer` | `{text}` |
| `error` | `{message}` (model failure, server stopped, hard stop) |
| `done` | `{}` (always last, so a stream never hangs) |

```text
event: meta
data: {"conversation_id": "9f1c..."}

event: tool_call
data: {"tool": "search_neurons", "args": {"nt": "glutamate"}}

event: highlight
data: {"selection_id": "sel_3f9a2c", "total": 1, "indices": [60]}

event: answer
data: {"text": "One glutamate neuron."}

event: done
data: {}
```

## Viewer

![The viewer with left dopamine neurons lit](assets/viewer.png)

<sub>Real data and real highlight (2,926 left dopamine neurons from the backend). The chat text in this screenshot is a mocked stream, because the model provider was overloaded when it was taken.</sub>

A Vite + TypeScript + three.js app in `web/`: all 139,248 neurons as one point cloud (loaded
from the binary `/brain/points` contract, orbit with the mouse) and a chat panel on the right.

```bash
# terminal 1: backend (drop --model for search-only: chips work, chat reports "no model")
uv run --env-file .env flybrain-web --model google_genai:gemini-3.5-flash

# terminal 2: viewer (the dev server proxies /api to 127.0.0.1:8000)
pnpm -C web install && pnpm -C web dev      # open http://localhost:5173
```

The look follows a Penpot design: Inter and JetBrains Mono are self-hosted (`@fontsource`, no CDN),
every color, size and spacing is a custom property in `web/src/style.css`, and selected neurons are
drawn on their own larger, additive layer so they stand out against the dim base cloud. The status
block is real: `GET /health` reports `chat` and `model`, and `stream` is live while a turn runs.

Without a model, run the data and search endpoints only with
`uv run uvicorn --factory flybrain.api:create_app --port 8000`.

- **Chips** (dopamine, dopamine · left, GABA, serotonin, PEN, clear) call
  `GET /neurons/search`, which uses the same search tool and ticket store as the agent but no
  LLM, so they keep working when the model provider is down.
- **Highlights**: when the agent finds neurons, the stream sends a `highlight` event with row
  indices (or a URL to fetch them as binary when the set is large); those points turn white
  and everything else stays grey. `clear` restores the base colors.
- **Look**: all styling lives in `web/src/style.css` as custom properties (colors, fonts, panel
  width, point size and opacity, highlight color); the three.js parameters are read from them
  in `web/src/theme.ts`.
- Tests: `pnpm -C web test` (SSE parser, color buffers, binary decoding, chat stream).

## Roadmap

- [x] Atlas core: positions, types, neurotransmitters
- [x] MCP server with selection tickets
- [x] LangGraph agent + `flybrain-chat` CLI, tested end to end with Gemini
- [x] FastAPI backend: binary brain points + streamed chat (`flybrain-web`)
- [x] Three.js viewer: the brain in 3D, neurons light up as the agent works (`web/`)
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
uv run pytest        # 216 tests, no network or API key needed (fake chat model)
```

Work is tracked per feature in [`odd/tasks/`](odd/tasks/), with test-first commits.

## License

Code: [MIT](LICENSE). FlyWire data is not covered by this license (see [Data](#data)).
