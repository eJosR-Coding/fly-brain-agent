# token-budget

Objective: cut the tokens sent per model call and make rate limits visible, without losing the
agent's memory or the tool-use quality.

Problem (measured): every call sent ~3,100 tokens of tool definitions + ~230 of system prompt +
the whole checkpointed conversation including old tool results. A 4-tool question cost 20-40k
tokens; Groq's free tier answered 429 and the SDK retried silently until the 90 s turn limit.

TDD: strict, runner `uv run pytest -q` and `pnpm -C web test`.

## Tasks
- [x] T1 compact the history sent to the model (`compact_history`, `FLYBRAIN_KEEP_TURNS`). Commit 0a075eb.
- [x] T2 concise MCP tool descriptions + pinned budget test. Commit 2a3b2b9.
- [x] T3 fail fast on provider rate limits (`max_retries=1`). Commit fc9ea3a.
- [x] T4 token usage per turn (SSE `usage`, web line). Commit a93736b.

Route: one delegated writer, all four tasks (writer trigger: 2+ non-trivial files per task).

## Evidence
- T1 RED: `ImportError: cannot import name 'MAX_TOOL_CHARS' from 'flybrain.agent'`; GREEN 20 passed (pairing invariant for keep_turns 1..9 via `find_dangling_tool_calls`, kept/dropped counts, idempotence, turn in progress, truncation, env validation, end-to-end: checkpoint keeps all 5 tool results while the model input holds 2).
- T2 RED: `assert 12330 <= 6500`; GREEN at 6,495. Method: `json.dumps({name, description, inputSchema})` per tool from `build_server(...).list_tools()`, summed.
  - Before (b9eb37b): search_neurons 1503, count_neurons 1147, describe_neuron 353, list_selection 680, neighbors 1306, find_path 1887, lesion_effect 1242, top_hubs 1293, preset_selection 735, stimulate 2184 = 12,330.
  - After: 783, 687, 265, 342, 613, 874, 735, 704, 315, 1177 = 6,495 (-47%).
  - Levers: concise docstrings; the auto-generated `title` keys stripped from the input schemas (they repeat the argument name); the four optional filters (cell_type, nt, super_class, side) are now `str = ""` instead of `str | None = None` (empty already meant "no filter"), which drops an `anyOf null` block per filter.
- T3 RED: three tests failed (`init_chat_model` called without `max_retries`); GREEN 422 passed. Verified the kwarg is honoured by the installed langchain-groq 1.1.3 and langchain-google-genai 4.4.3 (`.max_retries == 1` through `init_chat_model`); anthropic and openai use the same name.
- T4 RED: `ImportError: cannot import name 'sum_usage'` (python) and 10 failing vitest cases (`formatTokens`/`formatUsage`/`parseUsage` not exported); GREEN: 433 pytest, 186 vitest, `pnpm -C web build` ok. Existing event-order tests updated to expect `usage` before `answer`.
- Estimate of tokens sent (fake models, history + system prompt only, chars/4, 2 tool calls of ~6,000 chars per question; tool definitions are extra, 12,330 -> 6,495 chars per call):
  - 2 questions, keep_turns=1: 13,592 -> 7,611 tokens over 6 model calls (-44%); last call 4,298 -> 2,305.
  - 5 questions, keep_turns=3 (default): 80,617 -> 62,676 (-23%); last call 10,516 -> 6,529. The gain grows with the conversation: old turns cost only question + answer.

## Decisions
- `usage` is emitted just before `answer`/`error` (answer stays the terminal event of `stream_events`); only when the model was called.
- Usage is read from the finished AIMessage of each agent step (LangChain sums streamed chunk usage into it); chunk usage is only a fallback, so streaming is never double counted.
- A turn that ended without a final answer keeps only its question in the compacted history.

## Not done
- No real-provider run (no real LLM by instruction): the usage line and the 429 fast-fail were exercised with fakes only.
- `--keep-turns` CLI flag not added (env only).

## Next step
Try a real Groq turn and compare the usage line against the provider dashboard.
