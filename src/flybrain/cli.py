"""flybrain-chat: ask the fly-brain agent one question from the terminal."""

import argparse
import asyncio
import os
import sys

from langchain.chat_models import init_chat_model

from flybrain.agent import MAX_STEPS, ask, build_graph
from flybrain.agent_mcp import mcp_tools

DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="flybrain-chat", description="Ask the fly-brain agent.")
    parser.add_argument("question")
    parser.add_argument(
        "--model",
        default=os.environ.get("FLYBRAIN_MODEL", DEFAULT_MODEL),
        help="provider:model (default: $FLYBRAIN_MODEL or %(default)s)",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    return parser.parse_args(argv)


async def run(args) -> str:
    model = init_chat_model(args.model)
    async with mcp_tools(os.environ.get("FLYBRAIN_ANNOTATIONS")) as tools:
        return await ask(build_graph(model, tools), args.question, args.max_steps)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        print(asyncio.run(run(args)))
    except Exception as exc:  # friendly message, no traceback dump (missing pkg/key, bad model)
        while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            exc = exc.exceptions[0]  # TaskGroup from the MCP session wraps the real cause
        print(
            f"flybrain-chat: could not use model {args.model!r}: {exc}\n"
            "Check the provider package (uv add langchain-<provider>) and its API key env var.",
            file=sys.stderr,
        )
        return 1
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
