"""flybrain-chat: ask the fly-brain agent one question from the terminal."""

import argparse
import asyncio
import os
import sys
from contextlib import AsyncExitStack

from langchain.chat_models import init_chat_model

from flybrain.agent import MAX_STEPS, ask, build_graph
from flybrain.agent_mcp import ServerStoppedError, mcp_tools

DEFAULT_MODEL = "anthropic:claude-sonnet-5-5"


class CliError(Exception):
    """A failure with its final, user-facing one-line message."""


def _model_spec(value: str) -> str:
    provider, sep, name = value.partition(":")
    if not sep or not provider.strip() or not name.strip():
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid model; use provider:model (e.g. anthropic:claude-sonnet-5-5)"
        )
    return f"{provider.strip()}:{name.strip()}"


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def _one_line(exc: BaseException) -> str:
    return " ".join(str(exc).split()) or type(exc).__name__


def _unwrap(exc: BaseException, wanted: type | None = None) -> BaseException:
    """Dig through (nested) exception groups; prefer a `wanted` member if there is one."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        pick = next((e for e in exc.exceptions if wanted and isinstance(e, wanted)), None)
        exc = pick or exc.exceptions[0]
    return exc


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="flybrain-chat", description="Ask the fly-brain agent.")
    parser.add_argument("question")
    parser.add_argument(
        "--model",
        type=_model_spec,
        default=os.environ.get("FLYBRAIN_MODEL", DEFAULT_MODEL),
        help="provider:model (default: $FLYBRAIN_MODEL or %(default)s)",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=MAX_STEPS,
        help="max tool rounds before a hard stop; 1 allows one tool round plus the answer "
        "(default: %(default)s)",
    )
    return parser.parse_args(argv)


async def run(args) -> str:
    """Run the question; every failure class is turned into its own CliError message."""
    try:
        model = init_chat_model(args.model)
    except Exception as exc:
        raise CliError(
            f"could not set up model {args.model!r}: {_one_line(_unwrap(exc))}\n"
            "Check the provider package (uv add langchain-<provider>) and its API key env var."
        ) from None
    async with AsyncExitStack() as stack:
        try:
            tools = await stack.enter_async_context(mcp_tools(os.environ.get("FLYBRAIN_ANNOTATIONS")))
        except Exception as exc:
            # the server's own stderr line (printed just above) has the real cause
            raise CliError(
                f"flybrain server failed to start: {_one_line(_unwrap(exc))} "
                "(see the server message above, if any)"
            ) from None
        try:
            return await ask(build_graph(model, tools), args.question, args.max_steps)
        except ServerStoppedError as exc:
            raise CliError(str(exc)) from None
        except Exception as exc:
            raise CliError(f"model call failed: {_one_line(_unwrap(exc))}") from None


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        print(asyncio.run(run(args)))
    except BaseException as exc:
        # the MCP session's TaskGroup can wrap our CliError in an ExceptionGroup on the way out
        err = _unwrap(exc, CliError)
        if not isinstance(err, CliError):
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            err = CliError(f"unexpected error: {_one_line(err)}")
        print(f"flybrain-chat: {err}", file=sys.stderr)
        return 1
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
