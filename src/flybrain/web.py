"""flybrain-web: run the HTTP backend (brain data + streamed chat) with uvicorn."""

import argparse
import os
import sys

from flybrain.agent import MAX_STEPS
from flybrain.cli import _model_spec, _one_line, _positive_int, _unwrap
from flybrain.models import CHAT_MODEL_KWARGS, PROVIDER_KEYS, parse_model_list, startup_registry


def _model_list(value: str) -> str:
    try:
        return ",".join(parse_model_list(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="flybrain-web", description="Serve the fly-brain API (brain points + chat over SSE)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: %(default)s)")
    parser.add_argument(
        "--model",
        type=_model_spec,
        default=os.environ.get("FLYBRAIN_MODEL") or None,
        help="provider:model the chat starts on (default: $FLYBRAIN_MODEL, else the first "
        "available catalog model, cheapest first)",
    )
    parser.add_argument(
        "--models",
        type=_model_list,
        default=os.environ.get("FLYBRAIN_MODELS") or None,
        help="comma-separated provider:model ids offered in the picker "
        "(default: $FLYBRAIN_MODELS, else the built-in catalog of tool-calling models "
        "for every provider whose API key is set, checked at startup against the provider's "
        "model list so retired ids are hidden); each is usable only if its key is set",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=MAX_STEPS,
        help="max tool rounds per chat turn (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # heavy imports only after --help had its chance: --help needs no model or key
    import uvicorn
    from langchain.chat_models import init_chat_model

    from flybrain.api import create_app

    # one discovery pass at startup (never per request); it cannot crash: failed fetches fail open
    default, models, served = startup_registry(args.model, args.models, os.environ)
    if default is None:
        keys = ", ".join(PROVIDER_KEYS.values())
        print(
            "flybrain-web: no chat model available: no provider API key is set.\n"
            f"Set one of {keys} in .env, or pass --model provider:model.",
            file=sys.stderr,
        )
        return 1
    try:
        model = init_chat_model(default, **CHAT_MODEL_KWARGS)
    except Exception as exc:
        print(
            f"flybrain-web: could not set up model {default!r}: {_one_line(_unwrap(exc))}\n"
            "Check the provider package (uv add langchain-<provider>) and its API key env var.",
            file=sys.stderr,
        )
        return 1
    try:
        app = create_app(model=model, max_steps=args.max_steps, model_name=default,
                         models=models, served=served)
    except ValueError as exc:  # e.g. a bad FLYBRAIN_TURN_TIMEOUT
        print(f"flybrain-web: {_one_line(exc)}", file=sys.stderr)
        return 1
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
