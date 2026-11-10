"""flybrain-web: run the HTTP backend (brain data + streamed chat) with uvicorn."""

import argparse
import os
import sys

from flybrain.agent import MAX_STEPS
from flybrain.cli import DEFAULT_MODEL, _model_spec, _one_line, _positive_int, _unwrap


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="flybrain-web", description="Serve the fly-brain API (brain points + chat over SSE)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: %(default)s)")
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
        help="max tool rounds per chat turn (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # heavy imports only after --help had its chance: --help needs no model or key
    import uvicorn
    from langchain.chat_models import init_chat_model

    from flybrain.api import create_app

    try:
        model = init_chat_model(args.model)
    except Exception as exc:
        print(
            f"flybrain-web: could not set up model {args.model!r}: {_one_line(_unwrap(exc))}\n"
            "Check the provider package (uv add langchain-<provider>) and its API key env var.",
            file=sys.stderr,
        )
        return 1
    uvicorn.run(create_app(model=model, max_steps=args.max_steps), host=args.host, port=args.port)
    return 0


def entrypoint() -> None:
    raise SystemExit(main())
