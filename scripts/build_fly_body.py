"""Thin wrapper: `uv run python scripts/build_fly_body.py --wheel <flygym-2.1.1-py3-none-any.whl>`."""

from flybrain.fly_body_build import main

if __name__ == "__main__":
    main()
