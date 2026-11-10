"""Thin wrapper: `uv run python scripts/build_connectome.py --source <feather>`."""

from flybrain.connectome_build import main

if __name__ == "__main__":
    main()
