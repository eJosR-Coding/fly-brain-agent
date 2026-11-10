"""Thin MCP layer over flybrain.tools. stdout belongs to the protocol: never print."""

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from flybrain import tools
from flybrain.atlas import Atlas, AtlasLoadError, load_atlas
from flybrain.selections import SelectionStore

DEFAULT_ANNOTATIONS = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "flywire_neuron_annotations.tsv"
)


def build_server(atlas: Atlas, store: SelectionStore | None = None) -> FastMCP:
    store = store if store is not None else SelectionStore()
    server = FastMCP("flybrain")

    @server.tool()
    def search_neurons(
        cell_type: str | None = None,
        nt: str | None = None,
        super_class: str | None = None,
        side: str | None = None,
    ) -> dict:
        """Search the fruit-fly brain (FlyWire, ~139k neurons). Filters are AND-ed; all optional.

        Use this FIRST to find neurons. It never returns the full list: matches are stored
        server-side and you get a small summary (total, 5 sample neurons, counts per
        super_class) plus a `selection_id` ticket (e.g. "sel_3f9a2c") that later tools act on.
        With no filters the ticket is the whole brain. An empty string for a filter means
        "no filter". Filter values are trimmed and limited to 200 characters. A no-match
        answer lists the valid values when a closed-set filter (nt, super_class, side) is off.

        Args:
            cell_type: prefix match, case-insensitive (e.g. "PEN" matches PEN_a, PEN_b).
            nt: neurotransmitter, exact: acetylcholine, glutamate, gaba, dopamine,
                serotonin, octopamine.
            super_class: e.g. optic, central, sensory, visual_projection, ascending,
                descending, motor, endocrine.
            side: left, right, center or na (na = side not annotated).
        """
        return tools.search_neurons(atlas, store, cell_type, nt, super_class, side)

    @server.tool()
    def count_neurons(
        group_by: str,
        cell_type: str | None = None,
        nt: str | None = None,
        super_class: str | None = None,
        side: str | None = None,
        top: int = 20,
    ) -> dict:
        """Count neurons grouped by a field, optionally filtered. Cheap; makes no ticket.

        Use it to explore ("how many dopamine neurons per super_class?") or to find valid
        values before a search. Returns the top N groups and `other` (sum of the rest).

        Args:
            group_by: one of super_class, cell_class, cell_type, top_nt, side, flow.
            cell_type, nt, super_class, side: same filters as search_neurons.
            top: how many groups to show (default 20, capped at 100; a `note` says so).
        """
        return tools.count_neurons(atlas, group_by, cell_type, nt, super_class, side, top)

    @server.tool()
    def describe_neuron(root_id: int) -> dict:
        """Full details of ONE neuron by its FlyWire root_id: position in micrometers,
        super_class, cell_class, cell_type, neurotransmitter, side, flow."""
        return tools.describe_neuron(atlas, root_id)

    @server.tool()
    def list_selection(selection_id: str, offset: int = 0, limit: int = 20) -> dict:
        """Page through the neurons behind a selection_id from search_neurons.

        Only use this when you really need individual neurons; limit is capped at 50 per
        call. Out-of-range offset/limit are clamped, never an error: a `note` field in the
        answer says what changed. Prefer search_neurons/count_neurons summaries for big selections.
        """
        return tools.list_selection(atlas, store, selection_id, offset, limit)

    return server


def main() -> None:
    path = Path(os.environ.get("FLYBRAIN_ANNOTATIONS", DEFAULT_ANNOTATIONS))
    try:
        atlas = load_atlas(path)
        store = SelectionStore()
    except (AtlasLoadError, ValueError) as exc:
        # stdout is the MCP channel: one plain line on stderr, then a clean exit code
        print(f"flybrain-mcp: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    build_server(atlas, store).run(transport="stdio")


if __name__ == "__main__":
    main()
