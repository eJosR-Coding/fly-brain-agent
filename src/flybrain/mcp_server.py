"""Thin MCP layer over flybrain.tools. stdout belongs to the protocol: never print."""

import asyncio
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from flybrain import tools
from flybrain.atlas import Atlas, AtlasLoadError, load_atlas
from flybrain.connectome import Connectome, ConnectomeError
from flybrain.selections import SelectionStore
from flybrain.sim_service import SimulationService

DEFAULT_ANNOTATIONS = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "flywire_neuron_annotations.tsv"
)
DEFAULT_CONNECTOME = Path(__file__).resolve().parents[2] / "data" / "processed" / "connectome_783.npz"


def connectome_path() -> Path:
    return Path(os.environ.get("FLYBRAIN_CONNECTOME", DEFAULT_CONNECTOME))


def load_connectome(atlas: Atlas, path=None) -> Connectome | None:
    """The connectome is optional: missing or mismatched file -> None (one stderr line if mismatched)."""
    path = Path(path) if path is not None else connectome_path()
    try:
        return Connectome.load(path, len(atlas.neurons))
    except ConnectomeError as exc:
        if path.is_file():  # present but unusable: say so; a missing file is the normal "not built" case
            print(f"flybrain: connectome disabled: {exc}", file=sys.stderr)
        return None


def _strip_titles(schema):
    """Drop the auto-generated "title" noise from a tool's JSON schema (it repeats the name)."""
    if isinstance(schema, dict):
        return {
            k: _strip_titles(v) for k, v in schema.items()
            if not (k == "title" and isinstance(v, str))  # a property NAMED "title" is a dict, kept
        }
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def build_server(
    atlas: Atlas, store: SelectionStore | None = None, conn: Connectome | None = None,
    sims: SimulationService | None = None,
) -> FastMCP:
    store = store if store is not None else SelectionStore()
    sims = sims if sims is not None else SimulationService.default(atlas)
    server = FastMCP("flybrain")

    @server.tool()
    def search_neurons(
        cell_type: str = "",
        nt: str = "",
        super_class: str = "",
        side: str = "",
    ) -> dict:
        """Find neurons (FlyWire, ~139k). Use FIRST. Filters AND-ed; empty = no filter.

        Returns total, 5 samples, counts per super_class and a `selection_id` ticket (no filters =
        whole brain). No match lists valid values.

        Args:
            cell_type: prefix, case-insensitive ("PEN" -> PEN_a).
            nt: acetylcholine|glutamate|gaba|dopamine|serotonin|octopamine.
            super_class: optic|central|sensory|visual_projection|ascending|descending|motor|endocrine.
            side: left|right|center|na (na = not annotated).
        """
        return tools.search_neurons(atlas, store, cell_type, nt, super_class, side)

    @server.tool()
    def count_neurons(
        group_by: str,
        cell_type: str = "",
        nt: str = "",
        super_class: str = "",
        side: str = "",
        top: int = 20,
    ) -> dict:
        """Count neurons grouped by a field; cheap, no ticket. Also finds valid filter values.

        Returns top groups plus `other`.
        Args:
            group_by: super_class|cell_class|cell_type|top_nt|side|flow.
            cell_type, nt, super_class, side: filters as in search_neurons.
            top: groups shown (default 20, max 100).
        """
        return tools.count_neurons(atlas, group_by, cell_type, nt, super_class, side, top)

    @server.tool()
    def describe_neuron(root_id: int) -> dict:
        """One neuron by FlyWire root_id: position (um), super_class, cell_class, cell_type, neurotransmitter, side, flow."""
        return tools.describe_neuron(atlas, root_id)

    @server.tool()
    def list_selection(selection_id: str, offset: int = 0, limit: int = 20) -> dict:
        """Page a selection's neurons (id, type, class, side); only for individuals. limit 20, max 25."""
        return tools.list_selection(atlas, store, selection_id, offset, limit)

    @server.tool()
    def neighbors(selection_id: str, direction: str = "out", top: int = 10) -> dict:
        """Partners wired to a group, STRUCTURAL connectome (not signal flow); pairs need >= 5 synapses.

        Returns top partners (root_id, cell_type, synapses), counts by super_class/cell_type and a
        NEW `selection_id` of ALL partners (inputs excluded).

        Args:
            selection_id: ticket.
            direction: out (targets), in (inputs) or both.
            top: default 10, max 50.
        """
        return tools.neighbors(atlas, store, conn, selection_id, direction, top)

    @server.tool()
    def find_path(from_selection: str, to_selection: str, max_hops: int = 8) -> dict:
        """Strongest of the shortest DIRECTED structural paths between two groups (pairs >= 5 synapses).

        Returns `hops`, `path` [{root_id, cell_type, super_class, synapses_from_previous}],
        `min_hop_synapses` (bottleneck = weakest hop; 5 barely, 50+ solid),
        `total_synapses`, `shortest_path_count` (large = redundant), up to 3
        `alternative_routes`, and a `selection_id`. Ties: strongest weakest hop. found=false if nothing within max_hops. Synapses proxy
        weight, not sign: wiring, not proof of signal flow. Swap groups to reverse.

        Args:
            from_selection, to_selection: tickets.
            max_hops: default 8, max 12.
        """
        return tools.find_path(atlas, store, conn, from_selection, to_selection, max_hops)

    @server.tool()
    def lesion_effect(
        source_selection: str, target_selection: str, removed_selection: str, max_hops: int = 8
    ) -> dict:
        """Virtual lesion on the STRUCTURAL graph: targets reachable from the source within max_hops (pairs >= 5 synapses), before vs after removing a group.

        Returns counts, `lost`, `lost_share`, a `selection_id` of lost targets.
        Reachability only, no activity: "lost" = no route remains.

        Args:
            source_selection, target_selection, removed_selection: tickets.
            max_hops: default 8, max 12.
        """
        return tools.lesion_effect(atlas, store, conn, source_selection, target_selection, removed_selection, max_hops)

    @server.tool()
    def top_hubs(
        metric: str = "total",
        top: int = 10,
        cell_type: str = "",
        nt: str = "",
        super_class: str = "",
        side: str = "",
    ) -> dict:
        """Most connected neurons in the STRUCTURAL connectome (pairs >= 5 synapses), by summed synapses.

        Returns top (root_id, cell_type, synapses) + `selection_id`. Volume, not importance.

        Args:
            metric: in|out|total (default).
            top: default 10, max 50.
            cell_type, nt, super_class, side: candidate filters as in search_neurons.
        """
        return tools.top_hubs(atlas, store, conn, metric, top, cell_type, nt, super_class, side)

    @server.tool()
    def preset_selection(name: str) -> dict:
        """Ticket for a published set (Shiu et al. 2024): "sugar" = sugar-sensing gustatory neurons, "mn9" = MN9 proboscis motor neuron. Unknown name lists presets.
        """
        return tools.preset_selection(atlas, store, name)

    @server.tool()
    async def stimulate(
        selection_id: str,
        rate_hz: float = 100,
        duration_ms: float = 500,
        trials: int = 3,
        silence_selection_id: str | None = None,
        readout_selection_id: str | None = None,
    ) -> dict:
        """Whole-brain SIMULATION: Poisson-drive a group, see who responds (LIF model, Shiu et al. 2024). A MODEL, not the living fly.

        Returns active count (> 1 Hz), top 10 responders, `top_non_sensory` (silencing candidates), readout
        rates, a NEW `selection_id` of active neurons and a `simulation_id`. Silencing adds a baseline run (2x).
        Slow (~5 s/sim-s/trial): run once.

        Args:
            selection_id: neurons to stimulate.
            rate_hz: 1..300 (default 100).
            duration_ms: 100..1000 (default 500).
            trials: averaged, 1..10 (default 3).
            silence_selection_id: optional; fire but pass nothing on.
            readout_selection_id: optional; rates to report.
        Out-of-range values are capped.
        """
        return await asyncio.to_thread(
            tools.stimulate, atlas, store, sims, selection_id, rate_hz, duration_ms, trials,
            silence_selection_id, readout_selection_id,
        )

    # the tool definitions are sent on EVERY model call: keep them lean (see the budget test)
    for registered in server._tool_manager.list_tools():
        registered.parameters = _strip_titles(registered.parameters)
    return server


def main() -> None:
    path = Path(os.environ.get("FLYBRAIN_ANNOTATIONS", DEFAULT_ANNOTATIONS))
    try:
        atlas = load_atlas(path)
        store = SelectionStore()
        conn = load_connectome(atlas)
    except (AtlasLoadError, ValueError) as exc:
        # stdout is the MCP channel: one plain line on stderr, then a clean exit code
        print(f"flybrain-mcp: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    build_server(atlas, store, conn).run(transport="stdio")


if __name__ == "__main__":
    main()
