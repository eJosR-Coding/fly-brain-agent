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


def build_server(
    atlas: Atlas, store: SelectionStore | None = None, conn: Connectome | None = None,
    sims: SimulationService | None = None,
) -> FastMCP:
    store = store if store is not None else SelectionStore()
    sims = sims if sims is not None else SimulationService.default(atlas)
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

    @server.tool()
    def neighbors(selection_id: str, direction: str = "out", top: int = 10) -> dict:
        """Who is wired to a group of neurons? STRUCTURAL connectome (FlyWire synapses), not signal flow.

        Takes a `selection_id` from search_neurons and lists the partner neurons, biggest total
        synapse count first, excluding the input neurons. You get the top partners (root_id,
        cell_type, synapses), counts by super_class / cell_type, and a NEW `selection_id`
        holding ALL partners (the viewer highlights it; pass it to other tools).
        Only connections of at least 5 synapses between a pair count: single or double synapses
        are mostly segmentation noise, so the default threshold filters them out. A `note` states it.
        Example: neighbors(selection_id=<dopamine PAM cells>, direction="out").

        Args:
            selection_id: ticket from search_neurons or another tool.
            direction: "out" = neurons the group sends synapses to, "in" = neurons sending to it,
                "both" = either way.
            top: partners to list (default 10, capped at 50; a `note` says so).
        """
        return tools.neighbors(atlas, store, conn, selection_id, direction, top)

    @server.tool()
    def find_path(from_selection: str, to_selection: str, max_hops: int = 8) -> dict:
        """Strongest of the shortest DIRECTED structural paths between two neuron groups.

        Finds the fewest synapse hops from any neuron in `from_selection` to any neuron in
        `to_selection` (connections of at least 5 synapses only, the noise filter). Among all
        routes with that hop count it returns the strongest one: the route whose weakest hop
        has the most synapses (ties: most synapses in total). Fields: `hops`; `path`, a short
        list of {root_id, cell_type, super_class, synapses_from_previous} in order (null for
        the first neuron); `min_hop_synapses`, the bottleneck, i.e. the weakest hop on the
        route (a route is only as strong as that hop; 5 = barely connected, 50+ = solid);
        `total_synapses`; `shortest_path_count`, how many distinct routes of that same length
        exist (large = redundant wiring; capped at 1,000,000 with shortest_path_count_capped);
        `alternative_routes`, up to 3 other equally short routes as {chain of cell types,
        min_hop_synapses}, strongest first; and a `selection_id` for the chosen path neurons.
        If nothing is within `max_hops` you get found=false and a hint. Strength is synapse
        count, a proxy for connection weight, NOT sign or efficacy; a path is wiring that
        exists, not proof that a signal travels it. Direction matters: swap the groups to ask
        the reverse. Example: from photoreceptors (cell_type R1-6) to Kenyon cells (KC).

        Args:
            from_selection, to_selection: tickets from search_neurons or another tool.
            max_hops: search depth (default 8, capped at 12; a `note` says so).
        """
        return tools.find_path(atlas, store, conn, from_selection, to_selection, max_hops)

    @server.tool()
    def lesion_effect(
        source_selection: str, target_selection: str, removed_selection: str, max_hops: int = 8
    ) -> dict:
        """Virtual lesion on the STRUCTURAL graph: what stays reachable if a group is removed?

        Counts how many neurons of `target_selection` can be reached from `source_selection`
        within `max_hops` (connections of at least 5 synapses), before and after deleting every
        neuron of `removed_selection`. Returns the counts, `lost` and `lost_share`, a few lost
        examples, and a `selection_id` of the lost targets (highlight it). Pure graph
        reachability: it does NOT simulate activity, so "lost" means "no structural route
        remains", not "the behaviour stops". Run find_path first if unsure the targets are
        reachable at all.

        Args:
            source_selection, target_selection, removed_selection: tickets.
            max_hops: reachability depth (default 8, capped at 12).
        """
        return tools.lesion_effect(atlas, store, conn, source_selection, target_selection, removed_selection, max_hops)

    @server.tool()
    def top_hubs(
        metric: str = "total",
        top: int = 10,
        cell_type: str | None = None,
        nt: str | None = None,
        super_class: str | None = None,
        side: str | None = None,
    ) -> dict:
        """The most connected neurons in the STRUCTURAL connectome (>= 5 synapses per pair).

        Ranks neurons by summed synapse count and returns the top ones (root_id, cell_type,
        super_class, synapses) plus a `selection_id` ticket for them. Optional filters (same as
        search_neurons) restrict the candidates, e.g. the biggest hubs among Kenyon cells.
        Synapse count measures wiring volume, not importance or influence.

        Args:
            metric: "in" (synapses received), "out" (synapses sent) or "total" (default).
            top: how many to list (default 10, capped at 50).
            cell_type, nt, super_class, side: optional candidate filters, as in search_neurons.
        """
        return tools.top_hubs(atlas, store, conn, metric, top, cell_type, nt, super_class, side)

    @server.tool()
    def preset_selection(name: str) -> dict:
        """A ticket for a published, named neuron set, ready for stimulate or the other tools.

        Presets come from the paper that the simulator reproduces (Shiu et al., Nature 2024):
        "sugar" = the 21 sugar-sensing gustatory receptor neurons of its example (20 exist in
        this atlas version), "mn9" = MN9, the motor neuron it reads out (proboscis extension).
        Returns a ticket (`selection_id`), counts, and the `source` of the set. An unknown name
        lists the available presets.

        Args:
            name: preset name, e.g. "sugar" or "mn9".
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
        """Run a whole-brain SIMULATION: drive a group of neurons and see which others respond.

        Poisson-drives the neurons of `selection_id` at `rate_hz` for `duration_ms` in a leaky
        integrate-and-fire model of the whole fly brain (Shiu et al., Nature 2024 parameters,
        validated against their code). Returns how many neurons were active (> 1 Hz), the top 10
        responders (root_id, cell_type, super_class, rate_hz), the readout group's rates if
        `readout_selection_id` is given, a NEW `selection_id` of all active neurons (the viewer
        highlights it) and a `simulation_id` (the viewer plays the activity as an animation).
        With `silence_selection_id` those neurons still fire but pass nothing on; a baseline run
        is added to report the change in the readout (it costs 2x the time). Example: give sugar
        (preset "sugar") and read out MN9 (preset "mn9"). Takes seconds (about 5 s per simulated
        second per trial, one simulation at a time). It is a MODEL of the wiring diagram, not the
        living fly: no neuromodulation, plasticity or nerve cord.

        Args:
            selection_id: ticket of the neurons to stimulate (e.g. from preset_selection).
            rate_hz: Poisson input rate per stimulated neuron, 1..300 (default 100; capped, a `note` says so).
            duration_ms: simulated time, 100..1000 (default 500; capped).
            trials: independent repeats averaged, 1..10 (default 3; capped).
            silence_selection_id: optional ticket of neurons to silence.
            readout_selection_id: optional ticket of neurons whose rates to report.
        """
        return await asyncio.to_thread(
            tools.stimulate, atlas, store, sims, selection_id, rate_hz, duration_ms, trials,
            silence_selection_id, readout_selection_id,
        )

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
