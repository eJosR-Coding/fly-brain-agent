"""stimulate(preset=...): one call resolves a published set and its default readout."""

import asyncio

import pytest

from flybrain import tools
from flybrain.mcp_server import build_server
from test_stimulate import sims, store, ticket  # noqa: F401  (fixtures)

# tiny_atlas: ORN_DA1 root 102 (row 62); EPG root 100 (row 60)
PRESETS = {
    "orn": {"description": "olfactory input", "root_ids": [102], "source": "test", "readout": "epg",
            "meaning": "smell", "validation": "validated"},
    "epg": {"description": "compass neuron", "root_ids": [100], "source": "test", "meaning": "compass"},
    "plain": {"description": "no readout", "root_ids": [102], "source": "test"},
    "ghost": {"description": "absent", "root_ids": [999], "source": "test"},
    "blind": {"description": "readout missing", "root_ids": [102], "source": "test", "readout": "ghost"},
}
KW = {"rate_hz": 300, "duration_ms": 200, "trials": 2}


@pytest.fixture(autouse=True)
def presets(monkeypatch):
    monkeypatch.setattr(tools, "load_presets", lambda: PRESETS)


def call(tiny_atlas, store, sims, selection_id=None, **kw):
    return tools.stimulate(tiny_atlas, store, sims, selection_id, **{**KW, **kw})


def test_preset_equals_the_old_two_step_result(tiny_atlas, store, sims):
    two_step = call(tiny_atlas, store, sims, tools.preset_selection(tiny_atlas, store, "plain")["selection_id"])
    one_step = call(tiny_atlas, store, sims, preset="plain")
    assert one_step["active_neurons"] == two_step["active_neurons"]
    assert one_step["stimulus"] == two_step["stimulus"]
    assert "readout" not in one_step
    assert one_step["preset"] == {"name": "plain", "readout_preset": None}
    assert "meaning" not in one_step["preset"] and "validation" not in one_step["preset"]


def test_preset_reads_out_its_default_readout(tiny_atlas, store, sims):
    out = call(tiny_atlas, store, sims, preset="orn")
    assert out["readout"]["neurons"] == 1
    assert out["preset"] == {"name": "orn", "readout_preset": "epg", "meaning": "smell",
                             "validation": "validated", "readout_meaning": "compass"}
    assert "readout_stats" not in out  # only when silencing
    assert sims.get(out["simulation_id"]).readout["label"] == "EPG"


def test_a_missing_default_readout_degrades_instead_of_aborting(tiny_atlas, store, sims):
    out = call(tiny_atlas, store, sims, preset="blind")
    assert out["active_neurons"] >= 1  # the stimulation still ran
    assert "readout" not in out
    assert out["preset"] == {"name": "blind", "readout_preset": None}
    assert "ghost" in out["note"] and "readout" in out["note"]


def test_default_readout_gets_stats_when_silencing(tiny_atlas, store, sims):
    silence = ticket(tiny_atlas, store, cell_type="T4a")
    out = call(tiny_atlas, store, sims, preset="orn", silence_selection_id=silence)
    assert "readout_stats" in out and "readout_change" in out


def test_explicit_readout_wins(tiny_atlas, store, sims):
    pen = ticket(tiny_atlas, store, cell_type="PEN_a")
    out = call(tiny_atlas, store, sims, preset="orn", readout_selection_id=pen)
    assert out["readout"]["neurons"] == 60
    assert out["preset"]["readout_preset"] is None


def test_both_or_neither_is_a_one_line_error(tiny_atlas, store, sims):
    sid = ticket(tiny_atlas, store, cell_type="ORN_DA1")
    with pytest.raises(ValueError, match="exactly one of selection_id or preset"):
        call(tiny_atlas, store, sims, sid, preset="orn")
    with pytest.raises(ValueError, match="exactly one of selection_id or preset"):
        call(tiny_atlas, store, sims)


def test_unknown_preset_lists_the_available_ones(tiny_atlas, store, sims):
    with pytest.raises(ValueError, match="orn, plain") as e:
        call(tiny_atlas, store, sims, preset="nope")
    assert "\n" not in str(e.value)


def test_preset_absent_from_the_atlas_reports_its_hint(tiny_atlas, store, sims):
    with pytest.raises(ValueError, match="None of its 1 neurons"):
        call(tiny_atlas, store, sims, preset="ghost")


def test_mcp_exposes_preset_and_optional_selection_id(tiny_atlas):
    tool = {t.name: t for t in asyncio.run(build_server(tiny_atlas).list_tools())}["stimulate"]
    props = tool.inputSchema["properties"]
    assert "preset" in props
    assert "selection_id" not in tool.inputSchema.get("required", [])
