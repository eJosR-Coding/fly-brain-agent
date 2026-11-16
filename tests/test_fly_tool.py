"""The `fly` tool: a persistent doll whose stimuli run TOGETHER through one scheduled simulation."""

import asyncio
import json
import re

import numpy as np
import pytest

from flybrain import tools
from flybrain.doll import Doll
from flybrain.mcp_server import build_server
from test_stimulate import sims, store, ticket  # noqa: F401  (fixtures)

# tiny_atlas: PEN_a roots 1..60 (rows 0..59), EPG 100 (row 60), T4a 101 (row 61), ORN_DA1 102 (row 62)
# edges: ORN -> T4a -> EPG -> PEN row 0 (excitatory), EPG -> PEN row 1 (inhibitory)
PRESETS = {
    "orn": {"description": "olfactory input", "root_ids": [102], "source": "test", "role": "stimulus",
            "readout": "epg", "meaning": "smell", "validation": "validated"},
    "pen": {"description": "PEN neurons", "root_ids": [1, 2, 3], "source": "test", "role": "stimulus",
            "meaning": "compass input", "validation": "exploratory"},
    "t4": {"description": "T4a", "root_ids": [101], "source": "test", "meaning": "motion"},  # no role = stimulus
    "epg": {"description": "compass neuron", "root_ids": [100], "source": "test", "role": "readout",
            "meaning": "compass", "validation": "validated"},
    "pen0": {"description": "one PEN", "root_ids": [1], "source": "test", "role": "readout",
             "meaning": "heading", "validation": "exploratory"},
    "ghost": {"description": "absent", "root_ids": [999], "source": "test", "role": "stimulus"},
    "blind": {"description": "readout missing", "root_ids": [102], "source": "test", "role": "stimulus",
              "readout": "ghost"},
}
KW = {"duration_ms": 200, "trials": 2}


@pytest.fixture(autouse=True)
def presets(monkeypatch):
    monkeypatch.setattr(tools, "load_presets", lambda: PRESETS)


@pytest.fixture
def doll():
    return Doll()


@pytest.fixture
def schedules(sims, monkeypatch):  # noqa: F811 - records every (schedule, silence_rows) the tool sends
    seen = []
    real = sims.run_schedule

    def spy(schedule, duration_ms=1000.0, trials=1, seed=0, silence_rows=None, bin_ms=None):
        seen.append({"segments": [(s[0], s[1], list(s[2]), s[3]) for s in schedule.segments],
                     "silence": None if silence_rows is None else sorted(int(r) for r in silence_rows),
                     "trials": trials, "duration": duration_ms})
        return real(schedule, duration_ms, trials, seed=seed, silence_rows=silence_rows, bin_ms=bin_ms)

    monkeypatch.setattr(sims, "run_schedule", spy)
    return seen


def fly(tiny_atlas, store, sims, doll, **kw):  # noqa: F811
    return tools.fly(tiny_atlas, store, sims, doll, **{**KW, **kw})


def test_no_stimulus_means_no_run_and_a_hint(tiny_atlas, store, sims, doll, schedules):
    out = fly(tiny_atlas, store, sims, doll)
    assert out["ran"] is False and schedules == []
    assert out["conditions"] == {"stimuli": {}, "silenced": [], "readouts": []}
    assert "fly(stimuli={'sugar': 100})" in out["hint"]
    out = fly(tiny_atlas, store, sims, doll, readouts=["epg"])  # a readout alone is still no stimulus
    assert out["ran"] is False and out["conditions"]["readouts"] == ["epg"]


def test_first_run_reports_conditions_readouts_and_no_change(tiny_atlas, store, sims, doll):
    out = fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    assert out["ran"] is True and out["change_vs_previous"] is None
    assert out["conditions"] == {"stimuli": {"orn": 300.0}, "silenced": [], "readouts": []}
    assert out["active_neurons"] >= 3 and out["simulation_id"].startswith("sim_")
    assert len(out["top_responders"]) <= 5 and len(out["top_non_sensory"]) <= 5
    assert all(r["super_class"] not in ("sensory", "sensory_ascending") for r in out["top_non_sensory"])
    # default readout derived from the stimulus preset: orn -> epg
    (ro,) = out["readouts"]
    assert ro["name"] == "epg" and ro["meaning"] == "compass" and ro["validation"] == "validated"
    assert ro["neurons"] == 1 and ro["mean_hz"] > 0 and ro["max_hz"] >= ro["mean_hz"]
    assert out["stimuli_info"] == [{"name": "orn", "meaning": "smell", "validation": "validated", "rate_hz": 300.0}]
    assert out["cached"] is False and out["runtime_s"] >= 0
    assert "exploratory" not in out and "warning" not in out
    assert tools.SIM_NOTE in out["note"] and "MODEL" in out["note"]
    assert len(json.dumps(out)) < 2500


def test_one_segment_per_stimulus_with_its_own_rate(tiny_atlas, store, sims, doll, schedules):
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300, "pen": 80.5})
    (call,) = schedules
    assert len(call["segments"]) == 2 and call["silence"] is None
    by_rows = {tuple(rows): (start, end, rate) for start, end, rows, rate in call["segments"]}
    assert by_rows[(62,)] == (0.0, 200.0, 300.0)
    assert by_rows[(0, 1, 2)] == (0.0, 200.0, 80.5)


def test_second_call_keeps_the_conditions_and_reports_the_change(tiny_atlas, store, sims, doll, schedules):
    first = fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300}, readouts=["pen0"])
    second = fly(tiny_atlas, store, sims, doll, silence=["epg"])  # cut EPG's output: PEN row 0 loses its drive
    assert second["conditions"] == {"stimuli": {"orn": 300.0}, "silenced": ["epg"], "readouts": ["pen0"]}
    assert len(schedules[1]["segments"]) == 1 and schedules[1]["silence"] == [60]  # epg is row 60
    ch = second["change_vs_previous"]
    assert ch["active_neurons"] == second["active_neurons"] - first["active_neurons"] < 0
    assert ch["conditions_changed"] == "silenced epg"
    by_name = {r["name"]: r for r in ch["readouts"]}
    assert set(by_name) == {"epg", "pen0"}
    assert by_name["pen0"]["before_hz"] > 0 and by_name["pen0"]["after_hz"] == 0
    assert by_name["pen0"]["change_hz"] == pytest.approx(-by_name["pen0"]["before_hz"], abs=0.11)
    assert by_name["epg"]["change_hz"] == pytest.approx(0, abs=0.11)  # silencing cuts outputs, not the neuron itself


def test_adding_a_stimulus_reports_it_in_the_change(tiny_atlas, store, sims, doll):
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    out = fly(tiny_atlas, store, sims, doll, stimuli={"pen": 100})
    assert out["conditions"]["stimuli"] == {"orn": 300.0, "pen": 100.0}
    assert out["change_vs_previous"]["conditions_changed"] == "added pen 100 Hz"


def test_readouts_only_in_one_run_are_left_out_of_the_comparison(tiny_atlas, store, sims, doll):
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    out = fly(tiny_atlas, store, sims, doll, readouts=["pen0"])  # epg (derived) + pen0 (explicit)
    assert {r["name"] for r in out["readouts"]} == {"epg", "pen0"}
    assert [r["name"] for r in out["change_vs_previous"]["readouts"]] == ["epg"]
    assert out["conditions"]["readouts"] == ["pen0"]  # only the explicit ones are listed as conditions


def test_remove_and_reset_then_apply(tiny_atlas, store, sims, doll):
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300, "pen": 100})
    out = fly(tiny_atlas, store, sims, doll, remove=["pen"])
    assert out["conditions"]["stimuli"] == {"orn": 300.0}
    assert out["change_vs_previous"]["conditions_changed"] == "removed pen"
    out = fly(tiny_atlas, store, sims, doll, remove=["orn"])
    assert out["ran"] is False and out["conditions"]["stimuli"] == {}
    out = fly(tiny_atlas, store, sims, doll, reset=True, stimuli={"t4": 300})
    assert out["conditions"] == {"stimuli": {"t4": 300.0}, "silenced": [], "readouts": []}
    assert out["ran"] is True


def test_unknown_remove_is_a_note_not_an_error(tiny_atlas, store, sims, doll):
    out = fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300}, remove=["bitter"])
    assert out["ran"] is True and "bitter" in out["note"]


def test_a_stimulus_with_no_role_is_accepted(tiny_atlas, store, sims, doll):
    assert fly(tiny_atlas, store, sims, doll, stimuli={"t4": 300})["ran"] is True


def test_unknown_or_wrong_role_stimulus_lists_the_valid_ones_and_changes_nothing(tiny_atlas, store, sims, doll):
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    before = doll.conditions()
    for bad in ("nope", "epg"):
        with pytest.raises(ValueError, match="orn") as exc:
            fly(tiny_atlas, store, sims, doll, stimuli={bad: 100})
        assert "epg" not in str(exc.value).split("valid stimulus presets:")[1]
    with pytest.raises(ValueError, match="stimulus presets"):
        fly(tiny_atlas, store, sims, doll, remove=["orn"], stimuli={"nope": 1})
    assert doll.conditions() == before  # nothing was applied, not even the valid parts


def test_readout_with_a_stimulus_role_is_rejected(tiny_atlas, store, sims, doll):
    with pytest.raises(ValueError, match="valid readout presets: .*epg"):
        fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300}, readouts=["pen"])
    assert doll.conditions()["stimuli"] == {}


def test_a_preset_missing_from_the_atlas_is_a_clear_error(tiny_atlas, store, sims, doll):
    with pytest.raises(ValueError, match="None of its"):
        fly(tiny_atlas, store, sims, doll, stimuli={"ghost": 100})
    assert doll.conditions()["stimuli"] == {}


def test_a_missing_default_readout_is_skipped_with_a_note(tiny_atlas, store, sims, doll):
    out = fly(tiny_atlas, store, sims, doll, stimuli={"blind": 300})
    assert out["ran"] is True and out["readouts"] == [] and "ghost" in out["note"]


def test_rates_duration_and_trials_are_clamped_with_notes(tiny_atlas, store, sims, doll, schedules):
    out = tools.fly(tiny_atlas, store, sims, doll, stimuli={"orn": 5000, "pen": 0.1}, duration_ms=5, trials=50)
    assert out["conditions"]["stimuli"] == {"orn": 300.0, "pen": 1.0}
    assert schedules[0]["duration"] == 100.0 and schedules[0]["trials"] == 10
    for text in ("rate_hz capped to 300.0", "rate_hz 0.1 clamped to 1.0", "duration_ms 5.0 clamped to 100.0",
                 "trials capped to 10"):
        assert text in out["note"]


def test_default_trials_and_bad_types(tiny_atlas, store, sims, doll, schedules):
    tools.fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300}, duration_ms=100)
    assert schedules[0]["trials"] == tools.DEFAULT_TRIALS
    for kw in (dict(stimuli={"orn": "fast"}), dict(stimuli={"orn": True}), dict(trials=2.5)):
        with pytest.raises(TypeError):
            fly(tiny_atlas, store, sims, doll, **kw)


def test_seventh_stimulus_is_refused(tiny_atlas, store, sims, doll, monkeypatch):
    many = {f"p{i}": {"description": "d", "root_ids": [102], "source": "t", "role": "stimulus"} for i in range(7)}
    monkeypatch.setattr(tools, "load_presets", lambda: many)
    with pytest.raises(ValueError, match="at most 6"):
        fly(tiny_atlas, store, sims, doll, stimuli={name: 100 for name in many})


def test_exploratory_stimulus_is_flagged(tiny_atlas, store, sims, doll):
    out = fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300, "pen": 100})
    assert out["exploratory"] is True and "exploratory" in out["note"]


def test_runaway_warning_reuses_the_stimulate_text(tiny_atlas, store, sims, doll, monkeypatch):
    monkeypatch.setattr(tools, "RUNAWAY_ACTIVE", 3)
    out = fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    assert out["warning"].startswith("likely runaway excitation") and "artifact" in out["warning"]


def test_the_run_is_recorded_for_the_viewer_with_a_readout_trace(tiny_atlas, store, sims, doll):
    out = fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300, "pen": 100})
    rec = sims.get(out["simulation_id"])
    assert rec.rows.dtype == np.uint32 and rec.frames.dtype == np.uint8 and rec.trials == 2
    assert rec.stimulus["stimuli"] == {"orn": 300.0, "pen": 100.0} and rec.stimulus["duration_ms"] == 200.0
    assert rec.readout["label"] == "EPG" and rec.active_count == out["active_neurons"]


def test_a_repeated_run_is_served_from_the_cache(tiny_atlas, store, sims, doll):
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    assert fly(tiny_atlas, store, sims, doll)["cached"] is True


def test_honesty_wording_in_the_result_and_the_docstring(tiny_atlas, store, sims, doll, monkeypatch):
    monkeypatch.setattr(tools, "RUNAWAY_ACTIVE", 3)
    outs = [fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300, "pen": 100}),
            fly(tiny_atlas, store, sims, doll, readouts=["pen0"]), fly(tiny_atlas, store, sims, doll, reset=True)]
    server = build_server(tiny_atlas, sims=sims)
    description = {t.name: t.description for t in asyncio.run(server.list_tools())}["fly"]
    text = (json.dumps(outs) + description + tools.fly.__doc__).lower()
    for word in ("die", "dead", "death", "harm", "overdose", "kill", "suffer", "pain"):
        assert not re.search(rf"\b{word}\b", text), word
    for word in ("temperature", "hunger", "sleep"):
        assert word not in json.dumps(outs).lower()
    assert "saturat" in text or "artifact" in text


def test_stimulate_output_is_unchanged_by_the_shared_helper(tiny_atlas, store, sims):
    sid = ticket(tiny_atlas, store, cell_type="ORN_DA1")
    out = tools.stimulate(tiny_atlas, store, sims, sid, rate_hz=300, duration_ms=200, trials=2)
    assert list(out) == ["stimulus", "active_neurons", "top_responders", "top_non_sensory", "selection_id",
                         "simulation_id", "runtime_s", "cached", "note"]
    assert out["stimulus"] == {"neurons": 1, "rate_hz": 300.0, "duration_ms": 200.0, "trials": 2}
    assert set(out["top_responders"][0]) == {"root_id", "cell_type", "super_class", "rate_hz"}


# --- over MCP ----------------------------------------------------------------------------


def _call(server, name, args):
    res = asyncio.run(server.call_tool(name, args))
    payload = res[1] if isinstance(res, tuple) else res
    return json.loads(payload[0].text) if isinstance(payload, list) else payload


def test_the_server_registers_fly_and_keeps_one_doll_per_server(tiny_atlas, sims):
    server = build_server(tiny_atlas, sims=sims)
    tool = {t.name: t for t in asyncio.run(server.list_tools())}["fly"]
    assert set(tool.inputSchema["properties"]) == {
        "stimuli", "remove", "silence", "readouts", "reset", "duration_ms", "trials"}
    assert "shared" in tool.description.lower() or "persist" in tool.description.lower()
    first = _call(server, "fly", {"stimuli": {"orn": 300}, "duration_ms": 100, "trials": 1})
    second = _call(server, "fly", {"duration_ms": 100, "trials": 1})  # nothing new: same doll, same stimuli
    assert first["ran"] and second["ran"] and second["conditions"]["stimuli"] == {"orn": 300.0}
    other = build_server(tiny_atlas, sims=sims)  # a new server process starts with an empty doll
    assert _call(other, "fly", {})["ran"] is False


# --- review follow-ups: a failed run must not leave the doll half changed; one change at a time ------------

def test_a_failed_run_leaves_the_doll_as_it_was(tiny_atlas, store, sims, doll, monkeypatch):  # noqa: F811
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    before_conditions, before_last = doll.conditions(), dict(doll.last)

    def boom(*a, **k):
        raise RuntimeError("the simulator went away")

    monkeypatch.setattr(sims, "run_schedule", boom)
    with pytest.raises(RuntimeError):
        fly(tiny_atlas, store, sims, doll, stimuli={"pen": 80}, silence=["epg"])
    assert doll.conditions() == before_conditions
    assert doll.last == before_last


def test_the_doll_is_locked_while_a_change_runs(tiny_atlas, store, sims, doll, monkeypatch):  # noqa: F811
    held = []
    real = sims.run_schedule

    def spy(*a, **k):
        held.append(doll.lock.locked())
        return real(*a, **k)

    monkeypatch.setattr(sims, "run_schedule", spy)
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    assert held == [True]
    assert not doll.lock.locked()  # released afterwards, also after a failure


def test_the_lock_is_released_after_a_failed_run(tiny_atlas, store, sims, doll, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sims, "run_schedule", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    assert not doll.lock.locked()


# --- chat evaluation follow-ups (2026-11-16): reset starts a new baseline, wind is findable ------------------

def test_a_reset_starts_a_new_baseline(tiny_atlas, store, sims, doll):  # noqa: F811
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    out = fly(tiny_atlas, store, sims, doll, reset=True, stimuli={"pen": 80})
    assert out["ran"] is True
    assert out["change_vs_previous"] is None  # nothing before a reset is comparable


def test_a_reset_that_fails_keeps_the_old_baseline(tiny_atlas, store, sims, doll, monkeypatch):  # noqa: F811
    fly(tiny_atlas, store, sims, doll, stimuli={"orn": 300})
    last = dict(doll.last)
    monkeypatch.setattr(sims, "run_schedule", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        fly(tiny_atlas, store, sims, doll, reset=True, stimuli={"pen": 80})
    assert doll.last == last


def test_the_tool_description_names_every_packaged_stimulus_and_points_wind_to_the_antenna(tiny_atlas, store, sims):  # noqa: F811
    import importlib.resources as res

    packaged = json.loads(res.files("flybrain").joinpath("data/presets.json").read_text(encoding="utf-8"))["presets"]
    stimulus_names = [n for n, p in packaged.items() if p.get("role") == "stimulus"]
    server = build_server(tiny_atlas, store, sims)
    desc = next(t.description for t in asyncio.run(server.list_tools()) if t.name == "fly")
    for name in stimulus_names:
        assert name in desc
    assert "wind" in desc.lower()


def test_the_prompt_tells_the_agent_not_to_quote_tool_fields():
    from flybrain.agent import SYSTEM_PROMPT

    assert "field names" in SYSTEM_PROMPT
    assert "change_vs_previous" not in SYSTEM_PROMPT


def test_the_prompt_routes_presets_and_conversational_commands_to_fly_not_stimulate():
    from flybrain.agent import SYSTEM_PROMPT

    low = SYSTEM_PROMPT.lower()
    assert "use fly" in low
    for name in ("sugar", "bitter", "water", "jo_ce"):
        assert name in low  # the agent is told which published presets go to fly
    assert "dale" in low  # the Spanish command form of 'give'
    assert "stimulate(preset" not in low  # the competing instruction is gone
    assert "only for a selection_id" in low


@pytest.mark.parametrize("field", ["remove", "silence", "readouts"])
def test_the_name_lists_are_bounded(tiny_atlas, store, sims, doll, field):  # noqa: F811
    with pytest.raises(ValueError, match=rf"{field}: at most {tools.MAX_DOLL_NAMES} names"):
        fly(tiny_atlas, store, sims, doll, stimuli={"orn": 50}, **{field: ["orn"] * (tools.MAX_DOLL_NAMES + 1)})
    assert doll.conditions() == {"stimuli": {}, "silenced": [], "readouts": []}


@pytest.mark.parametrize("field", ["remove", "silence", "readouts"])
def test_a_name_longer_than_the_limit_is_refused(tiny_atlas, store, sims, doll, field):  # noqa: F811
    with pytest.raises(ValueError, match=rf"{field}: names are at most {tools.MAX_DOLL_NAME_LEN} characters"):
        fly(tiny_atlas, store, sims, doll, stimuli={"orn": 50}, **{field: ["x" * (tools.MAX_DOLL_NAME_LEN + 1)]})
    assert doll.conditions() == {"stimuli": {}, "silenced": [], "readouts": []}


def test_a_stimulus_name_longer_than_the_limit_is_refused(tiny_atlas, store, sims, doll):  # noqa: F811
    with pytest.raises(ValueError, match="stimuli: names are at most"):
        fly(tiny_atlas, store, sims, doll, stimuli={"x" * (tools.MAX_DOLL_NAME_LEN + 1): 50})
