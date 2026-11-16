"""State-machine fixes in the simulation layer: selection store races, experiment progress, record pairs, load state."""

import threading

import numpy as np
import pytest

from flybrain import experiments, silencing, tools
from flybrain.selections import SelectionStore, UnknownSelectionError
from flybrain.sim_service import SimulationService, SimulationUnavailable
from flybrain.simulation import Simulation

# --- S17: SelectionStore under concurrent add/get ----------------------------------------


def test_get_survives_a_competing_add_between_the_check_and_the_move():
    """Deterministic interleaving: another thread's add() (which evicts the wanted id) runs right after
    get's membership test. Unlocked, get raises a raw KeyError; locked, the add waits and get succeeds."""
    store = SelectionStore(max_items=2)
    wanted = store.add([1], "a")
    store.add([2], "b")
    items = store._items

    class Racy(type(items)):
        armed = True

        def __contains__(self, key):
            hit = super().__contains__(key)
            if hit and key == wanted and Racy.armed:
                Racy.armed = False
                t = threading.Thread(target=store.add, args=([3], "c"))
                t.start()
                t.join(0.3)  # unlocked: finishes (and evicts `wanted`); locked: still waiting for get
            return hit

    store._items = Racy(items)
    assert store.get(wanted).id == wanted


def test_get_never_raises_a_raw_keyerror_under_threaded_add_and_get():
    store = SelectionStore(max_items=3)
    ids = [store.add([i], f"g{i}") for i in range(3)]
    errors, stop = [], threading.Event()

    def adder():
        while not stop.is_set():
            store.add([9], "x")

    def getter():
        for _ in range(3000):
            for sid in ids:
                try:
                    store.get(sid)
                except UnknownSelectionError:
                    pass  # evicted or expired: typed, fine
                except Exception as exc:  # noqa: BLE001 - the thing under test
                    errors.append(exc)
                    return

    threads = [threading.Thread(target=adder) for _ in range(2)] + [threading.Thread(target=getter) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads[2:]:
        t.join()
    stop.set()
    for t in threads[:2]:
        t.join()
    assert errors == []


def test_unknown_listing_does_not_break_while_others_add():
    store = SelectionStore(max_items=5)
    errors, stop = [], threading.Event()

    def adder():
        while not stop.is_set():
            store.add([1], "x")

    t = threading.Thread(target=adder)
    t.start()
    for _ in range(2000):
        try:
            store.get("sel_nope00")
        except UnknownSelectionError:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            break
    stop.set()
    t.join()
    assert errors == []


def test_valid_and_expired_paths_keep_their_messages():
    store = SelectionStore(max_items=1)
    a = store.add([1], "a", {"cell_type": "EPG"})
    store.add([2], "b")
    with pytest.raises(UnknownSelectionError, match="expired.*search_neurons"):
        store.get(a)
    with pytest.raises(UnknownSelectionError, match="unknown selection"):
        store.get("sel_zzzzzz")


# --- helpers for the simulation-layer tests ------------------------------------------------

EDGES = [(62, 61, 1000), (61, 60, 1000), (62, 5, 1000), (60, 0, 1000)]


def _sim(atlas):
    pre, post, w = (np.array(c) for c in zip(*EDGES))
    return Simulation.from_arrays(pre, post, w, n_neurons=len(atlas.neurons))


@pytest.fixture
def lazy(tiny_atlas):
    """A service that has NOT loaded yet; `seen` records progress_snapshot() at load and at every run."""
    seen = {"load": [], "run": []}
    sim = _sim(tiny_atlas)

    def loader():
        seen["load"].append(svc.progress_snapshot())
        return sim

    svc = SimulationService(loader)
    real_run = svc.run

    def run(*a, **kw):
        seen["run"].append(svc.progress_snapshot())
        return real_run(*a, **kw)

    svc.run = run
    return svc, seen


@pytest.fixture
def store():
    return SelectionStore()


def _ticket(tiny_atlas, store, **filters):
    return tools.search_neurons(tiny_atlas, store, **filters)["selection_id"]


# --- S18: progress covers the whole experiment --------------------------------------------


def test_silencing_progress_is_running_during_load_and_baseline_and_cleared_after(tiny_atlas, store, lazy):
    svc, seen = lazy
    orn, epg = _ticket(tiny_atlas, store, cell_type="ORN_DA1"), _ticket(tiny_atlas, store, cell_type="EPG")
    events = []
    svc.subscribe(lambda p: events.append((p["done"], p["total"])))
    silencing.silencing_screen(tiny_atlas, store, svc, orn, epg, rate_hz=300, duration_ms=300, trials=2, n=50)
    assert seen["load"][0] is not None and seen["load"][0]["experiment"] == "silencing_screen"
    assert seen["run"][0] is not None  # the baseline run
    assert events[0][0] == 0 and events[-1][0] == events[-1][1]  # ends complete
    assert events[-1] == (4, 4)  # 3 candidates + the baseline
    assert svc.progress_snapshot() is None


def test_silencing_progress_is_cleared_when_the_load_fails(tiny_atlas, store):
    orn, epg = _ticket(tiny_atlas, store, cell_type="ORN_DA1"), _ticket(tiny_atlas, store, cell_type="EPG")

    def loader():
        raise SimulationUnavailable("no weights")

    svc = SimulationService(loader)
    with pytest.raises(SimulationUnavailable):
        silencing.silencing_screen(tiny_atlas, store, svc, orn, epg)
    assert svc.progress_snapshot() is None


def test_dose_response_progress_is_running_during_load(tiny_atlas, store, lazy):
    svc, seen = lazy
    orn, epg = _ticket(tiny_atlas, store, cell_type="ORN_DA1"), _ticket(tiny_atlas, store, cell_type="EPG")
    experiments.dose_response(tiny_atlas, store, svc, orn, epg, rates_hz=[100, 200], duration_ms=100, trials=1)
    assert seen["load"][0] == {"experiment": "dose_response", "done": 0, "total": 2}
    assert svc.progress_snapshot() is None


# --- S19: baseline/silenced pair stored atomically -----------------------------------------


def _rec(svc, **kw):
    return dict(rows=[1, 2], frames=np.ones((2, 3), dtype=np.uint8), bin_ms=1.0, duration_ms=3.0, trials=1,
                stimulus={}, **kw)


def test_add_pair_returns_two_ids_and_links_them(tiny_atlas):
    svc = SimulationService.from_simulation(_sim(tiny_atlas), max_results=4)
    base_id, run_id = svc.add_pair(_rec(svc), _rec(svc, silenced_rows=[1]))
    assert svc.get(run_id).compare_id == base_id and svc.get(base_id).compare_id is None


def test_pairs_never_dangle_however_many_are_added(tiny_atlas):
    svc = SimulationService.from_simulation(_sim(tiny_atlas), max_results=3)
    for _ in range(40):
        svc.add_pair(_rec(svc), _rec(svc, silenced_rows=[1]))
        svc.add(**_rec(svc))
    for r in svc._records.values():
        assert r.compare_id is None or r.compare_id in svc._records


def test_a_failing_second_record_leaves_no_orphan_baseline(tiny_atlas):
    svc = SimulationService.from_simulation(_sim(tiny_atlas), max_results=4)
    bad = _rec(svc)
    bad["trials"] = "not a number"
    with pytest.raises(ValueError):
        svc.add_pair(_rec(svc), bad)
    assert len(svc._records) == 0


# --- S20: load failure is remembered -------------------------------------------------------


def _flaky(errors):
    calls = []

    def loader():
        calls.append(1)
        err = errors.pop(0) if errors else None
        if err is not None:
            raise err
        return object()

    return loader, calls


def test_an_unavailable_load_is_recorded_and_available_turns_false():
    loader, _ = _flaky([SimulationUnavailable("cannot read the weights cache")])
    svc = SimulationService(loader, lambda: True)
    assert svc.available is True
    with pytest.raises(SimulationUnavailable):
        svc.simulation()
    assert svc.available is False


def test_a_transient_failure_is_not_recorded():
    loader, calls = _flaky([MemoryError("no ram")])
    svc = SimulationService(loader, lambda: True)
    with pytest.raises(MemoryError):
        svc.simulation()
    assert svc.available is True  # retryable: still offered
    svc.simulation()
    assert len(calls) == 2


def test_a_later_successful_load_clears_the_failure():
    loader, _ = _flaky([SimulationUnavailable("weights missing")])
    svc = SimulationService(loader, lambda: True)
    with pytest.raises(SimulationUnavailable):
        svc.simulation()
    svc.simulation()  # the weights were built meanwhile
    assert svc.available is True
