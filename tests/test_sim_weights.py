"""The weight-cache build command and the default data locations."""

import pytest

from flybrain import sim_service
from flybrain.simulation import Simulation


def test_build_weights_writes_the_cache_and_a_second_call_hits_it(reference_files, tiny_atlas, tmp_path):
    con, comp = reference_files
    cache = tmp_path / "out" / "w.npz"
    info = sim_service.build_weights(con, comp, tiny_atlas, cache)
    assert cache.is_file() and info["cache_hit"] is False and info["edges"] == 3
    assert sim_service.build_weights(con, comp, tiny_atlas, cache)["cache_hit"] is True


def test_default_locations_follow_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYBRAIN_SIM_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("FLYBRAIN_SIM_WEIGHTS", str(tmp_path / "c.npz"))
    assert sim_service.connectivity_path() == tmp_path / "d" / "Connectivity_783.parquet"
    assert sim_service.completeness_path() == tmp_path / "d" / "Completeness_783.csv"
    assert sim_service.weights_cache_path() == tmp_path / "c.npz"


def test_build_weights_without_source_files_says_where_to_get_them(tiny_atlas, tmp_path):
    with pytest.raises(sim_service.SimulationUnavailable, match="Connectivity_783.parquet.*philshiu"):
        sim_service.build_weights(tmp_path / "Connectivity_783.parquet", tmp_path / "c.csv", tiny_atlas, tmp_path / "w.npz")
