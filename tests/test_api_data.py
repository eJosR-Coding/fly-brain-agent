"""Brain data endpoints: health, meta contract, binary points. No model needed."""

import numpy as np
from fastapi.testclient import TestClient

from flybrain.api import create_app
from flybrain.atlas import RowIndex



def client_for(path):
    return TestClient(create_app(atlas_path=path))


def test_health(tiny_tsv_path):
    with client_for(tiny_tsv_path) as c:
        assert c.get("/health").json() == {"status": "ok", "neurons": 63}


def test_meta_contract(tiny_tsv_path, tiny_atlas):
    with client_for(tiny_tsv_path) as c:
        meta = c.get("/brain/meta").json()
    assert meta["count"] == 63
    assert meta["dtype"] == "float32"
    assert meta["endianness"] == "little"
    assert meta["layout"] == "xyz-interleaved"
    assert meta["order"] == "atlas rows"
    assert meta["bytes"] == 63 * 12
    assert meta["extent_um"] == list(tiny_atlas.extent_um())


def test_points_binary_matches_atlas(tiny_tsv_path, tiny_atlas):
    with client_for(tiny_tsv_path) as c:
        r = c.get("/brain/points")
    assert r.headers["content-type"].startswith("application/octet-stream")
    assert len(r.content) == 63 * 12
    got = np.frombuffer(r.content, "<f4").reshape(-1, 3)
    want = tiny_atlas.neurons[["x_um", "y_um", "z_um"]].to_numpy().astype("<f4")
    assert np.array_equal(got, want)


def test_row_index_maps_ids_and_counts_missing(tiny_atlas):
    idx = RowIndex(tiny_atlas)
    rows, missing = idx.lookup([1, 100, 999999, 102])
    assert rows.dtype == np.dtype("<u4")
    assert rows.tolist() == [0, 60, 62]  # unknown id is ignored, not an error
    assert missing == 1
