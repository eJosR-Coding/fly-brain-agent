"""GET /neurons/{row}: the click-to-inspect card. Real tiny atlas, optional tiny connectome."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import TINY_EDGES
from flybrain.api import create_app
from flybrain.connectome_build import write_npz


@pytest.fixture
def conn_path(tmp_path):
    pre, post, syn = (np.array(c, dtype=np.int32) for c in zip(*TINY_EDGES))
    path = tmp_path / "c.npz"
    write_npz(path, pre, post, syn, n_neurons=63, source="x",
              stats={"rows_in": 1, "dropped_unknown": 0, "dropped_self_loops": 0})
    return path


def test_neuron_card_has_identity_position_region_and_partners(tiny_tsv_path, conn_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path, connectome_path=conn_path)) as c:
        r = c.get("/neurons/60")
    assert r.status_code == 200
    body = r.json()
    assert body["row"] == 60
    assert body["root_id"] == "100" and isinstance(body["root_id"], str)  # string: no JS precision loss
    assert (body["cell_type"], body["cell_class"], body["super_class"]) == ("EPG", "CX", "central")
    assert (body["side"], body["top_nt"], body["region"]) == ("right", "glutamate", "central")
    assert body["position_um"] == pytest.approx([3000 * 4 / 1000, 2500 * 4 / 1000, 200 * 40 / 1000])
    assert body["partners"] is True
    assert body["outputs"] == [{"row": 0, "cell_type": "PEN_a", "super_class": "central", "synapses": 10}]
    assert body["inputs"] == [{"row": 61, "cell_type": "T4a", "super_class": "optic", "synapses": 6}]
    assert body["synapses_out"] == 10 and body["synapses_in"] == 6


def test_partners_are_top_five_strongest_first(tiny_tsv_path, tmp_path):
    # row 0 -> rows 1..7 with growing synapse counts
    pre = np.zeros(7, dtype=np.int32)
    post = np.arange(1, 8, dtype=np.int32)
    syn = np.arange(10, 17, dtype=np.int32)
    path = tmp_path / "many.npz"
    write_npz(path, pre, post, syn, n_neurons=63, source="x",
              stats={"rows_in": 1, "dropped_unknown": 0, "dropped_self_loops": 0})
    with TestClient(create_app(atlas_path=tiny_tsv_path, connectome_path=path)) as c:
        body = c.get("/neurons/0").json()
    assert [p["synapses"] for p in body["outputs"]] == [16, 15, 14, 13, 12]
    assert body["synapses_out"] == sum(range(10, 17))  # the total counts every partner, not just the top 5


def test_unannotated_fields_come_back_as_null(tiny_tsv_path, conn_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path, connectome_path=conn_path)) as c:
        body = c.get("/neurons/62").json()  # ORN_DA1: no cell_class, no top_nt
    assert body["cell_class"] is None and body["top_nt"] is None
    assert body["region"] == "sensory"


def test_without_a_connectome_partners_are_flagged_absent(tiny_tsv_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as c:
        body = c.get("/neurons/60").json()
    assert body["partners"] is False
    assert "inputs" not in body and "outputs" not in body
    assert body["cell_type"] == "EPG"


@pytest.mark.parametrize("row", [63, 999999, -1])
def test_out_of_range_row_is_404(tiny_tsv_path, row):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as c:
        assert c.get(f"/neurons/{row}").status_code == 404


@pytest.mark.parametrize("row", ["abc", "1.5", "60;drop"])
def test_non_integer_row_is_422(tiny_tsv_path, row):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as c:
        assert c.get(f"/neurons/{row}").status_code == 422


def test_search_route_still_wins_over_the_row_route(tiny_tsv_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as c:
        assert c.get("/neurons/search", params={"nt": "glutamate"}).status_code == 200


def test_neuron_card_carries_a_plain_language_line(tiny_tsv_path, conn_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path, connectome_path=conn_path)) as c:
        body = c.get("/neurons/60").json()  # EPG -> PEN_a (central)
    assert "heading" in body["plain"]  # the famous-type entry
    assert "glutamate" in body["plain"]


def test_plain_line_names_the_strongest_output_class_and_works_without_a_connectome(tiny_tsv_path, conn_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path, connectome_path=conn_path)) as c:
        with_conn = c.get("/neurons/61").json()  # T4a: class OL is unknown to the glossary, super optic
    assert "strongest output is" in with_conn["plain"]
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as c:
        assert c.get("/neurons/61").json()["plain"]


def test_plain_line_prefers_known_nt_from_the_file(tmp_path):
    from conftest import HEADER, _tiny_rows

    rows = _tiny_rows()
    path = tmp_path / "k.tsv"
    path.write_text("\n".join([HEADER + "\tknown_nt", *[f"{r}\t{'acetylcholine; sNPF' if i == 60 else ''}" for i, r in enumerate(rows)]]) + "\n")
    with TestClient(create_app(atlas_path=path)) as c:
        body = c.get("/neurons/60").json()
    assert "known to use acetylcholine" in body["plain"] and "machine prediction says glutamate" in body["plain"]
