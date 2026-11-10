"""GET /neurons/search: the no-LLM path the viewer's quick filters use."""

import numpy as np
from fastapi.testclient import TestClient

from flybrain.api import create_app


def client_for(path, **kw):
    return TestClient(create_app(atlas_path=path, **kw))


def test_search_returns_inline_indices(tiny_tsv_path):
    with client_for(tiny_tsv_path) as c:
        body = c.get("/neurons/search", params={"cell_type": "EPG"}).json()
    assert body["total"] == 1
    assert body["indices"] == [60]
    assert body["filters"] == {"cell_type": "EPG"}
    assert isinstance(body["selection_id"], str)
    assert "fetch" not in body


def test_search_combines_filters(tiny_tsv_path):
    with client_for(tiny_tsv_path) as c:
        body = c.get("/neurons/search", params={"nt": "acetylcholine", "side": "left"}).json()
    assert body["total"] == 61
    assert body["indices"] == list(range(60)) + [61]


def test_search_big_selection_is_fetched_as_binary(tiny_tsv_path):
    with client_for(tiny_tsv_path, max_inline_indices=10) as c:
        body = c.get("/neurons/search", params={"nt": "acetylcholine"}).json()
        assert body["indices"] is None
        assert body["fetch"] == f"/selections/{body['selection_id']}/indices"
        rows = np.frombuffer(c.get(body["fetch"]).content, "<u4")
    assert rows.tolist() == list(range(60)) + [61]  # same ticket store as the SSE path


def test_search_no_match_is_200_with_hint(tiny_tsv_path):
    with client_for(tiny_tsv_path) as c:
        r = c.get("/neurons/search", params={"cell_type": "NOPE"})
    body = r.json()
    assert r.status_code == 200
    assert body["total"] == 0
    assert body["selection_id"] is None
    assert body["indices"] == []
    assert "No neurons matched" in body["hint"]


def test_search_invalid_filter_is_422_with_message(tiny_tsv_path):
    with client_for(tiny_tsv_path) as c:
        r = c.get("/neurons/search", params={"cell_type": "x" * 500})
    assert r.status_code == 422
    assert "cell_type" in r.json()["detail"]


def test_search_unexpected_type_error_is_not_a_422(tiny_tsv_path, monkeypatch):
    import flybrain.api as api

    def boom(*a, **kw):
        raise TypeError("boom")

    monkeypatch.setattr(api, "search_neurons", boom)
    app = create_app(atlas_path=tiny_tsv_path)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/neurons/search", params={"cell_type": "EPG"})
    assert r.status_code == 500
