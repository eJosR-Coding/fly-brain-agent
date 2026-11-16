"""Brain preview: a tiny static, depth-shaded picture of the cloud at the viewer's home angle."""

import struct

import numpy as np
from fastapi.testclient import TestClient

from flybrain.api import create_app
from flybrain.brain_preview import HOME_AZIMUTH_DEG, HOME_ELEVATION_DEG, LONG_SIDE, PAD, render_preview


def grid(result):
    w, h, data = result
    assert len(data) == 4 + w * h
    assert struct.unpack("<HH", data[:4]) == (w, h)
    return np.frombuffer(data, dtype=np.uint8, offset=4).reshape(h, w)


def cloud(n=400, seed=0, scale=(1000.0, 400.0, 200.0)):
    return np.random.default_rng(seed).random((n, 3)) * np.array(scale)


def test_home_angles_match_the_viewer():
    assert (HOME_AZIMUTH_DEG, HOME_ELEVATION_DEG) == (16, 10)  # web/src/scene.ts HOME_VIEW


def test_header_size_and_long_side():
    w, h, data = render_preview(cloud())
    assert max(w, h) == LONG_SIDE == 384
    assert len(data) == 4 + w * h
    assert struct.unpack("<HH", data[:4]) == (w, h)


def test_aspect_ratio_is_kept():
    # a flat, wide slab seen almost front-on: much wider than tall
    pts = cloud(scale=(1000.0, 100.0, 1.0))
    w, h, _ = render_preview(pts)
    assert w == LONG_SIDE and h < w / 3


def test_deterministic():
    pts = cloud()
    assert render_preview(pts) == render_preview(pts.copy())


def test_empty_input_is_a_1x1_zero_grid():
    w, h, data = render_preview(np.zeros((0, 3)))
    assert (w, h) == (1, 1) and data == struct.pack("<HH", 1, 1) + b"\x00"


def test_single_point_does_not_crash():
    g = grid(render_preview(np.array([[5.0, 5.0, 5.0]])))
    assert g.max() > 0


def test_extreme_points_stay_inside_the_padding():
    g = grid(render_preview(cloud(2000)))
    assert g[:PAD].max() == 0 and g[-PAD:].max() == 0
    assert g[:, :PAD].max() == 0 and g[:, -PAD:].max() == 0
    assert g.max() > 0


def test_near_vs_far_in_distinct_cells():
    az, el = np.radians(16), np.radians(10)
    d = np.array([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)])
    right = np.cross(-d, [0, 1, 0])
    right /= np.linalg.norm(right)
    # two points with identical depth-independent screen spacing but opposite depth, each alone in its cell
    a_view = right * 200.0 + d * 300.0  # near
    b_view = -right * 200.0 - d * 300.0  # far
    pts = np.array([[v[0], -v[1], v[2]] for v in (a_view, b_view)])
    g = grid(render_preview(pts))
    cells = np.argwhere(g > 0)
    assert len(cells) == 2
    cols = sorted(cells, key=lambda c: c[1])
    far_cell, near_cell = g[tuple(cols[0])], g[tuple(cols[1])]
    assert near_cell > far_cell


def test_route(tiny_tsv_path, tiny_atlas):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as c:
        r = c.get("/brain/preview")
        r2 = c.get("/brain/preview")
    assert r.status_code == 200
    w, h = struct.unpack("<HH", r.content[:4])
    assert len(r.content) == 4 + w * h
    assert "max-age" in r.headers["cache-control"]
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.content == r2.content
