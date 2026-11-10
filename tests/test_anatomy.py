"""Region codes and data-derived landmarks (no model, no hardcoded geometry)."""

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import HEADER
from flybrain.anatomy import NT_CODES, REGION_CODES, landmarks, nt_codes, region_codes
from flybrain.api import create_app
from flybrain.atlas import load_atlas


def make_atlas(tmp_path, rows):
    """rows: (super_class, cell_type, side, x_nm_voxels, y, z). pos_* are voxels (4,4,40 nm)."""
    lines = [HEADER]
    for i, (sc, ct, side, x, y, z) in enumerate(rows, start=1):
        lines.append(f"{i}\t{x}\t{y}\t{z}\t{sc}\tCX\t{ct}\tacetylcholine\t{side}\tintrinsic")
    path = tmp_path / "a.tsv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_region_table_is_small_and_fixed():
    assert REGION_CODES == {0: "central", 1: "optic", 2: "sensory", 3: "other"}


def test_region_codes_follow_super_class(tmp_path):
    scs = ["central", "optic", "visual_projection", "visual_centrifugal", "sensory",
           "sensory_ascending", "ascending", "descending", "motor", "endocrine"]
    atlas = load_atlas(make_atlas(tmp_path, [(s, "X", "left", 1000, 1000, 100) for s in scs]))
    codes = region_codes(atlas)
    assert codes.dtype == np.uint8
    assert codes.tolist() == [0, 1, 1, 1, 2, 2, 3, 3, 3, 3]


def test_missing_super_class_is_other(tmp_path):
    path = tmp_path / "a.tsv"
    path.write_text(HEADER + "\n1\t10\t10\t10\t\tCX\tX\t\tleft\t\n")
    assert region_codes(load_atlas(path)).tolist() == [3]


def test_landmark_centroids_are_exact(tmp_path):
    # um = voxel * (4, 4, 40) / 1000
    rows = [
        ("central", "PEN_a", "left", 1000, 1000, 100),   # (4, 4, 4) um
        ("central", "EPG", "right", 3000, 2000, 300),    # (12, 8, 12) um
        ("optic", "T4a", "left", 500, 500, 50),          # (2, 2, 2)
        ("optic", "T4b", "left", 1500, 1500, 150),       # (6, 6, 6)
        ("optic", "T4c", "right", 4000, 500, 50),        # (16, 2, 2)
        ("optic", "L1", "left", 0, 500, 50),             # lamina L: (0, 2, 2)
        ("optic", "R1-6", "left", 0, 1500, 150),         # lamina L: (0, 6, 6)
        ("optic", "L3", "right", 5000, 500, 50),         # lamina R: (20, 2, 2)
    ]
    atlas = load_atlas(make_atlas(tmp_path, rows))
    out = landmarks(atlas)
    by = {m["name"]: m for m in out["landmarks"]}
    assert by["central brain"] == {"name": "central brain", "x": 8.0, "y": 6.0, "z": 8.0, "count": 2}
    assert by["lamina L"]["count"] == 2
    assert (by["lamina L"]["x"], by["lamina L"]["y"], by["lamina L"]["z"]) == (0.0, 4.0, 4.0)
    assert (by["lamina R"]["x"], by["lamina R"]["count"]) == (20.0, 1)
    assert by["optic lobe L"]["count"] == 4
    assert by["optic lobe L"]["x"] == pytest.approx((2 + 6 + 0 + 0) / 4)
    assert by["optic lobe R"]["count"] == 2


def test_lamina_direction_is_unit_vector_from_lobe_to_lamina(tmp_path):
    rows = [
        ("optic", "T4a", "left", 2500, 500, 50),    # lobe L pts: (10,2,2), (0,2,2)
        ("optic", "T4b", "left", 0, 500, 50),
        ("optic", "L1", "left", 0, 500, 50),        # lamina L at (0,2,2)
    ]
    atlas = load_atlas(make_atlas(tmp_path, rows))
    side = landmarks(atlas)["sides"]["left"]
    # lobe centroid x = (10 + 0 + 0) / 3, lamina x = 0 -> points toward -x
    assert side["lamina_direction"] == pytest.approx([-1.0, 0.0, 0.0])
    assert math.isclose(sum(c * c for c in side["lamina_direction"]), 1.0)
    assert side["lobe_center"] == pytest.approx([10 / 3, 2.0, 2.0])


def test_lobe_radius_is_80th_percentile_of_distances(tmp_path):
    # five optic points on the x axis around centroid x=10um: 0, 5, 10, 15, 20
    rows = [("optic", "T4a", "left", int(x * 250), 500, 50) for x in (0, 5, 10, 15, 20)]
    side = landmarks(load_atlas(make_atlas(tmp_path, rows)))["sides"]["left"]
    dists = np.abs(np.array([0, 5, 10, 15, 20]) - 10.0)
    assert side["radius"] == pytest.approx(float(np.percentile(dists, 80)))


def test_missing_groups_are_omitted_not_a_crash(tiny_atlas):
    out = landmarks(tiny_atlas)  # tiny: central + one optic (left), no lamina, no right lobe
    names = {m["name"] for m in out["landmarks"]}
    assert names == {"central brain", "optic lobe L"}
    assert set(out["sides"]) == {"left"}
    assert out["sides"]["left"]["lamina_direction"] is None


def test_sides_center_and_na_are_ignored(tmp_path):
    rows = [("optic", "T4a", "center", 100, 100, 10), ("optic", "T4a", "na", 100, 100, 10)]
    out = landmarks(load_atlas(make_atlas(tmp_path, rows)))
    assert out["landmarks"] == [] and out["sides"] == {}


def test_endpoints(tiny_tsv_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as c:
        meta = c.get("/brain/meta").json()
        assert meta["regions"] == {"0": "central", "1": "optic", "2": "sensory", "3": "other"}
        r = c.get("/brain/regions")
        assert r.headers["content-type"].startswith("application/octet-stream")
        assert len(r.content) == meta["count"] == 63
        codes = np.frombuffer(r.content, np.uint8)
        assert codes[:61].tolist() == [0] * 61 and codes[61] == 1 and codes[62] == 2
        body = c.get("/brain/landmarks").json()
        assert {m["name"] for m in body["landmarks"]} == {"central brain", "optic lobe L"}


def test_lamina_includes_photoreceptors_whatever_their_super_class(tmp_path):
    # R1-6 terminals are annotated "sensory" in FlyWire; they still belong to the lamina landmark
    rows = [
        ("optic", "T4a", "left", 2500, 500, 50),
        ("optic", "L1", "left", 0, 500, 50),          # (0, 2, 2)
        ("sensory", "R1-6", "left", 0, 1500, 150),    # (0, 6, 6)
        ("sensory", "ORN_DA1", "left", 0, 9000, 150),  # not lamina
    ]
    out = landmarks(load_atlas(make_atlas(tmp_path, rows)))
    lam = {m["name"]: m for m in out["landmarks"]}["lamina L"]
    assert (lam["count"], lam["y"], lam["z"]) == (2, 4.0, 4.0)
    lobe = {m["name"]: m for m in out["landmarks"]}["optic lobe L"]
    assert lobe["count"] == 2  # the lobe stays optic-only


def test_nt_table_is_fixed_and_ends_with_unknown():
    assert NT_CODES == {0: "acetylcholine", 1: "glutamate", 2: "gaba", 3: "dopamine",
                        4: "serotonin", 5: "octopamine", 6: "unknown"}


def test_nt_codes_follow_top_nt_case_insensitively(tmp_path):
    names = ["acetylcholine", "Glutamate", "GABA", "dopamine", "serotonin", "octopamine", "", "nonsense"]
    lines = [HEADER] + [f"{i}\t10\t10\t10\tcentral\tCX\tX\t{nt}\tleft\tintrinsic" for i, nt in enumerate(names, 1)]
    path = tmp_path / "nt.tsv"
    path.write_text("\n".join(lines) + "\n")
    codes = nt_codes(load_atlas(path))
    assert codes.dtype == np.uint8
    assert codes.tolist() == [0, 1, 2, 3, 4, 5, 6, 6]


def test_nt_endpoint_matches_meta_and_row_order(tiny_tsv_path):
    with TestClient(create_app(atlas_path=tiny_tsv_path)) as c:
        meta = c.get("/brain/meta").json()
        assert meta["nt"]["3"] == "dopamine" and meta["nt"]["6"] == "unknown"
        r = c.get("/brain/nt")
        assert r.headers["content-type"].startswith("application/octet-stream")
        assert len(r.content) == meta["count"] == 63
        codes = np.frombuffer(r.content, np.uint8)
        # 60 acetylcholine, 1 glutamate, 1 acetylcholine (optic), 1 blank -> unknown
        assert codes[:60].tolist() == [0] * 60 and codes[60] == 1 and codes[61] == 0 and codes[62] == 6
