import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from flybrain.fly_body_build import (
    MeshPart,
    Part,
    assemble_mesh,
    assemble_mesh_ranges,
    build_rig,
    rotate_about,
    mirror_joint,
    decode_mesh,
    encode_mesh,
    mirror_mesh_parts,
    surface_faces,
    body_transforms,
    decode_segments,
    dequantize,
    drop_midplane_faces,
    encode_segments,
    feature_edges,
    mirror_parts,
    mirror_y,
    parse_pose,
    parse_rigging,
    parse_stl,
    quantize,
    verify_sha256,
    weld,
    write_outputs,
)

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "web" / "public"


def stl_bytes(tris) -> bytes:
    tris = np.asarray(tris, dtype="<f4").reshape(-1, 3, 3)
    out = bytearray(b"\0" * 80 + struct.pack("<I", len(tris)))
    for t in tris:
        out += struct.pack("<3f", 0, 0, 0) + t.tobytes() + b"\0\0"
    return bytes(out)


def cube_tris(size=1.0):
    v = np.array([[x, y, z] for x in (0, size) for y in (0, size) for z in (0, size)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    return np.array(tris)


def grid_plane(n=4):
    tris = []
    for i in range(n):
        for j in range(n):
            p = lambda a, b: [a, b, 0.0]
            tris += [[p(i, j), p(i + 1, j), p(i + 1, j + 1)], [p(i, j), p(i + 1, j + 1), p(i, j + 1)]]
    return np.array(tris)


def pyramid_tris():
    a, b, c, d, e = [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]
    return np.array([[a, c, b], [a, d, c], [a, b, e], [b, c, e], [c, d, e], [d, a, e]], float)


def edges_of(tris, angle=30.0):
    verts, faces = weld(tris)
    return feature_edges(verts, faces, angle)


# ---- STL parsing ----------------------------------------------------------------------


def test_parse_stl_reads_binary_triangles():
    tris = cube_tris()
    got = parse_stl(stl_bytes(tris))
    assert got.shape == (12, 3, 3)
    assert np.allclose(got, tris)


def test_parse_stl_rejects_non_finite_coordinates():
    for bad in (np.nan, np.inf, -np.inf):
        tris = cube_tris()
        tris[0, 0, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            parse_stl(stl_bytes(tris))


def test_parse_stl_rejects_a_size_that_is_not_84_plus_50_n():
    data = stl_bytes(cube_tris())
    with pytest.raises(ValueError, match="does not match"):
        parse_stl(data[:-1])


def test_parse_stl_rejects_truncated_and_empty_files():
    data = stl_bytes(cube_tris())
    with pytest.raises(ValueError):
        parse_stl(data[:-10])
    with pytest.raises(ValueError):
        parse_stl(b"short")
    with pytest.raises(ValueError):
        parse_stl(data + b"extra")


# ---- weld -----------------------------------------------------------------------------


def test_weld_merges_shared_vertices():
    verts, faces = weld(cube_tris())
    assert len(verts) == 8 and faces.shape == (12, 3)


def test_weld_drops_degenerate_triangles():
    tris = np.concatenate([cube_tris(), [[[0, 0, 0], [0, 0, 0], [1, 1, 1]]]])
    _, faces = weld(tris)
    assert len(faces) == 12


# ---- feature edges --------------------------------------------------------------------


def test_cube_has_exactly_its_twelve_edges():
    seg = edges_of(cube_tris())
    assert seg.shape == (12, 2, 3)  # the 6 face diagonals are coplanar and must vanish
    assert np.allclose(np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1), 1.0)


def test_flat_subdivided_plane_has_only_its_boundary():
    assert edges_of(grid_plane(4)).shape == (16, 2, 3)


def test_pyramid_edges_are_apex_and_base_rim():
    assert edges_of(pyramid_tris()).shape == (8, 2, 3)


def test_threshold_angle_controls_which_creases_survive():
    # two triangles folded by 20 degrees: a crease at 30 degree threshold is dropped, at 10 kept
    a = np.radians(20)
    tris = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [0, 0, 0], [0, -np.cos(a), np.sin(a)]]], float
    )
    shared = lambda s: sum(
        1 for e in s if {tuple(e[0]), tuple(e[1])} == {(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)}
    )
    assert shared(edges_of(tris, 30.0)) == 0
    assert shared(edges_of(tris, 10.0)) == 1


def test_feature_edges_are_deterministic():
    assert np.array_equal(edges_of(cube_tris()), edges_of(cube_tris()))


def test_drop_midplane_faces_removes_an_internal_septum_only():
    box = cube_tris(2.0) - [0.0, 1.0, 0.0]  # y from -1 to 1
    septum = [
        [[0, 0, 0], [2, 0, 0], [2, 0, 2]],
        [[0, 0, 0], [2, 0, 2], [0, 0, 2]],
        [[0, 0, 0], [2, 0, 2], [2, 0, 0]],  # second side of the wall
    ]
    verts, faces = weld(np.concatenate([box, septum]))
    assert len(faces) == 15
    kept = drop_midplane_faces(verts, faces, tol=0.01)
    assert len(kept) == 12
    assert feature_edges(verts, kept, 30.0).shape == (12, 2, 3)


# ---- mirror ---------------------------------------------------------------------------


def test_mirror_y_flips_only_the_sagittal_axis():
    seg = np.array([[[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]])
    assert np.array_equal(mirror_y(seg), [[[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]])


def test_mirror_parts_adds_right_copies_and_keeps_center_once():
    seg = np.array([[[0.0, 1.0, 0.0], [1.0, 2.0, 0.0]]])
    parts = [Part("c_head", "head", "c", seg), Part("lf_coxa", "leg", "l", seg), Part("l_eye", "eye", "l", seg)]
    out = mirror_parts(parts)
    assert [p.name for p in out] == ["c_head", "lf_coxa", "l_eye", "rf_coxa", "r_eye"]
    assert [p.side for p in out] == ["c", "l", "l", "r", "r"]
    right = {p.name: p for p in out}["rf_coxa"]
    assert np.array_equal(right.segments, mirror_y(seg))
    assert right.group == "leg"


# ---- quantization ---------------------------------------------------------------------


def test_quantize_round_trip_error_is_half_a_step():
    rng = np.random.default_rng(0)
    lo, hi = np.array([-2500.0, -1500.0, -100.0]), np.array([1100.0, 1500.0, 1800.0])
    pts = rng.uniform(lo, hi, size=(2000, 3))
    q = quantize(pts, lo, hi)
    assert q.dtype == np.uint16
    err = np.abs(dequantize(q, lo, hi) - pts)
    assert (err <= (hi - lo) / 65535 / 2 + 1e-9).all()


def test_quantize_hits_the_extremes_exactly():
    lo, hi = np.zeros(3), np.array([10.0, 20.0, 30.0])
    q = quantize(np.array([lo, hi]), lo, hi)
    assert q[0].tolist() == [0, 0, 0] and q[1].tolist() == [65535, 65535, 65535]


def test_encode_is_little_endian_and_twelve_bytes_per_segment():
    seg = np.array([[[0.0, 0, 0], [10.0, 20, 30]]])
    parts = [Part("c_a", "head", "c", seg), Part("c_b", "thorax", "c", seg)]
    lo, hi = np.zeros(3), np.array([10.0, 20.0, 30.0])
    blob, index = encode_segments(parts, lo, hi)
    assert len(blob) == 24
    assert struct.unpack("<6H", blob[:12]) == (0, 0, 0, 65535, 65535, 65535)
    assert index == [(0, 1), (1, 1)]
    assert np.allclose(decode_segments(blob, lo, hi), np.concatenate([seg, seg]))


def test_encode_is_deterministic():
    seg = edges_of(cube_tris(100.0))
    parts = [Part("c_cube", "head", "c", seg)]
    lo, hi = np.zeros(3), np.full(3, 100.0)
    assert encode_segments(parts, lo, hi) == encode_segments(parts, lo, hi)


def test_encode_segments_handles_no_parts_and_empty_parts_without_misaligning_the_index():
    seg = np.array([[[0.0, 0, 0], [10.0, 20, 30]]])
    lo, hi = np.zeros(3), np.array([10.0, 20.0, 30.0])
    assert encode_segments([], lo, hi) == (b"", [])
    parts = [Part("c_a", "head", "c", seg), Part("c_e", "head", "c", np.zeros((0, 2, 3))),
             Part("c_f", "head", "c", np.zeros(0)), Part("c_b", "head", "c", seg)]
    blob, index = encode_segments(parts, lo, hi)
    assert len(blob) == 24 and index == [(0, 1), (1, 0), (1, 0), (1, 1)]


def test_encode_segments_refuses_a_malformed_part_loudly():
    with pytest.raises(ValueError, match="c_bad"):
        encode_segments([Part("c_bad", "head", "c", np.zeros((4, 3)))], np.zeros(3), np.ones(3))


# ---- frame transform ------------------------------------------------------------------


RIGGING = """\
c_thorax:
  pos: [0.0, 0.0, 1.0]
  quat: [1.0, 0.0, 0.0, 0.0]
  mass: 1e-4
c_head:
  pos: [1.0, 0.0, 0.0]
  quat: [1.0, 0.0, 0.0, 0.0]
  mass: 1e-5
c_rostrum:
  pos: [1.0, 0.0, 0.0]
  quat: [1.0, 0.0, 0.0, 0.0]
  mass: 1e-6
"""

POSE = """\
angle_unit: degree

axis_order: [roll, pitch, yaw]

joint_angles:
  c_thorax-c_head-roll: 90
  c_head-c_rostrum-pitch: -5   # trailing comment
"""


def test_parse_rigging_reads_pos_and_quat_in_um():
    rig = parse_rigging(RIGGING)
    assert set(rig) == {"c_thorax", "c_head", "c_rostrum"}
    pos, quat = rig["c_head"]
    assert pos.tolist() == [1000.0, 0.0, 0.0]  # mm -> um
    assert quat.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_parse_pose_returns_order_and_radians():
    order, angles = parse_pose(POSE)
    assert order == ["roll", "pitch", "yaw"]
    assert angles["c_thorax-c_head-roll"] == pytest.approx(np.pi / 2)
    assert angles["c_head-c_rostrum-pitch"] == pytest.approx(np.radians(-5))


def test_body_transforms_compose_the_hierarchy_with_roll_about_z():
    rig = parse_rigging(RIGGING)
    order, angles = parse_pose(POSE)
    parent = {"c_head": "c_thorax", "c_rostrum": "c_head"}
    T = body_transforms(rig, parent, order, angles)
    assert np.allclose(T["c_thorax"][1], [0, 0, 1000])
    assert np.allclose(T["c_head"][1], [1000, 0, 1000])
    # head rolled 90 degrees about z: the rostrum offset (1,0,0) now points along +y
    assert np.allclose(T["c_rostrum"][1], [1000, 1000, 1000], atol=1e-6)


def test_body_transforms_uses_quaternion_w_first():
    rig = parse_rigging(RIGGING.replace("quat: [1.0, 0.0, 0.0, 0.0]\n  mass: 1e-5", "quat: [0.7071068, 0.0, 0.0, 0.7071068]\n  mass: 1e-5"))
    T = body_transforms(rig, {"c_head": "c_thorax", "c_rostrum": "c_head"}, ["roll", "pitch", "yaw"], {})
    assert np.allclose(T["c_rostrum"][1], [1000, 1000, 1000], atol=1e-3)


# ---- sha256 ---------------------------------------------------------------------------


def test_verify_sha256_accepts_matching_and_refuses_other(tmp_path):
    f = tmp_path / "w.whl"
    f.write_bytes(b"payload")
    verify_sha256(f, hashlib.sha256(b"payload").hexdigest())
    with pytest.raises(ValueError, match="sha256"):
        verify_sha256(f, "0" * 64)


# ---- committed asset ------------------------------------------------------------------

ASSET_JSON = PUBLIC / "fly-body.json"
ASSET_BIN = PUBLIC / "fly-body.bin"
needs_asset = pytest.mark.skipif(not (ASSET_JSON.exists() and ASSET_BIN.exists()), reason="asset not built")


@needs_asset
def test_asset_parts_cover_every_segment_exactly_once():
    meta = json.loads(ASSET_JSON.read_text())
    total = ASSET_BIN.stat().st_size // 12
    covered = np.zeros(total, dtype=int)
    for p in meta["parts"]:
        covered[p["segment_start"] : p["segment_start"] + p["segment_count"]] += 1
    assert (covered == 1).all()


@needs_asset
def test_asset_size_is_twelve_bytes_per_segment_and_within_budget():
    meta = json.loads(ASSET_JSON.read_text())
    size = ASSET_BIN.stat().st_size
    assert size % 12 == 0
    assert size == 12 * sum(p["segment_count"] for p in meta["parts"])
    assert size <= 600_000


@needs_asset
def test_asset_has_every_group_and_both_sides():
    meta = json.loads(ASSET_JSON.read_text())
    groups = {p["group"] for p in meta["parts"]}
    assert groups == {"head", "thorax", "abdomen", "wing", "haltere", "leg", "antenna", "eye", "proboscis"}
    assert {"c", "l", "r"} == {p["side"] for p in meta["parts"]}


@needs_asset
def test_asset_body_length_head_center_and_symmetry_are_plausible():
    meta = json.loads(ASSET_JSON.read_text())
    lo, hi = np.array(meta["bounds"][0]), np.array(meta["bounds"][1])
    segs = decode_segments(ASSET_BIN.read_bytes(), lo, hi)
    body = np.zeros(len(segs), dtype=bool)
    for p in meta["parts"]:
        if p["group"] in ("head", "thorax", "abdomen"):
            body[p["segment_start"] : p["segment_start"] + p["segment_count"]] = True
    xs = segs[body][..., 0]
    assert 2000 <= xs.max() - xs.min() <= 3500  # real fly body is about 2.5-3 mm
    center = np.array(meta["head_center_um"])
    assert (center >= lo).all() and (center <= hi).all()
    assert xs.min() < center[0] < xs.max()
    # sagittal symmetry: y bounds are mirrored
    assert lo[1] == pytest.approx(-hi[1], abs=1.0)


@needs_asset
def test_asset_metadata_credits_the_source_and_license_file_exists():
    meta = json.loads(ASSET_JSON.read_text())
    assert meta["units"] == "um"
    assert meta["source"]["license"] == "Apache-2.0" and meta["source"]["version"] == "2.1.1"
    assert len(meta["source"]["sha256"]) == 64
    assert "not animated" in meta["note"]
    lic = PUBLIC / "LICENSE-neuromechfly.txt"
    assert lic.exists() and "Apache License" in lic.read_text()


# ---- surface mesh (hidden-line occluder) ----------------------------------------------

LO, HI = np.array([-1.0, -2.0, -1.0]), np.array([3.0, 2.0, 3.0])


def cube_mesh(size=1.0, offset=(0.0, 0.0, 0.0)):
    return weld(cube_tris(size) + np.asarray(offset))


def test_welded_cube_mesh_has_8_vertices_and_12_triangles():
    verts, faces = cube_mesh()
    assert verts.shape == (8, 3) and faces.shape == (12, 3)


def test_mirror_mesh_parts_gives_both_sides_and_flips_winding():
    v, f = cube_mesh(1.0, (0.0, 0.5, 0.0))
    out = mirror_mesh_parts([MeshPart("c_a", "c", v, f), MeshPart("l_b", "l", v, f)])
    assert [p.name for p in out] == ["c_a", "l_b", "r_b"]
    right = out[2]
    assert np.array_equal(right.verts[:, 1], -v[:, 1])
    assert np.array_equal(right.faces, f[:, ::-1])  # winding flipped so normals stay outward

    def volume(vv, ff):  # signed volume: positive for outward-facing triangles
        return np.einsum("ij,ij->i", vv[ff[:, 0]], np.cross(vv[ff[:, 1]], vv[ff[:, 2]])).sum() / 6

    assert np.sign(volume(right.verts, right.faces)) == np.sign(volume(v, f))


def test_surface_faces_drops_midplane_faces_of_central_parts_only():
    box = cube_tris(2.0) - [0.0, 1.0, 0.0]
    septum = [[[0, 0, 0], [2, 0, 0], [2, 0, 2]]]
    verts, faces = weld(np.concatenate([box, septum]))
    assert len(faces) == 13
    assert len(surface_faces("c_thorax", verts, faces, tol=0.01)) == 12
    assert len(surface_faces("l_wing", verts, faces, tol=0.01)) == 13


def test_assemble_mesh_merges_shared_vertices_across_parts_and_indexes_validly():
    a, b = cube_mesh(1.0), cube_mesh(1.0, (1.0, 0.0, 0.0))  # two cubes sharing a face of 4 vertices
    qv, faces = assemble_mesh([MeshPart("c_a", "c", *a), MeshPart("c_b", "c", *b)], LO, HI)
    assert qv.dtype == np.uint16 and qv.shape == (12, 3)
    assert len(faces) == 24
    assert faces.min() >= 0 and faces.max() < len(qv)


def test_mesh_round_trip_error_is_within_the_quantization_step():
    v, f = cube_mesh(1.3, (0.37, -0.81, 0.11))
    qv, faces = assemble_mesh([MeshPart("c_a", "c", v, f)], LO, HI)
    blob, info = encode_mesh(qv, faces)
    assert info["vertex_count"] == 8 and info["triangle_count"] == 12 and info["bytes"] == len(blob)
    dv, df = decode_mesh(blob, info, LO, HI)
    step = (HI - LO) / 65535
    assert np.array_equal(df, faces)
    # every decoded vertex sits within half a quantization step of an original vertex
    dist = np.abs(dv[:, None, :] - v[None, :, :]).max(axis=2).min(axis=1)
    assert (dist <= (step / 2).max() + 1e-9).all()


def test_encode_mesh_uses_uint16_indices_when_they_fit_and_uint32_otherwise():
    qv = np.zeros((4, 3), dtype="<u2")
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    blob, info = encode_mesh(qv, faces)
    assert info["index_dtype"] == "uint16" and len(blob) == 4 * 6 + 2 * 6
    qv = np.zeros((70000, 3), dtype="<u2")
    blob, info = encode_mesh(qv, np.array([[0, 1, 69999]]))
    assert info["index_dtype"] == "uint32" and len(blob) == 70000 * 6 + 12
    dv, df = decode_mesh(blob, info, np.zeros(3), np.ones(3))
    assert df.tolist() == [[0, 1, 69999]]


def test_encode_mesh_is_deterministic_and_rejects_bad_indices():
    v, f = cube_mesh()
    qv, faces = assemble_mesh([MeshPart("c_a", "c", v, f)], LO, HI)
    assert encode_mesh(qv, faces)[0] == encode_mesh(qv, faces)[0]
    with pytest.raises(ValueError):
        encode_mesh(qv, np.array([[0, 1, 99]]))


MESH_BIN = PUBLIC / "fly-body-mesh.bin"
needs_mesh = pytest.mark.skipif(
    not (ASSET_JSON.exists() and ASSET_BIN.exists() and MESH_BIN.exists()), reason="mesh asset not built"
)


@needs_mesh
def test_committed_mesh_matches_its_metadata_and_decodes_inside_the_line_bounds():
    meta = json.loads(ASSET_JSON.read_text())
    info = meta["mesh"]
    blob = MESH_BIN.read_bytes()
    assert len(blob) == info["bytes"] <= 1_500_000
    assert info["triangle_count"] > 50_000
    lo, hi = np.array(meta["bounds"][0]), np.array(meta["bounds"][1])
    verts, faces = decode_mesh(blob, info, lo, hi)
    assert verts.shape == (info["vertex_count"], 3) and faces.shape == (info["triangle_count"], 3)
    assert faces.min() >= 0 and faces.max() < info["vertex_count"]
    assert (verts >= lo - 1e-6).all() and (verts <= hi + 1e-6).all()
    # same global bounds as the line asset, and the line segments' extent matches the mesh's within the padding
    segs = decode_segments(ASSET_BIN.read_bytes(), lo, hi).reshape(-1, 3)
    assert np.abs(verts.max(axis=0) - segs.max(axis=0)).max() < 5.0
    assert np.abs(verts.min(axis=0) - segs.min(axis=0)).max() < 5.0


# ---- atomic outputs ---------------------------------------------------------------------

OUT_NAMES = ("fly-body-mesh.bin", "fly-body.bin", "fly-body.json")


def test_write_outputs_writes_the_three_files_and_leaves_no_temp_files(tmp_path):
    write_outputs(tmp_path, b"mesh", b"bin", "{}\n")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(OUT_NAMES)
    assert (tmp_path / "fly-body-mesh.bin").read_bytes() == b"mesh"
    assert (tmp_path / "fly-body.bin").read_bytes() == b"bin"
    assert (tmp_path / "fly-body.json").read_text() == "{}\n"


def test_an_interrupted_write_never_leaves_a_mixed_set(tmp_path, monkeypatch):
    write_outputs(tmp_path, b"mesh0", b"bin0", "old\n")
    import os

    real_replace = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(Path(dst).name)
        if Path(dst).name == "fly-body.json":
            raise OSError("interrupted")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(OSError):
        write_outputs(tmp_path, b"mesh1", b"bin1", "new\n")
    assert calls[-1] == "fly-body.json"  # the json is committed last
    assert (tmp_path / "fly-body.json").read_text() == "old\n"  # the json still describes the old set...
    assert not [p for p in tmp_path.iterdir() if p.name not in OUT_NAMES]  # ...and no temp file is left behind


def test_a_failure_while_staging_leaves_the_previous_files_untouched(tmp_path, monkeypatch):
    write_outputs(tmp_path, b"mesh0", b"bin0", "old\n")
    real_write = Path.write_bytes

    def boom(self, data):
        if b"bin1" == data:
            raise OSError("disk full")
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", boom)
    with pytest.raises(OSError):
        write_outputs(tmp_path, b"mesh1", b"bin1", "new\n")
    assert [(tmp_path / n).read_bytes() for n in OUT_NAMES[:2]] == [b"mesh0", b"bin0"]
    assert (tmp_path / "fly-body.json").read_text() == "old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(OUT_NAMES)


# ---- rig (animatable joints) and per-part triangle ranges ------------------------------------

RIG_IDS = ["proboscis", "antenna_l", "antenna_r", "wing_l", "wing_r", "haltere_l", "haltere_r"]
WHEEL_ENV = "FLYBRAIN_FLYGYM_WHEEL"


def synthetic_origins():
    return {
        "c_rostrum": np.array([950.0, 0.0, 1000.0]),
        "l_pedicel": np.array([950.0, 90.0, 1380.0]),
        "l_wing": np.array([-180.0, 400.0, 1380.0]),
        "l_haltere": np.array([-300.0, 210.0, 1150.0]),
    }


def synthetic_rostrum():
    return np.array([[900.0, 0.0, 1250.0], [900.0, 10.0, 1240.0], [800.0, 0.0, 800.0], [700.0, 0.0, 900.0]])


def test_rotate_about_zero_degrees_is_identity_and_ninety_keeps_distance_to_the_pivot():
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]])
    pivot, axis = np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.0])
    assert np.allclose(rotate_about(pts, pivot, axis, 0.0), pts)
    out = rotate_about(pts, pivot, axis, 90.0)
    assert np.allclose(np.linalg.norm(out - pivot, axis=1), np.linalg.norm(pts - pivot, axis=1))
    assert np.allclose(out[0], [-1.0, 1.0, 3.0])  # (0.5, 1.5, .) -> (-1.5, 0.5, .) about the pivot, z kept
    assert np.allclose(rotate_about(pivot[None], pivot, axis, 33.0), pivot[None])  # the pivot never moves


def test_rotate_about_a_non_unit_axis_is_normalised():
    pts = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(rotate_about(pts, np.zeros(3), np.array([0.0, 0.0, 5.0]), 90.0), [[0.0, 1.0, 0.0]])


def test_build_rig_has_the_contract_shape():
    rig = build_rig(synthetic_origins(), synthetic_rostrum())
    assert rig["units"] == "um" and "x anterior" in rig["frame"]
    assert [j["id"] for j in rig["joints"]] == RIG_IDS
    for j in rig["joints"]:
        assert set(j) == {"id", "parts", "pivot_um", "axis", "rest_deg", "range_deg", "source"}
        assert j["rest_deg"] == 0
        lo, hi = j["range_deg"]
        assert lo <= 0 <= hi and lo < hi
        assert abs(np.linalg.norm(j["axis"]) - 1.0) < 1e-6
        assert len(j["pivot_um"]) == 3
        assert j["source"] in ("rigging.yaml", "geometry (our choice)")
    by = {j["id"]: j for j in rig["joints"]}
    assert by["proboscis"]["parts"] == ["c_rostrum", "c_haustellum"]
    assert by["antenna_l"]["parts"] == ["l_pedicel", "l_funiculus", "l_arista"]
    assert by["antenna_r"]["parts"] == ["r_pedicel", "r_funiculus", "r_arista"]
    assert by["wing_r"]["parts"] == ["r_wing"] and by["haltere_l"]["parts"] == ["l_haltere"]


def test_right_joints_are_the_left_ones_mirrored_on_y():
    by = {j["id"]: j for j in build_rig(synthetic_origins(), synthetic_rostrum())["joints"]}
    for side in ("antenna", "wing", "haltere"):
        left, right = by[f"{side}_l"], by[f"{side}_r"]
        assert right["pivot_um"] == [left["pivot_um"][0], -left["pivot_um"][1], left["pivot_um"][2]]
        assert right["range_deg"] == left["range_deg"] and right["source"] == left["source"]
        # a mirrored rotation: the same angle moves the right part as the mirror image of the left motion
        pts = np.array([[-900.0, 120.0, 1300.0], [-500.0, 300.0, 1100.0]])
        ang = 25.0
        moved_l = rotate_about(pts, np.array(left["pivot_um"]), np.array(left["axis"]), ang)
        moved_r = rotate_about(mirror_y(pts), np.array(right["pivot_um"]), np.array(right["axis"]), ang)
        assert np.allclose(mirror_y(moved_l), moved_r)


def test_mirror_joint_flips_pivot_y_and_turns_the_axis_as_a_pseudovector():
    j = {"id": "wing_l", "parts": ["l_wing"], "pivot_um": [1.0, 2.0, 3.0], "axis": [0.0, 0.6, 0.8],
         "rest_deg": 0, "range_deg": [0, 90], "source": "geometry (our choice)"}
    r = mirror_joint(j)
    assert r["id"] == "wing_r" and r["parts"] == ["r_wing"] and r["pivot_um"] == [1.0, -2.0, 3.0]
    assert r["axis"] == [-0.0 + 0.0, 0.6, -0.8]


def test_build_rig_is_deterministic():
    a = json.dumps(build_rig(synthetic_origins(), synthetic_rostrum()))
    b = json.dumps(build_rig(synthetic_origins(), synthetic_rostrum()))
    assert a == b


def test_proboscis_pivot_is_the_dorsal_attachment_and_extension_swings_the_tip_forward():
    rostrum = synthetic_rostrum()
    j = {j["id"]: j for j in build_rig(synthetic_origins(), rostrum)["joints"]}["proboscis"]
    assert j["pivot_um"][1] == 0.0 and j["pivot_um"][2] > 1200  # top of the rostrum, on the midline
    tip = np.array([[700.0, 0.0, 900.0]])
    moved = rotate_about(tip, np.array(j["pivot_um"]), np.array(j["axis"]), j["range_deg"][1])
    assert moved[0, 0] > tip[0, 0]  # extension moves the tip anteriorly


def test_rig_wings_rise_and_antennae_sweep_outward_with_a_positive_angle():
    by = {j["id"]: j for j in build_rig(synthetic_origins(), synthetic_rostrum())["joints"]}
    o = synthetic_origins()
    wing_tip = np.array([[-2300.0, 0.0, 900.0]])
    for jid in ("wing_l", "wing_r"):
        j = by[jid]
        sign = 1.0 if jid.endswith("l") else -1.0
        tip = wing_tip * np.array([1.0, sign, 1.0])
        moved = rotate_about(tip, np.array(j["pivot_um"]), np.array(j["axis"]), 30.0)
        assert moved[0, 2] > tip[0, 2]
    arista = np.array([[1150.0, 260.0, 1440.0]])
    j = by["antenna_l"]
    moved = rotate_about(arista, np.array(j["pivot_um"]), np.array(j["axis"]), 20.0)
    assert moved[0, 1] > arista[0, 1]  # positive sweep moves the left antenna tip outward (more +y)
    assert o["l_pedicel"][1] == j["pivot_um"][1]


def test_assemble_mesh_ranges_are_contiguous_per_part_and_count_dropped_triangles():
    a = MeshPart("c_a", "c", *weld(cube_tris(1.0)))
    b = MeshPart("c_b", "c", *weld(cube_tris(1.0) + 5))
    lo, hi = np.array([0.0] * 3), np.array([10.0] * 3)
    qv, faces, ranges = assemble_mesh_ranges([a, b], lo, hi)
    assert ranges == [(0, 12), (12, 12)] and len(faces) == 24
    qv2, faces2 = assemble_mesh([a, b], lo, hi)
    assert np.array_equal(qv, qv2) and np.array_equal(faces, faces2)
    # a part whose triangles all collapse after quantization gets an empty range, not a shifted one
    tiny = MeshPart("c_t", "c", *weld(cube_tris(0.00001) + 2))
    _, faces3, ranges3 = assemble_mesh_ranges([a, tiny, b], lo * 1.0, hi * 1.0)
    assert ranges3 == [(0, 12), (12, 0), (12, 12)] and len(faces3) == 24


@needs_mesh
def test_committed_mesh_parts_cover_every_triangle_once_in_part_order():
    meta = json.loads(ASSET_JSON.read_text())
    entries = meta["mesh"]["parts"]
    assert [e["name"] for e in entries] == [p["name"] for p in meta["parts"]]
    pos = 0
    for e in entries:
        assert e["triangle_start"] == pos and e["triangle_count"] >= 0
        pos += e["triangle_count"]
    assert pos == meta["mesh"]["triangle_count"]
    assert all(e["triangle_count"] > 0 for e in entries)


@needs_mesh
def test_committed_mesh_part_ranges_match_the_geometry_of_that_part():
    meta = json.loads(ASSET_JSON.read_text())
    lo, hi = np.array(meta["bounds"][0]), np.array(meta["bounds"][1])
    verts, faces = decode_mesh(MESH_BIN.read_bytes(), meta["mesh"], lo, hi)
    by = {e["name"]: e for e in meta["mesh"]["parts"]}
    def centroid(name):
        e = by[name]
        f = faces[e["triangle_start"] : e["triangle_start"] + e["triangle_count"]]
        return verts[f].reshape(-1, 3).mean(axis=0)
    left, right = centroid("l_wing"), centroid("r_wing")
    assert abs(left[1] + right[1]) < 5 and abs(left[0] - right[0]) < 5 and abs(left[1]) > 20  # mirrored pair
    assert centroid("c_head")[0] > centroid("c_abdomen4")[0]


@needs_asset
def test_committed_rig_matches_the_contract_and_the_parts():
    meta = json.loads(ASSET_JSON.read_text())
    rig = meta["rig"]
    assert rig["units"] == "um"
    names = {p["name"] for p in meta["parts"]}
    ids = [j["id"] for j in rig["joints"]]
    assert ids[: len(RIG_IDS)] == RIG_IDS and len(ids) == len(set(ids))
    lo, hi = np.array(meta["bounds"][0]), np.array(meta["bounds"][1])
    for j in rig["joints"]:
        assert j["parts"] and set(j["parts"]) <= names
        assert abs(np.linalg.norm(j["axis"]) - 1.0) < 1e-6
        assert (np.array(j["pivot_um"]) >= lo).all() and (np.array(j["pivot_um"]) <= hi).all()
    by = {j["id"]: j for j in rig["joints"]}
    for s in ("antenna", "wing", "haltere"):
        l, r = by[f"{s}_l"], by[f"{s}_r"]
        assert r["pivot_um"][0] == l["pivot_um"][0] and r["pivot_um"][1] == -l["pivot_um"][1]


@needs_asset
def test_rotating_a_committed_joint_to_its_range_moves_the_part_a_plausible_distance():
    meta = json.loads(ASSET_JSON.read_text())
    lo, hi = np.array(meta["bounds"][0]), np.array(meta["bounds"][1])
    verts, faces = decode_mesh(MESH_BIN.read_bytes(), meta["mesh"], lo, hi)
    tri = {e["name"]: e for e in meta["mesh"]["parts"]}
    for j in meta["rig"]["joints"]:
        idx = np.concatenate([faces[tri[n]["triangle_start"] : tri[n]["triangle_start"] + tri[n]["triangle_count"]].ravel() for n in j["parts"]])
        pts = verts[np.unique(idx)]
        pivot, axis = np.array(j["pivot_um"]), np.array(j["axis"])
        assert np.linalg.norm(pts - pivot, axis=1).min() < 150  # the part is attached at its pivot
        for ang in j["range_deg"]:
            moved = rotate_about(pts, pivot, axis, ang)
            shift = np.linalg.norm(moved - pts, axis=1).max()
            assert shift < 2500  # no teleporting
            assert np.allclose(np.linalg.norm(moved - pivot, axis=1), np.linalg.norm(pts - pivot, axis=1))
        assert np.allclose(rotate_about(pts, pivot, axis, 0), pts)


def _wheel():
    import os

    w = os.environ.get(WHEEL_ENV)
    return Path(w) if w and Path(w).exists() else None


@pytest.mark.skipif(_wheel() is None, reason=f"set {WHEEL_ENV} to the verified flygym wheel")
def test_a_fresh_build_is_deterministic_and_the_committed_json_is_in_sync(tmp_path):
    from flybrain.fly_body_build import build

    build(_wheel(), tmp_path / "a")
    build(_wheel(), tmp_path / "b")
    for name in ("fly-body.json", "fly-body.bin", "fly-body-mesh.bin"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
        assert (tmp_path / "a" / name).read_bytes() == (PUBLIC / name).read_bytes()
