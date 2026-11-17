"""Build step: NeuroMechFly v2 body meshes (flygym wheel) -> compact line-segment asset for the web viewer.

Pipeline (every decision is deliberate, none accidental):
- the SHA-256 of the downloaded wheel is checked before anything is read; nothing is extracted or imported
  from it, members are read straight from the zip;
- each STL part lives in its OWN body-local frame (verified against the MJCF rigging, see README); the world
  pose of every part is rebuilt from rigging.yaml (pos/quat relative to the parent body) plus the neutral
  pose joint angles, composed down the kinematic tree exactly as flygym/MuJoCo does;
- only the left side (and the unpaired central parts) is stored in the wheel: right parts are the left ones
  mirrored about the sagittal plane y = 0;
- parts are reduced to FEATURE EDGES (creases above an angle threshold plus open/boundary edges) because the
  viewer draws white lines on black, not surfaces;
- units: metres -> micrometres, the unit of the connectome atlas; coordinates are quantized to uint16 per axis.

Binary layout of fly-body.bin: no header, little-endian, per segment 6 x uint16 = x0 y0 z0 x1 y1 z1, parts
concatenated in the order given by fly-body.json (12 bytes per segment).

A second asset, fly-body-mesh.bin, holds the welded triangle surface of ALL parts (same transforms, same
mirror, same dropped mid-plane faces). The viewer draws it colourless into the depth buffer only, so the feature
edges behind the surface are hidden (hidden-line look). Layout, little-endian, no header: `vertex_count` vertices
as 3 x uint16 (x y z, quantized over the SAME bounds as fly-body.bin, one decode rule), then `triangle_count`
triangles as 3 indices each, uint16 when vertex_count <= 65535 else uint32 (fly-body.json `mesh.index_dtype`).
Vertices are deduplicated after quantization across all parts; `mesh.vertex_count`, `mesh.triangle_count` and
`mesh.bytes` are in the json.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLYGYM_VERSION = "2.1.1"
WHEEL_NAME = "flygym-2.1.1-py3-none-any.whl"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/40/a5/b73ded2014289b06c93fd1ce53e9b179f69f8e60ccfd6b83b5daaa0baea5/"
    + WHEEL_NAME
)
WHEEL_SHA256 = "802e586848c37d382988fa4f95200763cddbc07da4ade22816edd253a04dd2fb"

MODEL_DIR = "flygym/assets/model/neuromechfly/"
MESH_DIR = MODEL_DIR + "meshes/simplified_max2000faces/"
RIGGING_MEMBER = MODEL_DIR + "rigging.yaml"
POSE_MEMBER = MODEL_DIR + "pose/_manual_specs/neutral.yaml"
LICENSE_MEMBER = "flygym-2.1.1.dist-info/licenses/LICENSE"

DEFAULT_OUT_DIR = "web/public"
DEFAULT_ANGLE_DEG = 40.0
UM_PER_M = 1e6
ROOT = "c_thorax"
# flygym convention (anatomy.RotationAxis): pitch -> y, roll -> z, yaw -> x; x forward, y left, z up
AXIS = {"pitch": "y", "roll": "z", "yaw": "x"}
GROUPS = ("head", "thorax", "abdomen", "wing", "haltere", "leg", "antenna", "eye", "proboscis")


@dataclass(frozen=True)
class Part:
    name: str
    group: str
    side: str  # c (central), l, r
    segments: np.ndarray  # (n, 2, 3) float, micrometres


@dataclass(frozen=True)
class MeshPart:
    name: str
    side: str  # c (central), l, r
    verts: np.ndarray  # (n, 3) float, micrometres
    faces: np.ndarray  # (m, 3) int indices into verts


# ---- anatomy (fused trochanter+femur model, the NeuroMechFly default) -------------------

LEG_LINKS = ["coxa", "trochanterfemur", "tibia", *(f"tarsus{i}" for i in range(1, 6))]
PARENT = {
    "c_head": "c_thorax",
    "c_rostrum": "c_head",
    "c_haustellum": "c_rostrum",
    "c_abdomen12": "c_thorax",
    "c_abdomen3": "c_abdomen12",
    "c_abdomen4": "c_abdomen3",
    "c_abdomen5": "c_abdomen4",
    "c_abdomen6": "c_abdomen5",
    "l_eye": "c_head",
    "l_pedicel": "c_head",
    "l_funiculus": "l_pedicel",
    "l_arista": "l_funiculus",
    "l_wing": "c_thorax",
    "l_haltere": "c_thorax",
}
for _leg in ("lf", "lm", "lh"):
    _prev = ROOT
    for _link in LEG_LINKS:
        PARENT[f"{_leg}_{_link}"] = _prev
        _prev = f"{_leg}_{_link}"
# central parts first, then the left parts: a stable, documented order
PART_ORDER = [ROOT, *sorted(n for n in PARENT if n.startswith("c_")), *[n for n in PARENT if not n.startswith("c_")]]


def group_of(name: str) -> str:
    link = name.split("_", 1)[1]
    if link == "head":
        return "head"
    if link == "thorax":
        return "thorax"
    if link.startswith("abdomen"):
        return "abdomen"
    if link in ("rostrum", "haustellum"):
        return "proboscis"
    if link in ("pedicel", "funiculus", "arista"):
        return "antenna"
    if link in ("eye", "wing", "haltere"):
        return link
    return "leg"


# ---- STL, weld, feature edges ------------------------------------------------------------


def parse_stl(data: bytes) -> np.ndarray:
    """Binary STL -> (n, 3, 3) float64 triangles. Raises ValueError on anything but a well-formed file."""
    if len(data) < 84:
        raise ValueError("not a binary STL: shorter than the 84-byte header")
    (n,) = struct.unpack("<I", data[80:84])
    if len(data) != 84 + 50 * n:
        raise ValueError(f"bad binary STL: header says {n} triangles, size {len(data)} does not match")
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tris = np.frombuffer(data, dtype=rec, count=n, offset=84)["v"].astype(np.float64)
    if not np.isfinite(tris).all():
        raise ValueError("bad binary STL: a coordinate is not finite (NaN or infinity)")
    return tris


def weld(tris: np.ndarray, decimals: int = 3):
    """Merge coincident vertices (rounded to `decimals`). Returns (verts, faces); degenerate faces are dropped."""
    flat = np.round(np.asarray(tris, dtype=np.float64).reshape(-1, 3), decimals)
    verts, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[ok]
    area2 = np.linalg.norm(np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)
    return verts, faces[area2 > 1e-12]


def drop_midplane_faces(verts: np.ndarray, faces: np.ndarray, tol: float = 2.0) -> np.ndarray:
    """Drop faces lying entirely in the sagittal plane y = 0 (|y| < tol).

    The central CT meshes contain an internal mid-wall: thousands of edges that sit on the symmetry plane,
    are not surface and would draw a bright false line down the fly. Real surface skin stays.
    """
    return faces[~(np.abs(verts[faces][..., 1]) < tol).all(axis=1)]


def feature_edges(verts: np.ndarray, faces: np.ndarray, angle_deg: float = DEFAULT_ANGLE_DEG) -> np.ndarray:
    """Edges worth drawing: boundary / non-manifold edges and creases whose face normals differ by > angle_deg."""
    if len(faces) == 0:
        return np.zeros((0, 2, 3))
    p0, p1, p2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    e.sort(axis=1)
    key = e[:, 0] * len(verts) + e[:, 1]
    order = np.argsort(key, kind="stable")
    key, e, owner = key[order], e[order], owner[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    counts = np.diff(np.r_[starts, len(key)])
    keep = counts != 2  # open (1) and non-manifold (3+) edges always stay
    pair = np.flatnonzero(counts == 2)
    a, b = owner[starts[pair]], owner[starts[pair] + 1]
    cos = np.einsum("ij,ij->i", normals[a], normals[b])
    keep[pair] = cos < np.cos(np.radians(angle_deg))
    chosen = e[starts[keep]]
    return np.stack([verts[chosen[:, 0]], verts[chosen[:, 1]]], axis=1)


# ---- mirror, quantize, encode -------------------------------------------------------------


def mirror_y(segments: np.ndarray) -> np.ndarray:
    out = np.array(segments, dtype=np.float64, copy=True)
    out[..., 1] *= -1
    return out


def mirror_parts(parts: list[Part]) -> list[Part]:
    """Central parts once, then the left parts, then their right-hand mirror images (l.. -> r..)."""
    left = [p for p in parts if p.side == "l"]
    right = [Part("r" + p.name[1:], p.group, "r", mirror_y(p.segments)) for p in left]
    return [p for p in parts if p.side == "c"] + left + right


def quantize(points: np.ndarray, lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    q = np.rint((np.asarray(points, dtype=np.float64) - lo) / (hi - lo) * 65535)
    return np.clip(q, 0, 65535).astype("<u2")


def dequantize(q: np.ndarray, lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return lo + np.asarray(q, dtype=np.float64) / 65535 * (hi - lo)


def encode_segments(parts: list[Part], lo, hi):
    """-> (blob, [(segment_start, segment_count) per part])."""
    chunks, index, start = [], [], 0
    for p in parts:
        segments = np.asarray(p.segments, dtype=np.float64)
        if segments.size == 0:  # an empty part has no segments, whatever its declared shape
            segments = segments.reshape(0, 2, 3)
        elif segments.ndim != 3 or segments.shape[1:] != (2, 3):
            raise ValueError(f"part {p.name!r}: segments must be (n, 2, 3), got shape {segments.shape}")
        chunks.append(quantize(segments, lo, hi).astype("<u2").tobytes())
        index.append((start, len(segments)))
        start += len(segments)
    return b"".join(chunks), index


def decode_segments(blob: bytes, lo, hi) -> np.ndarray:
    q = np.frombuffer(blob, dtype="<u2").reshape(-1, 2, 3)
    return dequantize(q, lo, hi)


def mirror_mesh_parts(parts: list[MeshPart]) -> list[MeshPart]:
    """Central parts once, then the left parts, then their right-hand mirror images; winding is flipped."""
    left = [p for p in parts if p.side == "l"]
    right = [MeshPart("r" + p.name[1:], "r", mirror_y(p.verts), np.ascontiguousarray(p.faces[:, ::-1])) for p in left]
    return [p for p in parts if p.side == "c"] + left + right


def surface_faces(name: str, verts: np.ndarray, faces: np.ndarray, tol: float = 2.0) -> np.ndarray:
    """The faces that belong to the visible skin: central parts lose their internal mid-plane wall."""
    return drop_midplane_faces(verts, faces, tol) if name[0] == "c" else faces


def assemble_mesh_ranges(parts: list[MeshPart], lo, hi):
    """All parts -> one indexed mesh plus each part's triangle range.

    Returns (quantized vertices (n, 3) uint16, faces (m, 3) int, [(triangle_start, triangle_count) per part]).
    Vertices are quantized first, then deduplicated across parts; triangles that collapse are dropped. Triangles
    stay in part order, so every part owns one contiguous range (a part that collapses completely gets count 0).
    """
    qv = np.concatenate([quantize(p.verts, lo, hi) for p in parts])
    offsets = np.cumsum([0] + [len(p.verts) for p in parts[:-1]])
    faces = np.concatenate([p.faces + o for p, o in zip(parts, offsets)])
    owner = np.concatenate([np.full(len(p.faces), i) for i, p in enumerate(parts)])
    uniq, inverse = np.unique(qv, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    counts = np.bincount(owner[ok], minlength=len(parts))
    starts = np.cumsum(np.r_[0, counts[:-1]])
    return uniq.astype("<u2"), faces[ok].astype(np.int64), [(int(a), int(c)) for a, c in zip(starts, counts)]


def assemble_mesh(parts: list[MeshPart], lo, hi):
    """All parts -> one indexed mesh: (quantized vertices (n, 3) uint16, faces (m, 3) int); see assemble_mesh_ranges."""
    qv, faces, _ = assemble_mesh_ranges(parts, lo, hi)
    return qv, faces


def encode_mesh(qverts: np.ndarray, faces: np.ndarray):
    """-> (blob, info): vertices as uint16 x3, then triangle indices (uint16 if they fit, else uint32)."""
    if len(faces) and (faces.min() < 0 or faces.max() >= len(qverts)):
        raise ValueError("mesh face index out of range")
    dtype = "uint16" if len(qverts) <= 65535 else "uint32"
    blob = qverts.astype("<u2").tobytes() + faces.astype("<u2" if dtype == "uint16" else "<u4").tobytes()
    info = {"vertex_count": len(qverts), "triangle_count": len(faces), "index_dtype": dtype, "bytes": len(blob)}
    return blob, info


def decode_mesh(blob: bytes, info: dict, lo, hi):
    """-> (vertices (n, 3) float um, faces (m, 3) int)."""
    n, m = info["vertex_count"], info["triangle_count"]
    idx = "<u2" if info["index_dtype"] == "uint16" else "<u4"
    q = np.frombuffer(blob, dtype="<u2", count=n * 3).reshape(-1, 3)
    faces = np.frombuffer(blob, dtype=idx, count=m * 3, offset=n * 6).reshape(-1, 3).astype(np.int64)
    return dequantize(q, lo, hi), faces


# ---- rigging, pose, kinematics ------------------------------------------------------------

_NUM = r"[-+0-9.eE]+"


def parse_rigging(text: str) -> dict:
    """rigging.yaml subset -> {body: (pos_um (3,), quat_wxyz (4,))}. pos is relative to the parent body."""
    out, name = {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            name = line.rstrip(":").strip()
            out[name] = {}
            continue
        m = re.match(rf"\s+(pos|quat):\s*\[({_NUM}(?:,\s*{_NUM})*)\]", line)
        if m:
            out[name][m.group(1)] = np.array([float(v) for v in m.group(2).split(",")])
    return {n: (v["pos"] * 1000.0, v["quat"]) for n, v in out.items() if "pos" in v and "quat" in v}


def parse_pose(text: str):
    """Pose spec -> (axis_order list, {"parent-child-axis": angle in radians})."""
    order, angles, unit, in_angles = None, {}, "radian", False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("angle_unit:"):
            unit = line.split(":", 1)[1].strip()
        elif line.startswith("axis_order:"):
            order = [s.strip() for s in line.split(":", 1)[1].strip().strip("[]").split(",")]
        elif line.startswith("joint_angles:"):
            in_angles = True
        elif in_angles and line.startswith(" "):
            key, val = line.split(":", 1)
            angles[key.strip()] = float(val)
        else:
            in_angles = False
    if order is None:
        raise ValueError("pose spec has no axis_order")
    if unit == "degree":
        angles = {k: float(np.radians(v)) for k, v in angles.items()}
    return order, angles


def quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def axis_rotation(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def body_transforms(rigging: dict, parent: dict, order: list, angles: dict) -> dict:
    """World (R, t_um) of the root and every body in `parent`.

    child = parent_pose * T(pos) * R(quat) * Rj1 * Rj2 * Rj3: the hinge rotations of one joint are applied in
    the pose's axis order, each about the already-rotated axis (MuJoCo sequential hinge semantics).
    """
    cache = {}

    def world(name):
        if name in cache:
            return cache[name]
        pos, quat = rigging[name]
        if name == ROOT:
            cache[name] = (quat_matrix(quat), np.asarray(pos, dtype=np.float64))
            return cache[name]
        Rp, tp = world(parent[name])
        R = Rp @ quat_matrix(quat)
        for ax in order:
            R = R @ axis_rotation(AXIS[ax], angles.get(f"{parent[name]}-{name}-{ax}", 0.0))
        cache[name] = (R, tp + Rp @ pos)
        return cache[name]

    for name in [ROOT, *parent]:
        world(name)
    return cache


# ---- rig: animatable joints ---------------------------------------------------------------

RIG_FRAME = "fly frame as in the asset: x anterior, y left, z dorsal"
GEOMETRY = "geometry (our choice)"
# pivot = the part's own frame origin in the neutral pose (rigging.yaml pos/quat composed down the tree: the
# socket where flygym attaches it); the hinge DIRECTION is our choice because rigging.yaml stores no axes.
LEFT_JOINTS = [
    # id, left parts, axis (x anterior, y left, z dorsal), range in degrees
    ("antenna_l", ["l_pedicel", "l_funiculus", "l_arista"], (0.0, 0.0, 1.0), (-30.0, 30.0)),  # sweep about dorsal
    ("wing_l", ["l_wing"], (0.0, 1.0, 0.0), (0.0, 60.0)),  # raise: lateral axis, the folded wing lies along -x
    ("haltere_l", ["l_haltere"], (1.0, 0.0, 0.0), (-30.0, 30.0)),  # beat about the antero-posterior axis
]
PIVOT_BODY = {"antenna_l": "l_pedicel", "wing_l": "l_wing", "haltere_l": "l_haltere"}
PROBOSCIS_AXIS = (0.0, -1.0, 0.0)  # positive angle swings the hanging proboscis forward (extension)
PROBOSCIS_RANGE = (0.0, 60.0)


def _clean(vec) -> list:
    return [float(v) + 0.0 for v in np.round(np.asarray(vec, dtype=np.float64), 6)]


def mirror_joint(joint: dict) -> dict:
    """Right-hand twin of a left joint: pivot y flips like the mesh mirror; the axis, a rotation axis
    (pseudovector), turns by -M so that the same angle moves both sides as mirror images."""
    ax = np.asarray(joint["axis"], dtype=np.float64)
    parts = ["r" + n[1:] for n in joint["parts"]]
    px, py, pz = joint["pivot_um"]
    return {**joint, "id": joint["id"][:-1] + "r", "parts": parts, "pivot_um": [px, -py + 0.0, pz],
            "axis": _clean([-ax[0], ax[1], -ax[2]])}


def build_rig(origins: dict, rostrum_verts: np.ndarray) -> dict:
    """Joints of the animatable body, in the asset frame and units.

    `origins` maps the left/central body names (c_rostrum, l_pedicel, l_wing, l_haltere) to their world frame
    origin in micrometres; `rostrum_verts` are the world vertices of c_rostrum. Right-hand joints are mirrored.
    """
    joints = []
    rv = np.asarray(rostrum_verts, dtype=np.float64)
    z = rv[:, 2]
    top = rv[z >= z.max() - 0.05 * (z.max() - z.min())]  # the dorsal end of the rostrum, where it joins the head
    base = np.round([top[:, 0].mean(), 0.0, top[:, 2].mean()], 1)
    joints.append({"id": "proboscis", "parts": ["c_rostrum", "c_haustellum"], "pivot_um": _clean(base),
                   "axis": _clean(PROBOSCIS_AXIS), "rest_deg": 0, "range_deg": list(PROBOSCIS_RANGE), "source": GEOMETRY})
    left = []
    for jid, parts, axis, rng in LEFT_JOINTS:
        pivot = np.round(np.asarray(origins[PIVOT_BODY[jid]], dtype=np.float64), 1)
        left.append({"id": jid, "parts": parts, "pivot_um": _clean(pivot), "axis": _clean(axis), "rest_deg": 0,
                     "range_deg": list(rng), "source": GEOMETRY})
    by = {j["id"]: j for j in left}
    for kind in ("antenna", "wing", "haltere"):
        joints += [by[f"{kind}_l"], mirror_joint(by[f"{kind}_l"])]
    # legs are not rigged: each leg is a chain of 8 links with 2-3 hinge axes per link, not one hinge
    return {"units": "um", "frame": RIG_FRAME, "joints": joints}


def rotate_about(points: np.ndarray, pivot, axis, angle_deg: float) -> np.ndarray:
    """Rotate points by angle_deg (right-handed) about the unit direction of `axis` through `pivot`."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    pivot = np.asarray(pivot, dtype=np.float64)
    a = np.radians(angle_deg)
    v = np.asarray(points, dtype=np.float64) - pivot
    out = v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * (v @ k)[..., None] * (1 - np.cos(a))
    return out + pivot


def body_origins(wheel) -> dict:
    """World frame origin (micrometres, neutral pose) of every body in the kinematic tree."""
    with zipfile.ZipFile(wheel) as z:
        rigging = parse_rigging(z.read(RIGGING_MEMBER).decode())
        order, angles = parse_pose(z.read(POSE_MEMBER).decode())
    return {n: t for n, (_, t) in body_transforms(rigging, PARENT, order, angles).items()}


# ---- build --------------------------------------------------------------------------------


def verify_sha256(path, expected: str) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {digest}; refusing to use it")


def build_parts(wheel, angle_deg: float = DEFAULT_ANGLE_DEG) -> list[Part]:
    """Central + left parts in the neutral pose, world frame, micrometres (right side not yet added)."""
    return build_parts_and_meshes(wheel, angle_deg)[0]


def build_parts_and_meshes(wheel, angle_deg: float = DEFAULT_ANGLE_DEG):
    """Like build_parts, plus the welded surface of every part (mid-plane faces already dropped)."""
    with zipfile.ZipFile(wheel) as z:
        rigging = parse_rigging(z.read(RIGGING_MEMBER).decode())
        order, angles = parse_pose(z.read(POSE_MEMBER).decode())
        transforms = body_transforms(rigging, PARENT, order, angles)
        parts, meshes = [], []
        for name in PART_ORDER:
            R, t = transforms[name]
            tris = parse_stl(z.read(f"{MESH_DIR}{name}.stl")) * UM_PER_M
            verts, faces = weld(tris @ R.T + t)  # world frame first: the midplane is y = 0 there
            faces = surface_faces(name, verts, faces)
            parts.append(Part(name, group_of(name), name[0], feature_edges(verts, faces, angle_deg)))
            meshes.append(MeshPart(name, name[0], verts, faces))
    return parts, meshes


def write_outputs(out_dir, mesh_blob: bytes, blob: bytes, meta_text: str) -> None:
    """Write fly-body-mesh.bin, fly-body.bin and fly-body.json so an interruption never leaves a mixed set.

    Every file is staged as a temp file in the same directory first (a failure here touches nothing), then
    moved into place with os.replace, the json last: the json is what describes the binaries.
    """
    out_dir = Path(out_dir)
    final = [(out_dir / "fly-body-mesh.bin", mesh_blob), (out_dir / "fly-body.bin", blob),
             (out_dir / "fly-body.json", meta_text.encode("utf-8"))]
    staged = []
    try:
        for path, data in final:
            tmp = path.with_name(path.name + ".tmp")
            staged.append(tmp)
            tmp.write_bytes(data)
        for (path, _), tmp in zip(final, staged):
            os.replace(tmp, path)
    finally:
        for tmp in staged:
            tmp.unlink(missing_ok=True)


def build(wheel, out_dir=DEFAULT_OUT_DIR, angle_deg: float = DEFAULT_ANGLE_DEG, expected_sha256: str = WHEEL_SHA256) -> dict:
    wheel, out_dir = Path(wheel), Path(out_dir)
    verify_sha256(wheel, expected_sha256)
    raw_parts, raw_meshes = build_parts_and_meshes(wheel, angle_deg)
    parts = mirror_parts(raw_parts)
    allseg = np.concatenate([p.segments for p in parts])
    lo, hi = allseg.reshape(-1, 3).min(axis=0), allseg.reshape(-1, 3).max(axis=0)
    lo, hi = np.round(lo - 0.5, 1), np.round(hi + 0.5, 1)  # fixed padding, decoder-visible
    blob, index = encode_segments(parts, lo, hi)
    head = next(p for p in parts if p.name == "c_head").segments.reshape(-1, 3)
    meta = {
        "units": "um",
        "axes": "x anterior(+)/posterior(-), y left(+)/right(-), z dorsal(+)/ventral(-); origin = NeuroMechFly world "
        "origin of the neutral pose (not the fly's centre)",
        "bounds": [lo.tolist(), hi.tolist()],
        "quantization": "uint16 per axis over bounds: um = min + q / 65535 * (max - min)",
        "format": "fly-body.bin: little-endian uint16 x0 y0 z0 x1 y1 z1 per segment (12 bytes), parts concatenated",
        "feature_edge_angle_deg": angle_deg,
        "parts": [
            {"name": p.name, "group": p.group, "side": p.side, "segment_start": s, "segment_count": c}
            for p, (s, c) in zip(parts, index)
        ],
        "head_center_um": np.round((head.min(axis=0) + head.max(axis=0)) / 2, 1).tolist(),
        "brain_note": "brain bbox about 627 x 293 x 173 um (JRC2018 template, not measured here)",
        "source": {
            "name": "NeuroMechFly v2 / flygym",
            "version": FLYGYM_VERSION,
            "sha256": expected_sha256,
            "license": "Apache-2.0",
            "url": WHEEL_URL,
        },
        "note": "static micro-CT mesh of an adult female; approximate scale; not animated",
    }
    mesh_parts = mirror_mesh_parts(raw_meshes)
    qverts, mfaces, ranges = assemble_mesh_ranges(mesh_parts, lo, hi)
    mesh_blob, mesh_info = encode_mesh(qverts, mfaces)
    meta["mesh"] = {
        "file": "fly-body-mesh.bin",
        "format": "little-endian: vertex_count x (uint16 x y z, same bounds and quantization as fly-body.bin), then "
        "triangle_count x 3 vertex indices of index_dtype; depth-only occluder for hidden-line drawing",
        **mesh_info,
        "parts": [
            {"name": p.name, "triangle_start": a, "triangle_count": c} for p, (a, c) in zip(mesh_parts, ranges)
        ],
        "parts_note": "triangles are concatenated in part order (same order as `parts`); each part owns the "
        "contiguous triangle range [triangle_start, triangle_start + triangle_count)",
    }
    rostrum = next(m for m in raw_meshes if m.name == "c_rostrum")
    meta["rig"] = build_rig(body_origins(wheel), rostrum.verts)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, mesh_blob, blob, json.dumps(meta, indent=1) + "\n")
    with zipfile.ZipFile(wheel) as z:
        (out_dir / "LICENSE-neuromechfly.txt").write_bytes(z.read(LICENSE_MEMBER))
    return {"segments": len(allseg), "bytes": len(blob), "parts": len(parts), "bounds": meta["bounds"], "mesh": mesh_info}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build the fly body line-segment asset from the flygym wheel.")
    p.add_argument("--wheel", required=True, help=f"{WHEEL_NAME} downloaded from {WHEEL_URL}")
    p.add_argument("--out", default=DEFAULT_OUT_DIR)
    p.add_argument("--angle", type=float, default=DEFAULT_ANGLE_DEG, help="feature-edge angle threshold in degrees")
    args = p.parse_args(argv)
    try:
        stats = build(args.wheel, args.out, args.angle)
    except ValueError as e:
        p.error(str(e))
    print(f"{stats['parts']} parts, {stats['segments']:,} segments, {stats['bytes'] / 1e3:.0f} KB + mesh {stats['mesh']['bytes'] / 1e3:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
