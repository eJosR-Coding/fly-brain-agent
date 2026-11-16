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
"""

import argparse
import hashlib
import json
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
    return np.frombuffer(data, dtype=rec, count=n, offset=84)["v"].astype(np.float64)


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
        chunks.append(quantize(p.segments, lo, hi).astype("<u2").tobytes())
        index.append((start, len(p.segments)))
        start += len(p.segments)
    return b"".join(chunks), index


def decode_segments(blob: bytes, lo, hi) -> np.ndarray:
    q = np.frombuffer(blob, dtype="<u2").reshape(-1, 2, 3)
    return dequantize(q, lo, hi)


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


# ---- build --------------------------------------------------------------------------------


def verify_sha256(path, expected: str) -> None:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {digest}; refusing to use it")


def build_parts(wheel, angle_deg: float = DEFAULT_ANGLE_DEG) -> list[Part]:
    """Central + left parts in the neutral pose, world frame, micrometres (right side not yet added)."""
    with zipfile.ZipFile(wheel) as z:
        rigging = parse_rigging(z.read(RIGGING_MEMBER).decode())
        order, angles = parse_pose(z.read(POSE_MEMBER).decode())
        transforms = body_transforms(rigging, PARENT, order, angles)
        parts = []
        for name in PART_ORDER:
            R, t = transforms[name]
            tris = parse_stl(z.read(f"{MESH_DIR}{name}.stl")) * UM_PER_M
            verts, faces = weld(tris @ R.T + t)  # world frame first: the midplane is y = 0 there
            if name[0] == "c":
                faces = drop_midplane_faces(verts, faces)
            parts.append(Part(name, group_of(name), name[0], feature_edges(verts, faces, angle_deg)))
    return parts


def build(wheel, out_dir=DEFAULT_OUT_DIR, angle_deg: float = DEFAULT_ANGLE_DEG, expected_sha256: str = WHEEL_SHA256) -> dict:
    wheel, out_dir = Path(wheel), Path(out_dir)
    verify_sha256(wheel, expected_sha256)
    parts = mirror_parts(build_parts(wheel, angle_deg))
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
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fly-body.bin").write_bytes(blob)
    (out_dir / "fly-body.json").write_text(json.dumps(meta, indent=1) + "\n")
    with zipfile.ZipFile(wheel) as z:
        (out_dir / "LICENSE-neuromechfly.txt").write_bytes(z.read(LICENSE_MEMBER))
    return {"segments": len(allseg), "bytes": len(blob), "parts": len(parts), "bounds": meta["bounds"]}


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
    print(f"{stats['parts']} parts, {stats['segments']:,} segments, {stats['bytes'] / 1e3:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()
