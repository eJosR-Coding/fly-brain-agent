"""A tiny static picture of the brain: orthographic projection at the viewer's home angle, depth-shaded.

Wire format: 4-byte little-endian header [width uint16, height uint16], then width*height uint8
brightness, row-major, top row first (0 = empty). Pure numpy.
"""

from __future__ import annotations

import struct

import numpy as np

# Keep in sync with HOME_VIEW in web/src/scene.ts (the 3D home camera angle).
HOME_AZIMUTH_DEG = 16
HOME_ELEVATION_DEG = 10

LONG_SIDE = 384  # cells on the longer side, padding included
PAD = 4  # empty cells on every edge
_GAMMA = 0.6  # lower = brighter midtones; 384 cells with 0.6 compared best against 256/0.8 and the noisier 512 grids


def _camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, toward_camera) in view space; mirrors homeCamera() in web/src/anatomy.ts."""
    az, el = np.radians(HOME_AZIMUTH_DEG), np.radians(HOME_ELEVATION_DEG)
    toward = np.array([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)])
    right = np.cross(-toward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -toward)
    return right, up, toward


def render_preview(xyz: np.ndarray) -> tuple[int, int, bytes]:
    """(n, 3) atlas positions -> (width, height, header + brightness bytes)."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return 1, 1, struct.pack("<HH", 1, 1) + b"\x00"
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    view = (pts - center) * np.array([1.0, -1.0, 1.0])  # the viewer flips y (FlyWire y grows downward)
    right, up, toward = _camera_basis()
    u, v, depth = view @ right, view @ up, view @ toward

    inner = LONG_SIDE - 2 * PAD
    span_u, span_v = np.ptp(u), np.ptp(v)
    longest = max(span_u, span_v)
    scale = (inner - 1) / longest if longest > 0 else 0.0
    cw = int(np.floor(span_u * scale)) + 1
    ch = int(np.floor(span_v * scale)) + 1
    width, height = cw + 2 * PAD, ch + 2 * PAD

    ix = np.minimum(np.floor((u - u.min()) * scale).astype(np.int64), cw - 1)
    iy = np.minimum(np.floor((v - v.min()) * scale).astype(np.int64), ch - 1)
    cell = (PAD + ch - 1 - iy) * width + PAD + ix  # top row first: larger v is higher up

    counts = np.bincount(cell, minlength=width * height).astype(np.float64)
    span_d = np.ptp(depth)
    near = (depth - depth.min()) / span_d if span_d > 0 else np.full(len(pts), 0.5)
    near_sum = np.bincount(cell, weights=near, minlength=width * height)
    mean_near = np.divide(near_sum, counts, out=np.zeros_like(near_sum), where=counts > 0)

    density = np.log1p(counts) / np.log1p(counts.max())
    bright = 255.0 * density**_GAMMA * (0.55 + 0.45 * mean_near)
    grid = np.where(counts > 0, np.clip(np.rint(bright), 1, 255), 0).astype(np.uint8)
    return width, height, struct.pack("<HH", width, height) + grid.tobytes()
