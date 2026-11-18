"""BodyStream: the physics body as a live process. Pure helpers always; the physics only when flygym is importable."""

import importlib.util
import sys

import numpy as np
import pytest

from flybrain import body_runner as br

HAS_FLYGYM = all(importlib.util.find_spec(m) is not None for m in ("flygym", "flygym_demo", "mujoco"))
phys = pytest.mark.skipif(not HAS_FLYGYM, reason="flygym is an optional extra: uv sync --extra body")


# ---- pure ----

def test_continuous_with_a_previous_frame_keeps_the_hemisphere_across_chunks():
    prev = np.array([[0.9, 0.1, 0.0, 0.0]])
    flipped = np.array([[[-0.9, -0.1, 0.0, 0.0]], [[-0.8, -0.2, 0.0, 0.0]]])  # same rotations as prev, sign flipped
    out = br.continuous(flipped, prev=prev)
    assert (out[:, 0, 0] > 0).all() and np.allclose(out[0], -flipped[0])


def test_continuous_without_prev_is_unchanged():
    q = np.array([[[-1.0, 0, 0, 0]], [[-0.99, 0.1, 0, 0]]])
    assert np.allclose(br.continuous(q), br.continuous(q, prev=None)) and br.continuous(q)[0, 0, 0] > 0


def test_unwrap_continue_follows_the_heading_across_the_180_degree_seam():
    raw = np.array([170.0, 178.0, -176.0, -170.0])  # turning left through 180
    out, last_raw, last_abs = br.unwrap_continue(raw, last_raw=None, last_abs=None)
    assert out.tolist() == pytest.approx([170.0, 178.0, 184.0, 190.0]) and (last_raw, last_abs) == (-170.0, 190.0)
    more, _, _ = br.unwrap_continue(np.array([-164.0]), last_raw, last_abs)  # next chunk keeps counting
    assert more.tolist() == pytest.approx([196.0])


def test_chunk_ms_must_be_whole_frames():
    assert br.chunk_frames(50.0) == 10
    with pytest.raises(ValueError):
        br.chunk_frames(52.0)
    with pytest.raises(ValueError):
        br.chunk_frames(0)


def test_missing_flygym_raises_the_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "flygym", None)
    with pytest.raises(br.BodyUnavailable, match=r"uv sync --extra body"):
        br.BodyStream()


# ---- physics (gated) ----

def cosine(a, b):
    """|cos| between the integer quaternions of two frame arrays (unit-norm free: the scaled ints are rounded)."""
    qa, qb = a[..., 3:].astype(float), b[..., 3:].astype(float)
    return (qa * qb).sum(-1) / (np.linalg.norm(qa, axis=-1) * np.linalg.norm(qb, axis=-1))


@pytest.fixture(scope="module")
def runs():
    """One stream in 10 x 50 ms chunks, one in a single 500 ms advance, same drive: the data the tests compare."""
    chunked = br.BodyStream(seed=0)
    chunks = [chunked.advance(50.0, [1.0, 1.0], False) for _ in range(10)]
    whole = br.BodyStream(seed=0).advance(500.0, [1.0, 1.0], False)
    return chunked, chunks, whole


@phys
def test_chunk_shapes_and_frame_format(runs):
    stream, chunks, _ = runs
    c = chunks[0]
    assert c["n_frames"] == 10 and c["frames"].shape == (10, 48, 7) and c["frames"].dtype.kind == "i"
    assert c["thorax"]["pos_um"].shape == (10, 3) and c["thorax"]["yaw_deg"].shape == (10,)
    assert c["escape"].tolist() == [0] * 10 and c["drive"].shape == (10, 2)
    assert stream.t_ms == pytest.approx(500.0)
    h = stream.header
    assert h["format"] == "fly-walk/1" and h["sample_rate_hz"] == 200 and len(h["bodies"]) == 48
    assert len(h["neutral"]["pos_um"]) == 48 and len(h["neutral"]["quat"]) == 48 and h["quat_scale"] == 10000


@phys
def test_concatenated_chunks_equal_one_long_advance(runs):
    _, chunks, whole = runs
    frames = np.concatenate([c["frames"] for c in chunks])
    assert frames.shape == whole["frames"].shape
    assert np.abs(frames[..., :3] - whole["frames"][..., :3]).max() <= 2  # micrometres
    assert np.abs(cosine(frames, whole["frames"])).min() > 0.99999  # same rotations (sign-insensitive)
    tp = np.concatenate([c["thorax"]["pos_um"] for c in chunks])
    assert np.abs(tp - whole["thorax"]["pos_um"]).max() <= 2
    yaw = np.concatenate([c["thorax"]["yaw_deg"] for c in chunks])
    assert np.abs(yaw - whole["thorax"]["yaw_deg"]).max() < 0.05
    assert np.array_equal(np.concatenate([c["contact"] for c in chunks]), whole["contact"])


@phys
def test_the_stream_matches_run_body_within_tolerance(runs):
    _, _, whole = runs
    clip = br.run_body([[1.0, 1.0]], 500.0, 500.0, seed=0)
    ref = np.array(clip["frames"])
    assert ref.shape == whole["frames"].shape
    assert np.abs(ref[..., :3] - whole["frames"][..., :3]).max() <= 3
    assert np.abs(cosine(ref, whole["frames"])).min() > 0.99999
    assert np.array(clip["thorax_world"]["pos_um"])[-1, 0] == pytest.approx(whole["thorax"]["pos_um"][-1, 0], abs=3)
    assert whole["thorax"]["pos_um"][-1, 0] > 100  # it walked forward


@phys
def test_drive_can_change_between_chunks_and_escape_stops_the_legs():
    s = br.BodyStream(seed=0)
    s.advance(100.0, [1.0, 1.0], False)
    turn = s.advance(200.0, [0.4, 1.2], False)  # lower LEFT drive: turns left (positive yaw)
    assert turn["thorax"]["yaw_deg"][-1] - turn["thorax"]["yaw_deg"][0] > 1.0
    assert (turn["drive"] == [0.4, 1.2]).all()
    esc = s.advance(50.0, [1.0, 1.0], True)
    assert esc["escape"].tolist() == [1] * 10 and (esc["drive"] == 0).all()


@phys
def test_bad_chunk_is_rejected_and_leaves_the_stream_usable():
    s = br.BodyStream(seed=0)
    with pytest.raises(ValueError):
        s.advance(52.0, [1, 1], False)
    assert s.advance(50.0, [1, 1], False)["n_frames"] == 10
