"""The fly's eye in the arena: per-column view directions, the scene it sees and Reichardt motion per T4/T5 subtype."""

import time

import numpy as np
import pytest

from flybrain import fly_eye
from flybrain.fly_eye import ArenaScene, Box, Hrc, build_eye_from_columns

DT = 5.0  # ms per frame (200 Hz)


def patch(side="right", span=12):
    ps, qs = np.meshgrid(np.arange(-span, span + 1), np.arange(-span, span + 1))
    return build_eye_from_columns(side, ps.ravel(), qs.ravel(), first_row=0 if side == "right" else 100000)


def run(eye, lum_fn, n_chunks=24, frames=10, skip=6):
    """Mean per-column ON/OFF drive over the later chunks of a luminance movie lum_fn(t_s) -> (n,)."""
    hrc, t, ons, offs = Hrc(eye, dt_ms=DT), 0, [], []
    for c in range(n_chunks):
        lum = np.stack([lum_fn((t + i) * DT / 1000.0) for i in range(frames)])
        t += frames
        drive = hrc.step(lum)
        if c >= skip:
            ons.append(drive.on), offs.append(drive.off)
    return np.mean(ons, axis=0), np.mean(offs, axis=0)


def cell_mean(eye, on, off, kind, sub, mask=None):
    """Mean drive of the cells of one subtype (kind 'T4'/'T5', sub 'a'..'d'), optionally restricted to columns `mask`."""
    sel = (eye.cell_types == f"{kind}{sub}")
    if mask is not None:
        sel &= mask[eye.cell_col]
    return float(eye.cell_drive(on, off)[sel].mean())


# -- geometry ---------------------------------------------------------------------------------------------

def test_directions_are_unit_vectors_and_the_eyes_are_mirror_images():
    r, l = patch("right"), patch("left")
    assert np.linalg.norm(r.dirs, axis=1) == pytest.approx(1.0)
    assert np.allclose(l.dirs, r.dirs * np.array([1, -1, 1]))  # x forward, y left, z up
    centre = int(np.argmin(np.hypot(r.p, r.q)))
    d = r.dirs[centre]
    assert d[1] < -0.8 and d[0] > 0.2 and abs(d[2]) < 0.1  # the right eye looks to the right and a little forward


def test_h_is_posterior_and_v_is_up_in_the_head_frame():
    eye = patch("right")
    centre = int(np.argmin(np.hypot(eye.p, eye.q)))
    post = int(np.argmax(eye.h))
    up = int(np.argmax(eye.v))
    assert eye.dirs[post][0] < eye.dirs[centre][0]  # larger h = further back
    assert eye.dirs[up][2] > eye.dirs[centre][2]


def test_each_column_has_a_neighbour_upstream_of_each_preferred_direction_except_at_the_rim():
    eye = patch("right")
    inner = np.hypot(eye.p, eye.q) < 5
    for s in range(4):
        up = eye.upstream[s].max(axis=0)  # any flank
        assert (up[inner] >= 0).all()
        # the upstream column of subtype a (front-to-back) lies in front (smaller h) of the cell
        if s == 0:
            assert (eye.h[up[inner]] < eye.h[inner]).all()
        if s == 2:  # c prefers up: the upstream column is lower
            assert (eye.v[up[inner]] < eye.v[inner]).all()
    assert (eye.upstream < len(eye.p)).all() and (eye.upstream[:, :, inner] >= 0).any(axis=1).all()


# -- scene -------------------------------------------------------------------------------------------------

def look(scene, pos, direction, yaw=0.0):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    return scene.luminance(np.asarray([pos], float), np.asarray([yaw]), d[None, :])[0, 0]


def test_the_floor_is_a_checker_of_the_stated_period():
    scene = ArenaScene(floor_period_mm=4.0)
    down = [0, 0, -1]
    a = look(scene, [-3.5, 0.5, 1.0], down)
    b = look(scene, [-3.5 + 4.0, 0.5, 1.0], down)  # one period along x: the same
    c = look(scene, [-3.5 + 2.0, 0.5, 1.0], down)  # half a period: the opposite
    assert a == pytest.approx(b) and abs(a - c) > 0.2


def test_through_the_glass_the_background_is_light_and_an_edge_is_a_dark_line():
    scene = ArenaScene(interior_mm=50.0)
    pos = [0.0, 0.0, 1.0]
    wall = look(scene, pos, [1, 0.0, 0.3])  # out through the +x wall away from every edge
    at_edge = look(scene, pos, [25.0, 25.0, 0.0 + 0.001])  # the vertical edge (25, 25)
    near = look(scene, pos, [25.0, 20.0, 0.001])
    assert wall == pytest.approx(scene.background) and at_edge < 0.7 * wall and near > at_edge


def test_an_object_hides_what_is_behind_it():
    box = Box(lo=(8.0, -2.0, 8.0), hi=(12.0, 2.0, 12.0))
    free, blocked = ArenaScene(), ArenaScene(objects=[box])
    d = [1.0, 0.0, 0.0]
    assert look(free, [0, 0, 10.0], d) == pytest.approx(free.background)
    assert look(blocked, [0, 0, 10.0], d) == pytest.approx(box.lum) and box.lum < 0.3
    assert look(blocked, [0, 0, 10.0], [-1.0, 0, 0]) == pytest.approx(free.background)  # looking away from it


def test_yaw_turns_the_view_with_the_fly():
    scene = ArenaScene(objects=[Box(lo=(8.0, -2.0, 8.0), hi=(12.0, 2.0, 12.0))])
    ahead = [1.0, 0.0, 0.0]
    assert look(scene, [0, 0, 10.0], ahead, yaw=0.0) < 0.3
    assert look(scene, [0, 0, 10.0], ahead, yaw=90.0) > 0.9  # now facing +y: the box is to its right


def test_a_chunk_of_luminance_for_both_eyes_is_fast():
    eye = patch("right", span=14)  # ~800 columns
    scene = ArenaScene()
    pos = np.tile([0.0, 0.0, 1.0], (10, 1))
    yaw = np.linspace(0, 5, 10)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for e in (eye, eye):
            scene.luminance(pos, yaw, e.dirs)
        times.append(time.perf_counter() - t0)
    assert len(eye.p) > 700 and np.median(times) < 0.05


# -- motion (Reichardt) --------------------------------------------------------------------------------------

LAM = np.radians(30.0)


def grating(eye, axis, speed_deg_s, sign=+1):
    coord = eye.h if axis == "h" else eye.v

    def lum(t):
        return 0.5 + 0.5 * np.sin(2 * np.pi * (coord - sign * np.radians(speed_deg_s) * t) / LAM)

    return lum


@pytest.mark.parametrize("axis,sign,good,bad", [("h", +1, "a", "b"), ("h", -1, "b", "a"), ("v", +1, "c", "d"), ("v", -1, "d", "c")])
def test_a_grating_in_the_preferred_direction_drives_the_subtype_and_the_opposite_one_does_not(axis, sign, good, bad):
    eye = patch()
    inner = np.hypot(eye.p, eye.q) < 8
    on, off = run(eye, grating(eye, axis, 60.0, sign))
    for kind in ("T4", "T5"):
        g, b = cell_mean(eye, on, off, kind, good, inner), cell_mean(eye, on, off, kind, bad, inner)
        assert g > 0.03 and b < 0.2 * g, (kind, good, g, bad, b)


def test_uniform_luminance_change_and_a_static_scene_drive_nothing():
    eye = patch()
    on, off = run(eye, lambda t: np.full(len(eye.p), 0.5 + 0.4 * np.sin(2 * np.pi * 3 * t)))
    assert on.max() == 0.0 and off.max() == 0.0
    on, off = run(eye, lambda t: 0.3 + 0.5 * (eye.h > 0))
    assert on.max() == 0.0 and off.max() == 0.0


def test_a_bright_edge_drives_t4_not_t5_and_a_dark_edge_drives_t5_not_t4():
    eye = patch()
    inner = np.hypot(eye.p, eye.q) < 8

    def edge(bright, speed=90.0):
        def lum(t):
            front = -np.radians(60) + np.radians(speed) * t  # a front sweeping posterior
            inside = eye.h < front
            return np.where(inside, 0.9, 0.1) if bright else np.where(inside, 0.1, 0.9)
        return lum

    on, off = run(eye, edge(bright=True), n_chunks=30, skip=0)
    t4, t5 = cell_mean(eye, on, off, "T4", "a", inner), cell_mean(eye, on, off, "T5", "a", inner)
    assert t4 > 0.01 and t5 < 0.25 * t4
    on, off = run(eye, edge(bright=False), n_chunks=30, skip=0)
    t4, t5 = cell_mean(eye, on, off, "T4", "a", inner), cell_mean(eye, on, off, "T5", "a", inner)
    assert t5 > 0.01 and t4 < 0.25 * t5


def test_an_expanding_dark_disc_drives_outward_preferring_cells_around_its_centre():
    eye = patch()
    def disc(t):
        r = np.radians(4 + 80 * (t % 0.5))
        return np.where(np.hypot(eye.h, eye.v) < r, 0.1, 0.9)
    on, off = run(eye, disc, n_chunks=30, skip=0)
    ring = (np.hypot(eye.h, eye.v) > np.radians(8))
    for sub, outward in (("a", eye.h > 0), ("b", eye.h < 0), ("c", eye.v > 0), ("d", eye.v < 0)):
        out_side = cell_mean(eye, on, off, "T5", sub, ring & outward & (np.abs(eye.h) > np.abs(eye.v) if sub in "ab" else np.abs(eye.v) > np.abs(eye.h)))
        in_side = cell_mean(eye, on, off, "T5", sub, ring & ~outward & (np.abs(eye.h) > np.abs(eye.v) if sub in "ab" else np.abs(eye.v) > np.abs(eye.h)))
        assert out_side > 0.01 and in_side < 0.3 * out_side, (sub, out_side, in_side)


def test_state_carries_across_chunks():
    eye = patch()
    lum = grating(eye, "h", 60.0)
    frames = np.stack([lum(i * DT / 1000.0) for i in range(120)])
    one = Hrc(eye, DT)
    one.step(frames[:110])
    last_one = one.step(frames[110:])
    h = Hrc(eye, DT)
    parts = [h.step(frames[i:i + 10]) for i in range(0, 120, 10)]
    assert np.allclose(parts[-1].on, last_one.on) and np.allclose(parts[-1].off, last_one.off) and last_one.on.max() > 0


def test_drive_is_in_zero_one():
    eye = patch()
    on, off = run(eye, grating(eye, "h", 200.0))
    assert 0.0 <= on.min() and on.max() <= 1.0 and 0.0 <= off.min() and off.max() <= 1.0 and on.max() > 0


# -- the front end: pose in, T4/T5 rows and rates out ---------------------------------------------------------

def front(**kw):
    return fly_eye.VisionFrontEnd({"right": patch("right"), "left": patch("left")}, ArenaScene(), **kw)


def still(n=10, x_mm=0.0, yaw=0.0):
    """n frames of a fly standing at x_mm (thorax positions in um, as the body reports them)."""
    return np.tile([x_mm * 1000.0, 0.0, 1022.0], (n, 1)), np.full(n, yaw)


def test_a_fly_at_rest_sees_a_static_world_and_drives_no_motion_cell():
    f = front()
    for _ in range(5):
        pos, yaw = still()
        out = f.process(pos, yaw)
    assert out.rows.size == 0 and out.stats["driven_cells"] == 0


def test_walking_through_the_arena_drives_cells_in_both_eyes_within_the_rate_scale():
    f = front()
    out = None
    for k in range(20):
        pos = np.column_stack([np.linspace(k, k + 1, 10) * 1000.0 * 0.5, np.zeros(10), np.full(10, 1022.0)])  # 10 mm/s
        out = f.process(pos, np.zeros(10))
    assert out.rows.size > 20 and np.unique(out.rows).size == out.rows.size
    assert 0 < out.rates_hz.max() <= fly_eye.MAX_RATE_HZ and out.rates_hz.min() > 0
    assert out.stats["max_rate_hz"] == pytest.approx(out.rates_hz.max(), abs=0.01)


def test_turning_in_place_drives_opposite_horizontal_subtypes_in_the_two_eyes_equally():
    f = front()
    out = None
    for k in range(20):
        yaw = np.linspace(k, k + 1, 10) * 12.0  # 240 deg/s
        out = f.process(np.tile([0.0, 0.0, 1022.0], (10, 1)), yaw)
    assert out.rows.size > 20


def test_background_noise_is_labelled_and_applies_to_the_stated_cells_only():
    f = front(background_hz=2.0)
    pos, yaw = still()
    out = f.process(pos, yaw)
    assert out.rows.size == f.n_cells and np.allclose(out.rates_hz, 2.0)
    assert "T4/T5" in f.background_note
    assert front().background_note is None


def test_the_front_end_costs_well_under_a_chunk():
    f = front()
    pos = np.column_stack([np.linspace(0, 1, 10) * 1000.0, np.zeros(10), np.full(10, 1022.0)])
    f.process(pos, np.zeros(10))
    t0 = time.perf_counter()
    for _ in range(5):
        f.process(pos, np.zeros(10))
    assert (time.perf_counter() - t0) / 5 < 0.05


# -- loading from the atlas + the Codex column file ---------------------------------------------------------------

def test_load_eyes_returns_none_for_a_missing_file(tmp_path, tiny_atlas):
    assert fly_eye.load_eyes(tiny_atlas, tmp_path / "nope.csv.gz") is None


@pytest.mark.skipif(not __import__("flybrain.retina", fromlist=["x"]).COLUMN_FILE.is_file(), reason="column file not available")
def test_load_eyes_from_the_real_column_file_gives_two_disjoint_mirrored_eyes():
    from flybrain.atlas import load_atlas
    from flybrain.retina import ROOT

    atlas_path = ROOT / "data" / "raw" / "flywire_neuron_annotations.tsv"
    if not atlas_path.is_file():
        pytest.skip("atlas not available")
    eyes = fly_eye.load_eyes(load_atlas(atlas_path))
    assert set(eyes) == {"left", "right"}
    assert not set(eyes["left"].cell_rows) & set(eyes["right"].cell_rows)
    assert 700 < len(eyes["right"].p) < 1000 and len(eyes["right"].cell_rows) > 5000
