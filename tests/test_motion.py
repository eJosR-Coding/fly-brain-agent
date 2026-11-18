"""Motion front end: T4/T5 retinotopy in a head frame, preferred directions and local edge motion of a disc."""

import numpy as np
import pytest

from flybrain import eye, motion

# head frame: h grows toward posterior (+z in FlyWire), v grows upward (-y in FlyWire)
GRID = [(h, v) for h in (-2, -1, 0, 1, 2) for v in (-2, -1, 0, 1, 2) if (h, v) != (0, 0)]


def head_eye(x_offset=0.0, kind="a", framed=True):
    """24 cells of one subtype on a 5x5 grid (centre left out) lying in the y-z plane (depth along x)."""
    pos = [(x_offset, -v * 1000.0, h * 1000.0) for h, v in GRID]  # up = -y, posterior = +z
    emap = eye.EyeMap(np.arange(len(GRID)), pos)
    return (emap.head_frame() if framed else emap), np.array([f"T4{kind}"] * len(GRID))


def rows_at(emap, h, v):
    i = GRID.index((h, v))
    return int(emap.rows[i])


def segment_rates(schedule):
    """row -> summed rate over the whole schedule."""
    out: dict[int, float] = {}
    for _, _, rows, rate in schedule.segments:
        for r in rows:
            out[int(r)] = out.get(int(r), 0.0) + rate
    return out


def test_preferred_directions_follow_maisak_2013():
    # Maisak et al. 2013 (Nature 500:212): a front-to-back, b back-to-front, c upward, d downward
    assert motion.PREFERRED == {"a": (1.0, 0.0), "b": (-1.0, 0.0), "c": (0.0, 1.0), "d": (0.0, -1.0)}


@pytest.mark.parametrize("x_offset", [0.0, 100000.0, -100000.0])  # left eye / right eye: same head frame
def test_head_frame_puts_posterior_at_plus_h_and_up_at_plus_v(x_offset):
    emap, _ = head_eye(x_offset, framed=False)
    framed = emap.head_frame()
    xy = {GRID[i]: framed.xy[i] for i in range(len(GRID))}
    assert xy[(2, 0)][0] > xy[(-2, 0)][0] and xy[(0, 2)][1] > xy[(0, -2)][1]
    assert abs(xy[(2, 0)][1]) < 1e-9 and abs(xy[(0, 2)][0]) < 1e-9  # axes line up with the grid


def test_head_frame_is_independent_of_the_pca_axis_signs():
    emap, _ = head_eye(framed=False)
    flipped = eye.EyeMap(emap.rows[::-1], np.asarray([(0.0, -v * 1000.0, h * 1000.0) for h, v in GRID[::-1]]))
    a, b = emap.head_frame(), flipped.head_frame()
    np.testing.assert_allclose(a.xy[::-1], b.xy, atol=1e-9)


def test_head_frame_refuses_a_plane_with_no_dorsoventral_extent():
    flat = eye.EyeMap(np.arange(4), [(0, 0, 0), (1000, 0, 0), (0, 0, 1000), (1000, 0, 1000)])
    with pytest.raises(ValueError, match="dorsoventral"):
        flat.head_frame()


def schedule(emap, types, **kw):
    opts = {"radii": [0.5, 1.0], "duration_ms": 100.0, "rate_hz": 100.0, "polarity": "both", **kw}
    return motion.motion_schedule(emap, types, **opts)


def test_outward_motion_drives_the_subtype_whose_preferred_direction_points_away_from_the_centre():
    for kind, (h, v) in {"a": (1, 0), "b": (-1, 0), "c": (0, 1), "d": (0, -1)}.items():
        emap, types = head_eye(kind=kind)
        rates = segment_rates(schedule(emap, types))
        assert rates[rows_at(emap, h, v)] == pytest.approx(100.0)  # edge moving straight along the preference
        assert rates.get(rows_at(emap, -h, -v), 0.0) == 0.0  # the opposite side sees the edge move against the preference


def test_inward_motion_selects_the_opposite_subtypes_at_the_same_position():
    emap, types = head_eye(kind="a")
    out = segment_rates(schedule(emap, types, direction=1))
    inn = segment_rates(schedule(emap, types, direction=-1))
    behind, front = rows_at(emap, 1, 0), rows_at(emap, -1, 0)
    assert out.get(behind, 0) > 0 and out.get(front, 0) == 0  # a likes outward at +h, silent at -h
    assert inn.get(behind, 0) == 0 and inn.get(front, 0) > 0  # reversed: now the -h side


def test_rate_is_base_rate_times_cosine_of_the_angle_to_the_edge_motion():
    emap, types = head_eye(kind="a")
    rates = segment_rates(schedule(emap, types, radii=[1.0, 1.0]))  # one wide band
    p = rows_at(emap, 2, 1)
    cos = 2 / np.hypot(2, 1)
    assert rates[p] == pytest.approx(100.0 * round(cos * motion.LEVELS) / motion.LEVELS)  # rates are quantised
    assert rates[p] > 0


def test_only_cells_inside_the_moving_edge_band_fire_in_each_window():
    emap, types = head_eye(kind="a")
    sched = schedule(emap, types, radii=[0.4, 1.0], duration_ms=100.0)
    xy = {int(r): np.hypot(*p) for r, p in zip(emap.rows, emap.xy)}
    for start, end, rows, _ in sched.segments:
        lo, hi = (0.0, 0.4) if start == 0.0 else (0.4, 1.0)
        assert all(lo < xy[int(r)] <= hi or (lo == 0.0 and xy[int(r)] <= hi) for r in rows)


def test_receding_is_the_looming_bands_in_reverse_order():
    emap, types = head_eye(kind="a")
    loom = schedule(emap, types, radii=[0.4, 1.0], polarity="both")
    rec = schedule(emap, types, radii=[0.4, 1.0], direction=-1)
    xy = {int(r): np.hypot(*p) for r, p in zip(emap.rows, emap.xy)}
    first_rec = [xy[int(r)] for s, _, rows, _ in rec.segments if s == 0.0 for r in rows]
    first_loom = [xy[int(r)] for s, _, rows, _ in loom.segments if s == 0.0 for r in rows]
    assert first_rec and min(first_rec) > 0.4 >= max(first_loom)  # receding starts at the outer band


def test_dark_disc_polarity_expansion_drives_off_cells_receding_drives_on_cells():
    n = len(GRID)
    emap, _ = head_eye()
    types = np.array(["T4a"] * (n // 2) + ["T5a"] * (n - n // 2))
    exp = segment_rates(schedule(emap, types, polarity="dark", direction=1))
    rec = segment_rates(schedule(emap, types, polarity="dark", direction=-1))
    t4 = {int(r) for r, t in zip(emap.rows, types) if t == "T4a"}
    assert exp and not (set(exp) & t4)  # expansion: leading OFF edge -> T5 only
    assert rec and set(rec) <= t4  # shrinking: trailing ON edge -> T4 only


def test_unknown_polarity_and_direction_are_refused():
    emap, types = head_eye()
    with pytest.raises(ValueError, match="polarity"):
        schedule(emap, types, polarity="grey")
    with pytest.raises(ValueError, match="direction"):
        schedule(emap, types, direction=0)


def test_no_drive_at_all_is_an_error():
    emap, types = head_eye(kind="b")  # a band at the very centre holds no cell
    with pytest.raises(ValueError, match="no motion"):
        schedule(emap, types, radii=[0.01, 0.02])
