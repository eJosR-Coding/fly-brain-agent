"""Eye map over synthetic point clouds (no real atlas)."""

import numpy as np
import pytest

from flybrain import eye
from flybrain.atlas import load_atlas
from flybrain.eye import EyeMap


def grid_cloud(n=21):
    """A flat n x n grid in the xy plane (z tiny noise-free), rows 100..: eye radius = corner distance."""
    xs = np.linspace(-10, 10, n)
    pts = np.array([(x, y, 0.0) for x in xs for y in xs])
    return np.arange(100, 100 + len(pts)), pts


def test_disc_membership_is_a_circle_around_the_centroid():
    rows, pts = grid_cloud()
    m = EyeMap(rows, pts)
    inside = m.disc(0.5)
    assert len(inside) and len(inside) < len(rows)
    r = np.linalg.norm(m.xy[np.isin(m.rows, inside)], axis=1)
    out = np.linalg.norm(m.xy[~np.isin(m.rows, inside)], axis=1)
    assert r.max() <= 0.5 < out.min()


def test_disc_grows_monotonically_and_nests():
    m = EyeMap(*grid_cloud())
    discs = [set(m.disc(r)) for r in (0.1, 0.3, 0.6, 1.0)]
    assert all(a < b for a, b in zip(discs, discs[1:]))
    assert discs[-1] == set(m.rows)  # radius 1 = the whole eye


def test_pca_ignores_axis_orientation_and_depth():
    rows, pts = grid_cloud()
    rot = pts[:, [2, 0, 1]] + [5, 7, 9]  # the same eye in other axes
    a, b = EyeMap(rows, pts), EyeMap(rows, rot)
    assert set(a.disc(0.4)) == set(b.disc(0.4))


def test_centre_choice_moves_the_disc():
    m = EyeMap(*grid_cloud())
    corner = m.xy[0]
    near = m.disc(0.2, centre=corner)
    assert 100 in near and set(near).isdisjoint(m.disc(0.2))


def test_empty_disc_is_empty_and_bad_input_is_refused():
    m = EyeMap(*grid_cloud(6))  # even grid: nobody at the exact centroid
    assert len(m.disc(0.0)) == 0
    with pytest.raises(ValueError, match="radius"):
        m.disc(-1)
    with pytest.raises(ValueError, match="centre"):
        m.disc(0.5, centre=[1, 2, 3])
    with pytest.raises(ValueError, match="at least 3"):
        EyeMap([1, 2], [[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError, match="identical"):
        EyeMap([1, 2, 3], np.ones((3, 3)))


HEADER = "root_id\tpos_x\tpos_y\tpos_z\tsuper_class\tcell_class\tcell_type\ttop_nt\tside\tflow"


def eye_atlas(tmp_path):
    lines = [HEADER]
    i = 1
    for side, shift in (("left", 0), ("right", 100000)):
        for x in range(5):
            for y in range(5):
                lines.append(f"{i}\t{shift + x * 1000}\t{y * 1000}\t0\tsensory\t\tR1-6\thistamine\t{side}\tafferent")
                i += 1
    lines.append(f"{i}\t0\t0\t0\tcentral\t\tDNp01\t\tright\tefferent")
    p = tmp_path / "eyes.tsv"
    p.write_text("\n".join(lines) + "\n")
    return load_atlas(p)


def test_eye_map_filters_by_side_and_cell_type(tmp_path):
    atlas = eye_atlas(tmp_path)
    right, left = eye.eye_map(atlas), eye.eye_map(atlas, side="left")
    assert len(right) == len(left) == 25
    assert set(right.rows) == set(range(25, 50)) and set(left.rows) == set(range(25))
    with pytest.raises(ValueError, match="side"):
        eye.eye_map(atlas, side="middle")
    with pytest.raises(ValueError, match="no DNp01 neurons on the left"):
        eye.eye_map(atlas, side="left", cell_type="DNp01")


def test_looming_radii_are_slow_then_accelerating_and_end_near_full_eye():
    radii = eye.looming_radii(500, 10)
    steps = np.diff([0, *radii])
    assert all(b > a for a, b in zip(radii, radii[1:]))
    assert steps[-1] > 5 * steps[0] and radii[0] < 0.1 and 0.85 < radii[-1] <= 1.0


def test_disc_schedule_one_segment_per_window_and_skips_empty_discs():
    m = EyeMap(*grid_cloud(20))  # even grid: the radius-0 disc is empty
    sched = eye.disc_schedule(m, [0.0, 0.3, 1.0], 300, 50.0)
    assert [(s[0], s[1], s[3]) for s in sched.segments] == [(100.0, 200.0, 50.0), (200.0, 300.0, 50.0)]
    assert len(sched.segments[0][2]) < len(sched.segments[1][2]) == len(m)
    with pytest.raises(ValueError, match="empty"):
        eye.disc_schedule(m, [0.0, 0.0], 100, 50.0)


def test_schedule_drive_is_rate_times_active_time_summed_over_neurons():
    m = EyeMap(*grid_cloud())
    sched = eye.disc_schedule(m, [0.3, 1.0], 200, 50.0)
    n0, n1 = (len(s[2]) for s in sched.segments)
    assert eye.schedule_drive(sched) == pytest.approx(50.0 * 0.1 * (n0 + n1))
