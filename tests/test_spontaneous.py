import numpy as np

from flybrain import body_groom, spontaneous
from flybrain.body_kinematic import KinematicBody, load_library

ROWS = {k: np.array([i]) for i, k in enumerate(("dng100", "dna02_left", "dna02_right", "groom", "dnp01"))}


def run(seed, secs=120.0, **kw):
    sp = spontaneous.Spontaneous(ROWS, seed=seed, **kw)
    seen = []
    for i in range(int(secs * 20)):
        out = sp.step(i * 50.0)
        seen.append(out)
    return sp, seen


def test_deterministic_and_covers_states():
    a, _ = run(1)
    b, _ = run(1)
    assert a.log == b.log
    assert {"rest", "walk", "groom"} <= {s for _, s in a.log}


def test_off_injects_nothing():
    _, seen = run(0, 10.0, mode="off")
    assert all(o["rates"] == {} and o["state"] == "off" for o in seen)


def test_states_inject_command_neurons_only():
    sp, seen = run(2)
    for o in seen:
        if o["state"] == "groom":
            assert set(o["rates"]) == {"groom"}
        if o["state"] == "turn_left":
            assert set(o["rates"]) == {"dng100", "dna02_left"}
    rows, hz = sp.inject({"dng100": 40.0, "missing": 5.0})
    assert list(rows) == [0] and list(hz) == [40.0]


def test_groom_and_head_kinematics_keep_legs_rigid():
    body = KinematicBody(load_library(), bounds=(-25, 25, -25, 25), height_mm=50)
    body.set_groom(True)
    body.set_head(9.0, 0.5)
    for _ in range(6):
        out = body.advance(50, [0, 0], pace=0, turn=0)
    assert out["groom"] and body.groom_w == 1.0
    lib = body.lib
    f = out["frames"][-1][:, :3].astype(float)
    tip, cox = lib.bodies.index("lf_tarsus5"), lib.bodies.index("lf_coxa")
    assert abs(np.linalg.norm(f[tip] - f[cox]) - np.linalg.norm(lib.stand.pos[tip] - lib.stand.pos[cox])) < 5
    assert f[tip][2] > 300  # the tip is lifted in front of the head
    assert out["head_yaw_deg"][-1] > 5.0
    body.set_groom(False)
    for _ in range(6):
        out = body.advance(50, [0, 0], pace=0, turn=0)
    assert not out["groom"] and body.groom_w == 0.0


def test_head_flying_is_zero():
    h = body_groom.Head()
    h.command(10.0, 1.0, False)
    for _ in range(100):
        h.step(0.005, 0.0, False)
    assert h.deg > 15
    for _ in range(100):
        h.step(0.005, 0.0, True)
    assert abs(h.deg) < 0.5


def _ctx(objects, touched=(), x=0.0, y=0.0, yaw=0.0):
    return {"x_mm": x, "y_mm": y, "z_um": 1000.0, "yaw_deg": yaw, "objects": objects, "touched": list(touched)}


def _sp():
    rows = {k: np.arange(2) for k in ("dng100", "dna02_left", "dna02_right", "groom", "dnp01")}
    return spontaneous.Spontaneous(rows, seed=1)


def test_seek_turns_toward_sugar_and_feed_stops_then_cooldown():
    sp = _sp()
    sugar = {"id": "sugar1", "kind": "sugar", "pos_mm": [5.0, 8.0, 0.75]}  # about 58 deg to the left
    out = sp.step(0.0, ctx=_ctx([sugar]))
    assert out["state"] == "seek" and "dna02_left" in out["rates"] and "dna02_right" not in out["rates"] and "dng100" not in out["rates"]  # 58 deg off: turn first, no walking
    assert sp.food["target"] == "sugar1"
    out = sp.step(50.0, ctx=_ctx([sugar], touched=["sugar1"]))
    assert out["state"] == "feed" and out["rates"] == {}
    for i in range(1, 400):
        out = sp.step(50.0 * (i + 1), ctx=_ctx([sugar]))
        if out["state"] != "feed":
            break
    assert 5.0 <= 50.0 * i / 1000 <= 15.5 and out["state"] != "seek"  # cooldown: the cube is ignored


def test_coffee_turns_away_and_touch_pauses():
    sp = _sp()
    coffee = {"id": "coffee1", "kind": "coffee", "pos_mm": [6.0, 0.0, 0.75]}
    out = sp.step(0.0, ctx=_ctx([coffee]))
    assert out["state"] == "avoid" and ("dna02_left" in out["rates"] or "dna02_right" in out["rates"])
    out = sp.step(50.0, ctx=_ctx([coffee], touched=["coffee1"]))
    assert out["state"] == "feed" and out["rates"] == {}


def test_seek_from_wall_and_ceiling_and_no_false_wall_contact():
    import numpy as np
    from flybrain import live_world, spontaneous as sp

    cube = {"id": "sugar1", "kind": "sugar", "pos_mm": [10.0, 0.0, 0.75], "size_mm": 1.5}
    # on the x_max wall (n = -x), heading up (+z): sugar present -> seek, bearing points down (180 deg)
    ctx = {"x_mm": 30.0, "y_mm": 0.0, "z_um": 8000.0, "yaw_deg": 0.0, "objects": [cube], "touched": [], "n": np.array([-1.0, 0, 0]),
           "f": np.array([0.0, 0, 1.0]), "bounds_mm": [-30, 30, -30, 30]}
    s = sp.Spontaneous(rows={})
    want, tgt, bearing, _ = s._rule(0.0, ctx)
    assert want == "seek" and tgt is cube and abs(abs(bearing) - 180.0) < 1e-6
    # ceiling: heads to the nearest wall (x_max here, +x)
    ctx2 = {**ctx, "x_mm": 25.0, "z_um": 30000.0, "n": np.array([0.0, 0, -1.0]), "f": np.array([1.0, 0, 0])}
    want, _, bearing, _ = s._rule(0.0, ctx2)
    assert want == "seek" and abs(bearing) < 1e-6
    # a fly on a wall does not "touch" a floor cube by 2D projection
    w = live_world.World((-30, 30, -30, 30))
    o = w.add("sugar", 10.0, 0.0)
    assert w.touching(np.array([[9.5e3, 0.0, 8000.0]]), np.array([0.0])) == []
    assert w.touching(np.array([[9.5e3, 0.0, 900.0]]), np.array([0.0])) == [o["id"]]
