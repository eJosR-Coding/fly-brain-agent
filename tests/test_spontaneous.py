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
