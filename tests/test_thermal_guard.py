"""Thermal guard: no new GPU work above the limit (fake telemetry and clock)."""

from flybrain.gpu_simulation import ThermalGuard


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def guard(temps, **kw):
    clock, reads = Clock(), []
    it = iter(temps)

    def read():
        reads.append(clock.t)
        t = next(it)
        return None if t is None else {"temperature_c": t}

    return ThermalGuard(read=read, clock=clock, **kw), clock, reads


def test_cool_gpu_is_allowed_and_reads_are_throttled():
    g, clock, reads = guard([60.0, 61.0, 62.0], interval_s=3.0)
    assert g.allow() and g.allow() and g.allow()
    assert len(reads) == 1  # three calls inside the interval, one nvidia-smi
    clock.t += 3.5
    assert g.allow() and len(reads) == 2


def test_above_the_limit_refuses_until_it_cools_with_hysteresis():
    g, clock, _ = guard([81.0, 78.0, 74.0], limit_c=80.0, interval_s=1.0, resume_margin_c=5.0)
    assert not g.allow() and g.hot and g.refusals == 1
    clock.t += 2
    assert not g.allow()  # 78 C: below the limit but not 5 C below it
    clock.t += 2
    assert g.allow() and not g.hot  # 74 C


def test_exactly_at_the_limit_is_allowed_and_above_is_not():
    g, clock, _ = guard([80.0, 80.1], limit_c=80.0, interval_s=1.0)
    assert g.allow()
    clock.t += 2
    assert not g.allow()


def test_unknown_telemetry_never_blocks_and_is_reported():
    g, _, _ = guard([None])
    assert g.allow() and g.status()["temperature_c"] is None


def test_status_reports_limit_temperature_and_refusals():
    g, _, _ = guard([85.0], limit_c=80.0)
    g.allow()
    s = g.status()
    assert s["limit_c"] == 80.0 and s["temperature_c"] == 85.0 and s["hot"] is True and s["refusals"] == 1


def test_a_failing_reader_is_treated_as_unknown():
    clock = Clock()

    def boom():
        raise OSError("no nvidia-smi")

    g = ThermalGuard(read=boom, clock=clock)
    assert g.allow() and g.status()["temperature_c"] is None
