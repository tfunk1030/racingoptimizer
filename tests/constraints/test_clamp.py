from __future__ import annotations

from racingoptimizer.constraints import ClampResult, clamp, load_constraints


def test_clamp_above_acura_max() -> None:
    r = clamp(11.0, "rear_wing_angle_deg", "acura")
    assert r == ClampResult(value=10.0, was_clamped=True, bound=(6.0, 10.0), status="clamped")


def test_clamp_inside_acura_range() -> None:
    r = clamp(8.0, "rear_wing_angle_deg", "acura")
    assert r.value == 8.0
    assert r.was_clamped is False
    assert r.status == "ok"
    assert r.bound == (6.0, 10.0)


def test_clamp_uses_default_for_bmw() -> None:
    r = clamp(15.0, "rear_wing_angle_deg", "bmw")
    assert r.value == 15.0
    assert r.was_clamped is False
    assert r.status == "ok"
    assert r.bound == (12.0, 17.0)


def test_clamp_below_default_min() -> None:
    r = clamp(100.0, "tyre_cold_pressure_kpa", "ferrari")
    assert r.value == 152.0
    assert r.was_clamped is True
    assert r.status == "clamped"
    assert r.bound == (152.0, 220.0)


def test_clamp_unbounded_parameter() -> None:
    r = clamp(2000.0, "damper_lsc_fl", "bmw")
    assert r.value == 2000.0
    assert r.was_clamped is False
    assert r.bound is None
    assert r.status == "unbounded"


def test_clamp_unknown_parameter() -> None:
    r = clamp(1.0, "made_up_thing", "bmw")
    assert r.status == "unknown_parameter"
    assert r.value == 1.0
    assert r.was_clamped is False
    assert r.bound is None


def test_clamp_normalizes_raw_filename_prefix() -> None:
    r = clamp(11.0, "rear_wing_angle_deg", "acuraarx06gtp")
    assert r.value == 10.0
    assert r.was_clamped is True
    assert r.bound == (6.0, 10.0)
    assert r.status == "clamped"


def test_clamp_at_boundary_is_ok() -> None:
    assert clamp(10.0, "rear_wing_angle_deg", "acura").status == "ok"
    assert clamp(6.0, "rear_wing_angle_deg", "acura").status == "ok"


def test_clamp_accepts_explicit_table() -> None:
    table = load_constraints()
    r = clamp(20.0, "rear_wing_angle_deg", "bmw", table=table)
    assert r.value == 17.0
    assert r.was_clamped is True
