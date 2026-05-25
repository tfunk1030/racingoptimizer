"""Tests for left/right DE symmetry and static-RH platform penalties."""
from __future__ import annotations

from racingoptimizer.physics.setup_symmetry import (
    DE_SYMMETRY_SLAVES,
    apply_setup_symmetry,
    static_rh_platform_penalty,
)


def test_apply_setup_symmetry_mirrors_camber_and_dampers() -> None:
    setup = {
        "camber_fl_deg": -1.6,
        "camber_fr_deg": -2.4,
        "camber_rl_deg": -1.7,
        "camber_rr_deg": -1.3,
        "damper_lsc_fl": 4.0,
        "damper_lsc_fr": 7.0,
        "damper_hsc_rl": 2.0,
        "damper_hsc_rr": 6.0,
    }
    out = apply_setup_symmetry(setup)
    assert out["camber_fl_deg"] == -1.6
    assert out["camber_fr_deg"] == -1.6
    assert out["camber_rl_deg"] == -1.7
    assert out["camber_rr_deg"] == -1.7
    assert out["damper_lsc_fl"] == 4.0
    assert out["damper_lsc_fr"] == 4.0
    assert out["damper_hsc_rl"] == 2.0
    assert out["damper_hsc_rr"] == 2.0


def test_de_symmetry_slaves_exclude_right_side_params() -> None:
    assert "camber_fr_deg" in DE_SYMMETRY_SLAVES
    assert "camber_fl_deg" not in DE_SYMMETRY_SLAVES
    assert "damper_lsc_rr" in DE_SYMMETRY_SLAVES


def test_static_rh_penalty_zero_inside_envelope() -> None:
    readouts = {
        "setup_static_lf_ride_height_mm": 35.0,
        "setup_static_rf_ride_height_mm": 35.0,
        "setup_static_lr_ride_height_mm": 45.0,
        "setup_static_rr_ride_height_mm": 45.0,
    }
    assert static_rh_platform_penalty(readouts) == 0.0


def test_static_rh_penalty_heavy_below_floor() -> None:
    readouts = {
        "setup_static_lf_ride_height_mm": 26.0,
        "setup_static_rf_ride_height_mm": 26.0,
    }
    penalty = static_rh_platform_penalty(readouts)
    assert penalty > 6.0
