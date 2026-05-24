"""Unit tests for static ride height envelope warnings (audit 2026-05-23)."""
from __future__ import annotations

from racingoptimizer.cli.recommend import _static_ride_height_envelope_warnings


def test_static_rh_warning_outside_front_envelope() -> None:
    warnings = _static_ride_height_envelope_warnings(
        {"setup_static_lf_ride_height_mm": 85.0},
    )
    assert len(warnings) == 1
    assert "LF static ride height" in warnings[0]
    assert "85.0 mm" in warnings[0]
    assert "30-80" in warnings[0]


def test_static_rh_no_warning_inside_envelope() -> None:
    warnings = _static_ride_height_envelope_warnings(
        {
            "setup_static_lf_ride_height_mm": 45.0,
            "setup_static_lr_ride_height_mm": 50.0,
        },
    )
    assert warnings == []


def test_static_rh_empty_readouts_no_warnings() -> None:
    assert _static_ride_height_envelope_warnings({}) == []
